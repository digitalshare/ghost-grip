"""GhostGripFlow: scene → references → techniques → shot list ⇄ robot feasibility → approved plan.

Run from the CLI:  uv run python -m ghost_grip.crew.flow scenes/closet.txt
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import yaml
from crewai import Agent
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from ghost_grip import config, llm, retrieval
from ghost_grip.clips import find_clip
from ghost_grip.robot import arm
from ghost_grip.robot.planner import plan_motion
from ghost_grip.schemas import Reference, References, RobotPlan, Scene, ShotList
from ghost_grip.youcom import read_page, search_film_references

AGENTS = yaml.safe_load((Path(__file__).parent / "agents.yaml").read_text())

Emit = Callable[[str, dict], None]

COORDS = """Set coordinates, metres, Z-up: x runs left to right across the set as seen from the camera side; y runs from the
open fourth wall (y=0) to the back wall (y=depth); z is height above the floor."""

RIG_RULES = f"""THE RIG (hard physical limits, checked by a simulator after you answer):
- A 6-axis arm on a pedestal (top at z={arm.PEDESTAL_HEIGHT} m) riding a straight {arm.RAIL_LENGTH:.0f} m rail along x at y = rig.rail_y,
  from x = rig.rail_x_start to rig.rail_x_start + {arm.RAIL_LENGTH:.0f}. You choose rig.rail_x_start and rig.rail_y.
- The camera can only be within about {arm.REACH} m of the pedestal top: horizontal distance from the rail line <= 1.0 m at
  eye height, camera height between 0.3 and 1.9 m. Put the rail about 0.7-0.9 m (in y) before where the camera needs to be.
- The rail and pedestal sit on the floor: keep the rail line clear of actor marks and props.
- Keep every camera position at least 0.8 m (horizontal) from every actor mark and outside every prop box, and the whole
  arm must not reach across an actor to get there.
- Axis speed limits: rail 1.5 m/s, base and shoulder 120 deg/s, wrists 200 deg/s. A pan of more than ~90 degrees needs
  at least 1 s. Snap moves need an easing keyframe ~0.2-0.3 s before the snap.
- Each shot is a separate take (the arm resets between shots); keyframe t starts at 0 in every shot."""


class GhostGripState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    scene_text: str = ""
    scene: Scene | None = None
    references: list[Reference] = []
    techniques: list[dict] = []
    shot_list: ShotList | None = None
    plan: RobotPlan | None = None
    revision: int = 0
    rounds: list[dict] = []
    approved: bool = False
    notes: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0


def _agent(key: str, **kwargs) -> Agent:
    cfg = AGENTS[key]
    return Agent(role=cfg["role"], goal=cfg["goal"], backstory=cfg["backstory"], llm=llm.crew_llm(), verbose=False, **kwargs)


def compose_prompt(s: GhostGripState) -> str:
    """The Cinematographer's brief: set plan, grounded references, retrieved techniques, and any rig violations to fix."""
    refs = "\n".join(f"- {r.film} — {r.moment} ({r.technique}) {r.url}: {r.why}" for r in s.references)
    techs = "\n".join(f"- {t['name']}: {t['move']} Lens {t['lens_mm']} mm, {t['duration_s']} s. Rig: {t['rig_notes']}" for t in s.techniques)
    prompt = (f"{COORDS}\n\n{RIG_RULES}\n\nSCENE (set plan):\n{s.scene.model_dump_json()}\n\n"
              f"GROUNDED REFERENCES (cite one per shot, use its exact URL):\n{refs}\n\nCANDIDATE TECHNIQUES:\n{techs}\n\n"
              "Actors with a path walk between their marks in scene time; a shot that starts at start_t sees them where they are at "
              "start_t + t, so follow moves should track the walker and keep clear of where they will be.\n"
              "Write 3 or 4 shots that together cover the beats and build to the scare. Give each shot start_t (the scene time it "
              "covers, matching its beat) and 2-6 keyframes, "
              "real lens choices, look_at points on the actor/prop that matters (around head height 1.5-1.6 m for standing actors). "
              "director_notes: 2-3 sentences on the visual plan.")
    if s.shot_list and s.plan:
        prompt += ("\n\nYOUR PREVIOUS SHOT LIST:\n" + s.shot_list.model_dump_json() +
                   "\n\nTHE RIG SIMULATOR REJECTED IT WITH THESE VIOLATIONS. Fix every one, keep what already worked, and say what "
                   "you changed in director_notes:\n" + "\n".join(f"- [{v.shot_id} @ {v.t}s] {v.kind}: {v.detail} Fix: {v.suggestion}" for v in s.plan.violations))
    return prompt


class GhostGripFlow(Flow[GhostGripState]):
    emit: Emit = staticmethod(lambda kind, data: print(f"[{kind}] {json.dumps(data)[:240]}"))

    def _step(self, name: str, status: str, **extra):
        self.emit("step", {"step": name, "status": status, **extra})

    @start()
    async def parse_scene(self):
        self.state.started_at = time.time()
        self._step("scene", "running", label="Reading the scene")
        system = ("You are a first assistant director breaking down a horror scene for a motion-control shoot. "
                  "Turn the scene into a concrete set plan. " + COORDS +
                  " Place actors and props at believable marks inside the set (not on the fourth wall line y<0.8). "
                  "List each person or figure exactly once: x,y is where they stand at t=0 and path holds their later marks "
                  "(scene time, x, y) if they walk; leave path empty if they stay. Use kind 'entity' for ghosts and apparitions. "
                  "The title is a short evocative name for the scene (2-5 words). "
                  "Give 3-6 beats with timestamps that cover the scene, ending on the scare.")
        scene = await asyncio.to_thread(
            llm.structured, Scene, system, self.state.scene_text,
            on_token=lambda tok: self.emit("token", {"channel": "scene", "text": tok}))
        title = scene.title.strip(" ,.;:-\"'")
        m = re.match(r"\s*((?:INT|EXT)\.?\s+[^.]+)", self.state.scene_text)  # slugline, e.g. "INT. CHILD'S BEDROOM - NIGHT"
        heading = m.group(1).strip()[:60] if m else ""
        scene.title = title if len(title) >= 3 and title.upper() not in ("INT", "EXT") else heading or " ".join(scene.logline.split()[:5]).strip(" ,.;:") or "Untitled scene"
        self.state.scene = scene
        self.emit("scene", scene.model_dump())
        self._step("scene", "done", label=f"{scene.title}: {len(scene.beats)} beats, {len(scene.actors)} actors, {len(scene.props)} props")

    @listen(parse_scene)
    async def research(self):
        self._step("research", "running", label="Scouting real film references (You.com)")
        scene = self.state.scene
        scout = _agent("reference_scout", tools=[search_film_references, read_page], max_iter=8)
        beats = "\n".join(f"- t={b.t:.0f}s: {b.description}" for b in scene.beats)
        prompt = (f"Scene: {scene.title}. Mood: {scene.mood}.\nBeats:\n{beats}\n\n"
                  "Find 3 or 4 real horror (or thriller) film moments whose CAMERA MOVE would serve these beats. "
                  "Use search_film_references with specific queries (film + scene + camera technique). "
                  "Only cite URLs that appeared in your search results, and each film at most once. For each: film (title and year), "
                  "moment (under 15 words), technique (a few words), url, why (one sentence on how it serves a beat).")
        out = await scout.kickoff_async(prompt, response_format=References)
        refs = out.pydantic if isinstance(out.pydantic, References) else await asyncio.to_thread(
            llm.structured, References, "Extract the film references from this research note as JSON.", out.raw)
        seen: set[str] = set()
        self.state.references = [Reference(**r.model_dump()) for r in refs.references
                                 if r.url.startswith("http") and not (r.url in seen or seen.add(r.url))]
        self.emit("references", {"references": [r.model_dump() for r in self.state.references]})
        self._step("research", "done", label=f"{len(self.state.references)} grounded references")

    @listen(research)
    async def find_clips(self):
        """Deterministic: a verified, playable YouTube clip for each reference (You.com search + oEmbed), so the
        audience can watch where every camera move comes from."""
        self._step("clips", "running", label="Finding the original clips (You.com + YouTube)")

        def one(ref: Reference):
            try:
                ref.clip = find_clip(ref)
            except Exception as e:  # a missing clip must never sink the run
                self.emit("log", {"text": f"clip search failed for {ref.film}: {type(e).__name__}"})

        await asyncio.gather(*(asyncio.to_thread(one, r) for r in self.state.references))
        self.emit("clips", {"clips": [r.clip.model_dump() if r.clip else None for r in self.state.references]})
        found = sum(1 for r in self.state.references if r.clip)
        self._step("clips", "done", label=f"{found} of {len(self.state.references)} reference clips found")

    @listen(find_clips)
    async def retrieve(self):
        self._step("retrieve", "running", label="Matching beats to camera techniques (Parasail embeddings)")
        beats = [f"{b.description} ({self.state.scene.mood})" for b in self.state.scene.beats]
        self.state.techniques = await asyncio.to_thread(retrieval.top_techniques, beats, 6)
        self.emit("techniques", {"techniques": [{k: t[k] for k in ("name", "score", "matched_beat", "effect")} for t in self.state.techniques]})
        self._step("retrieve", "done", label=", ".join(t["name"] for t in self.state.techniques[:4]))

    @listen(or_(retrieve, "revise"))
    async def compose(self):
        rnd = self.state.revision
        self._step("compose", "running", round=rnd, label="Designing the shot list" if rnd == 0 else f"Revising after rig feedback (round {rnd})")
        s = self.state
        prompt = compose_prompt(s)
        # Streamed, schema-locked decoding: the shot list appears in the UI as it is written (CrewAI's
        # non-streaming structured path left the UI silent for minutes on the same brief).
        cfg = AGENTS["cinematographer"]
        system = f"You are the {cfg['role']}. Goal: {cfg['goal']}. {cfg['backstory']}"
        shot_list = await asyncio.to_thread(
            llm.structured, ShotList, system, prompt,
            on_token=lambda tok: self.emit("token", {"channel": f"compose-{rnd}", "text": tok}))
        s.shot_list = shot_list
        self.emit("shot_list", {"round": rnd, **shot_list.model_dump()})
        self._step("compose", "done", round=rnd, label=f"{len(shot_list.shots)} shots, rail at y={shot_list.rig.rail_y:.2f}")

    @listen(compose)
    async def plan_robot(self):
        self._step("robot", "running", round=self.state.revision, label="Simulating the arm: reach, speed, clearance")
        t0 = time.time()
        plan = await asyncio.to_thread(plan_motion, self.state.scene, self.state.shot_list)
        self.state.plan = plan
        self.state.rounds.append({"round": self.state.revision, "violations": [v.model_dump() for v in plan.violations],
                                  "shot_list": self.state.shot_list.model_dump()})
        self.emit("robot_plan", {"round": self.state.revision, **plan.model_dump(), "feasible": plan.feasible, "solve_s": round(time.time() - t0, 2)})
        label = "Every move is feasible" if plan.feasible else f"{len(plan.violations)} violations"
        self._step("robot", "done" if plan.feasible else "warn", round=self.state.revision, label=label)

    @router(plan_robot)
    def gate(self):
        if self.state.plan.feasible:
            self.state.approved = True
            return "approved"
        if self.state.revision < config.MAX_REVISIONS:
            self.state.revision += 1
            return "revise"
        return "approved"  # out of revisions: hand over with violations clearly flagged

    @listen("approved")
    async def wrap(self):
        s = self.state
        self._step("notes", "running", label="Director's notes")
        shots = "\n".join(f"{sh.id} {sh.name} ({sh.technique}, {sh.keyframes[0].lens_mm:.0f}mm, {sh.keyframes[-1].t:.1f}s) — {sh.intent} [ref: {sh.reference_title}]" for sh in s.shot_list.shots)
        open_issues = "\n".join(f"- {v.shot_id}: {v.detail}" for v in s.plan.violations) or "none"
        system = "You are the director of a horror short, briefing the crew in a few punchy sentences. No markdown headers."
        user = (f"Scene: {s.scene.title} — {s.scene.logline}\nShots:\n{shots}\nRig revisions needed: {s.revision}\n"
                f"Open rig issues: {open_issues}\nBrief the crew in under 120 words: the scare, the rhythm of the shots, and one safety note.")
        s.notes = await asyncio.to_thread(llm.stream_text, system, user, lambda tok: self.emit("token", {"channel": "notes", "text": tok}))
        s.elapsed_s = round(time.time() - s.started_at, 1)
        config.RUNS_DIR.mkdir(exist_ok=True)
        (config.RUNS_DIR / f"{s.id}.json").write_text(s.model_dump_json())
        self._step("notes", "done", label="Plan approved" if s.plan.feasible else "Plan handed over with open issues")
        self.emit("done", {"run_id": s.id, "feasible": s.plan.feasible, "revisions": s.revision, "elapsed_s": s.elapsed_s})


def run(scene_text: str, emit: Emit | None = None) -> GhostGripState:
    flow = GhostGripFlow()
    if emit:
        flow.emit = emit
    flow.kickoff(inputs={"scene_text": scene_text})
    return flow.state


if __name__ == "__main__":
    text = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else sys.stdin.read()
    state = run(text)
    print(f"\nrun {state.id}: feasible={state.plan.feasible} revisions={state.revision}")
    for v in state.plan.violations:
        print(" -", v.shot_id, v.kind, v.detail)
