"""FastAPI app: start runs, stream their events over WebSocket, replay saved runs, dispatch call sheets via One.

Run:  uv run uvicorn ghost_grip.server:app --port 8017
"""
from __future__ import annotations

import asyncio
import re
import threading
import traceback
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ghost_grip import config
from ghost_grip.crew import flow as gg_flow

app = FastAPI(title="Ghost Grip")
WEB = config.ROOT / "web"
SCENES = config.ROOT / "scenes"


@app.middleware("http")
async def no_stale_assets(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class Channel:
    """An append-only event log for one background job; WebSocket subscribers get the backlog then live events."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.id = uuid.uuid4().hex[:8]
        self.loop = loop
        self.events: list[dict] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.finished = False

    def emit(self, kind: str, data: dict):  # called from worker threads
        event = {"type": kind, **data}
        self.loop.call_soon_threadsafe(self._publish, event)

    def _publish(self, event: dict):
        self.events.append(event)
        if event["type"] in ("done", "error", "dispatched"):
            self.finished = True
        for q in self.subscribers:
            q.put_nowait(event)


CHANNELS: dict[str, Channel] = {}


def start_job(target, *args) -> Channel:
    ch = Channel(asyncio.get_running_loop())
    CHANNELS[ch.id] = ch

    def work():
        try:
            target(*args, ch.emit)
        except Exception as e:
            traceback.print_exc()
            ch.emit("error", {"message": f"{type(e).__name__}: {e}"})

    threading.Thread(target=work, daemon=True).start()
    return ch


class RunRequest(BaseModel):
    scene_text: str


class DispatchRequest(BaseModel):
    shoot_start: datetime
    to: str = ""
    slack_channel: str = ""
    discord: bool = False


@app.get("/api/health")
def health():
    return {"youcom": bool(config.YDC_API_KEY), "parasail": bool(config.PARASAIL_API_KEY), "one": bool(config.ONE_SECRET),
            "model": config.PARASAIL_MODEL, "call_sheet_to": config.CALL_SHEET_TO, "slack_channel": config.SLACK_CHANNEL, "discord_channel": config.DISCORD_CHANNEL_ID}


@app.get("/api/scenes")
def scenes():
    return [{"name": p.stem, "text": p.read_text().strip()} for p in sorted(SCENES.glob("*.txt"))]


@app.post("/api/runs")
async def create_run(req: RunRequest):
    if len(req.scene_text.strip()) < 20:
        raise HTTPException(400, "Describe the scene in at least a sentence.")
    ch = start_job(lambda text, emit: gg_flow.run(text, emit), req.scene_text)
    return {"channel": ch.id}


@app.get("/api/runs")
def list_runs():
    out = []
    for p in sorted(config.RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        try:
            st = gg_flow.GhostGripState.model_validate_json(p.read_text())
            out.append({"id": st.id, "title": st.scene.title, "feasible": st.plan.feasible, "revisions": st.revision,
                        "saved": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")})
        except Exception:
            continue
    return out


def _load_state(run_id: str) -> gg_flow.GhostGripState:
    path = config.RUNS_DIR / f"{run_id}.json"
    if not re.fullmatch(r"[0-9a-f-]{8,36}", run_id) or not path.exists():
        raise HTTPException(404, "No such run")
    return gg_flow.GhostGripState.model_validate_json(path.read_text())


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _load_state(run_id).model_dump()


@app.get("/api/runs/{run_id}/clips")
async def get_clips(run_id: str):
    """Verified clips for a saved run's references (cached; the run file itself is never rewritten)."""
    from ghost_grip.clips import find_clip
    state = _load_state(run_id)

    def one(ref):
        try:
            clip = ref.clip or find_clip(ref)
        except Exception:
            return None
        return clip.model_dump() if clip else None

    return await asyncio.gather(*(asyncio.to_thread(one, r) for r in state.references))


@app.get("/api/runs/{run_id}/rounds/{n}")
def get_round(run_id: str, n: int):
    """Re-simulate one revision round's shot list (deterministic), so the UI can replay the rejected moves."""
    from ghost_grip.robot.planner import plan_motion
    from ghost_grip.schemas import ShotList
    state = _load_state(run_id)
    rnd = next((r for r in state.rounds if r["round"] == n), None)
    if rnd is None:
        raise HTTPException(404, "No such round")
    key = (run_id, n)
    if key not in ROUND_CACHE:
        plan = plan_motion(state.scene, ShotList.model_validate(rnd["shot_list"]))
        ROUND_CACHE[key] = {"round": n, "shot_list": rnd["shot_list"], **plan.model_dump(), "feasible": plan.feasible}
    return ROUND_CACHE[key]


ROUND_CACHE: dict[tuple[str, int], dict] = {}  # saved runs never change, so a round's simulation can be reused


def program_csv(state: gg_flow.GhostGripState) -> str:
    """Rig program: one row per 40 ms sample — rail metres, joint degrees, camera position, lens."""
    import math
    rows = ["shot,t_s,scene_t_s,rail_m,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg,cam_x_m,cam_y_m,cam_z_m,lens_mm"]
    for m in state.plan.shots:
        for k, q in enumerate(m.joints):
            t = min(k * m.dt, m.duration)
            joints = ",".join(f"{math.degrees(v):.3f}" for v in q[1:])
            cam = ",".join(f"{v:.3f}" for v in m.cam_pos[k])
            rows.append(f"{m.shot_id},{t:.2f},{m.start_t + t:.2f},{q[0]:.4f},{joints},{cam},{m.lens_mm[k]:.1f}")
    return "\n".join(rows) + "\n"


@app.get("/api/runs/{run_id}/program.csv")
def get_program(run_id: str):
    state = _load_state(run_id)
    name = re.sub(r"[^a-z0-9]+", "-", state.scene.title.lower()).strip("-") or "scene"
    return PlainTextResponse(program_csv(state), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="ghost-grip-{name}.csv"'})


@app.get("/api/runs/{run_id}/call-sheet")
def get_call_sheet(run_id: str, shoot_start: datetime | None = None):
    from ghost_grip.one_actions import call_sheet
    return {"text": call_sheet(_load_state(run_id), shoot_start or datetime.now().replace(minute=0, second=0, microsecond=0))}


@app.post("/api/runs/{run_id}/dispatch")
async def dispatch(run_id: str, req: DispatchRequest):
    state = _load_state(run_id)
    if req.to and "@" not in req.to:
        raise HTTPException(400, "That crew email address looks wrong.")
    if not (req.to or req.slack_channel.strip() or req.discord):
        raise HTTPException(400, "Give a Slack channel or a crew email.")

    def job(emit):
        from ghost_grip.one_actions import run_producer
        report = run_producer(state, req.shoot_start, to=req.to, slack_channel=req.slack_channel.strip(),
                              discord_channel=config.DISCORD_CHANNEL_ID if req.discord else "", emit=emit)
        emit("dispatched", {"report": report})

    ch = start_job(job)
    return {"channel": ch.id}


@app.websocket("/ws/{channel_id}")
async def stream(ws: WebSocket, channel_id: str):
    await ws.accept()
    ch = CHANNELS.get(channel_id)
    if ch is None:
        await ws.send_json({"type": "error", "message": "Unknown channel"})
        await ws.close()
        return
    q: asyncio.Queue = asyncio.Queue()
    backlog = list(ch.events)
    ch.subscribers.add(q)
    try:
        for e in backlog:
            await ws.send_json(e)
        if ch.finished and backlog:
            return
        while True:
            e = await q.get()
            await ws.send_json(e)
            if e["type"] in ("done", "error", "dispatched"):
                return
    except WebSocketDisconnect:
        pass
    finally:
        ch.subscribers.discard(q)
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
