"""Turn a shot list into a sampled, checked robot trajectory.

Pure numpy, no LLM: every violation it reports is a measured fact the Cinematographer
agent has to design around.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ghost_grip.robot import arm
from ghost_grip.robot.collision import Box, Capsule, actor_gap, check_pose
from ghost_grip.schemas import Keyframe, RigPlacement, RobotPlan, Scene, Shot, ShotList, ShotMotion, Violation

DT = 0.04  # 25 Hz
POS_TOL = 0.02
ANG_TOL = 0.05
ACTOR_RADIUS = 0.25
WALL = 0.1


def actors_at(scene: Scene, t: float) -> list[Capsule]:
    """Actor capsules where the blocking puts them at scene time t."""
    caps = []
    for a in scene.actors:
        x, y = a.position(t)
        caps.append(Capsule(a.name, np.array([x, y, ACTOR_RADIUS]), np.array([x, y, max(a.height - ACTOR_RADIUS, ACTOR_RADIUS)]), ACTOR_RADIUS))
    return caps


def set_boxes(scene: Scene) -> list[Box]:
    w, d, h = scene.set.width, scene.set.depth, scene.set.height
    boxes = [
        Box("left wall", np.array([-WALL, 0, 0]), np.array([0, d, h])),
        Box("right wall", np.array([w, 0, 0]), np.array([w + WALL, d, h])),
        Box("back wall", np.array([0, d, 0]), np.array([w, d + WALL, h])),
    ]
    for p in scene.props:
        boxes.append(Box(p.name, np.array([p.x - p.width / 2, p.y - p.depth / 2, 0]), np.array([p.x + p.width / 2, p.y + p.depth / 2, p.height])))
    return boxes


def _hermite(ts: np.ndarray, vals: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Piecewise cubic Hermite through keyframes with Catmull-Rom tangents; starts and ends at rest."""
    n = len(ts)
    tangents = np.zeros_like(vals)
    for i in range(1, n - 1):
        tangents[i] = (vals[i + 1] - vals[i - 1]) / (ts[i + 1] - ts[i - 1])
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, n - 2)
    t0, t1 = ts[idx], ts[idx + 1]
    h = (t1 - t0)[:, None]
    u = ((t - t0) / (t1 - t0))[:, None]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    return h00 * vals[idx] + h10 * h * tangents[idx] + h01 * vals[idx + 1] + h11 * h * tangents[idx + 1]


def _clean_keyframes(shot: Shot) -> tuple[list[Keyframe], list[Violation]]:
    kfs = sorted(shot.keyframes, key=lambda k: k.t)
    issues = []
    if not kfs:
        return [], [Violation(shot_id=shot.id, kind="bad_input", t=0, detail="Shot has no keyframes.", suggestion="Give every shot at least two keyframes.")]
    if len(kfs) == 1:  # a locked-off shot: hold the pose for two seconds
        kfs = [kfs[0], kfs[0].model_copy(update={"t": kfs[0].t + 2.0})]
    deduped = [kfs[0]]
    for k in kfs[1:]:
        if k.t - deduped[-1].t < 0.1 - 1e-6:
            issues.append(Violation(shot_id=shot.id, kind="bad_input", t=k.t, detail=f"Keyframes at t={deduped[-1].t:.2f}s and t={k.t:.2f}s are less than 0.1 s apart.", suggestion="Space keyframes at least 0.3 s apart; a physical rig cannot teleport."))
            continue
        deduped.append(k)
    return deduped, issues


def _fmt(p) -> str:
    return "(" + ", ".join(f"{v:.2f}" for v in p) + ")"


def plan_shot(shot: Shot, rig: arm.Rig, scene: Scene, boxes: list[Box]) -> tuple[ShotMotion | None, list[Violation]]:
    kfs, violations = _clean_keyframes(shot)
    if len(kfs) < 2:
        return None, violations
    ts = np.array([k.t for k in kfs]) - kfs[0].t
    duration = float(ts[-1])
    samples = np.append(np.arange(0.0, duration, DT), duration)
    pos = _hermite(ts, np.array([k.pos.as_list() for k in kfs]), samples)
    look = _hermite(ts, np.array([k.look_at.as_list() for k in kfs]), samples)
    roll = _hermite(ts, np.array([[k.roll_deg] for k in kfs]), samples)[:, 0]
    lens = _hermite(ts, np.array([[k.lens_mm] for k in kfs]), samples)[:, 0]

    found: OrderedDict[tuple, Violation] = OrderedDict()

    def note(key, v: Violation):
        if key not in found:
            found[key] = v

    qs, links, axes, cam_pos, cam_fwd, cam_up = [], [], [], [], [], []
    q = None
    min_gap = np.inf
    last_reseed = -10
    for i, t in enumerate(samples):
        target_r = arm.look_rotation(pos[i], look[i], roll[i])
        if q is None:
            q, pe, ae = arm.solve_ik_global(pos[i], target_r, rig)
        else:
            q, pe, ae = arm.solve_ik(pos[i], target_r, rig, q, iters=30)
            # lost track: re-seed globally, but at most every 10 samples (a fully unreachable stretch would
            # otherwise run the 36-seed search on every sample)
            if (pe > POS_TOL or ae > ANG_TOL) and i - last_reseed >= 10:
                last_reseed = i
                q2, pe2, ae2 = arm.solve_ik_global(pos[i], target_r, rig)
                if pe2 + ae2 < pe + ae:
                    q, pe, ae = q2, pe2, ae2
        if qs:  # IK may return an equivalent angle a full turn away; take the turn nearest the previous sample
            prev = qs[-1]
            q[1:] = prev[1:] + (q[1:] - prev[1:] + np.pi) % (2 * np.pi) - np.pi
        if pe > POS_TOL or ae > ANG_TOL:
            rail_pt = np.array([rig.rail_x_start + np.clip(pos[i][0] - rig.rail_x_start, 0, arm.RAIL_LENGTH), rig.rail_y, arm.PEDESTAL_HEIGHT])
            dist = float(np.linalg.norm(pos[i] - rail_pt))
            if pos[i][0] < rig.rail_x_start or pos[i][0] > rig.rail_x_start + arm.RAIL_LENGTH:
                hint = f"The rail covers x={rig.rail_x_start:.2f}..{rig.rail_x_start + arm.RAIL_LENGTH:.2f}; shift rig.rail_x_start so it spans x={pos[i][0]:.2f}, or move the keyframe."
            elif dist > arm.REACH:
                hint = f"That point is {dist:.2f} m from the pedestal (reach about {arm.REACH} m). Move rig.rail_y toward y={pos[i][1] - 0.9:.2f} or bring the camera closer to the rail."
            else:
                hint = "The wrist cannot hold that aim from here (too steep or too close to the pedestal). Aim less steeply, raise or lower the camera 20-40 cm, or move it further along the rail."
            note(("unreachable",), Violation(shot_id=shot.id, kind="unreachable", t=round(float(t), 2),
                 detail=f"Camera pose at {_fmt(pos[i])} aimed at {_fmt(look[i])} is off by {pe * 100:.0f} cm / {np.degrees(ae):.0f} deg.", suggestion=hint,
                 link="camera"))
        frames = arm.forward(q, rig)
        if i % 2 == 0 or i == len(samples) - 1:
            scene_t = shot.start_t + float(t)
            actors = {c.name: c for c in actors_at(scene, scene_t)}
            min_gap = min(min_gap, actor_gap(frames, list(actors.values())))
            for kind, obstacle, link, d, need in check_pose(frames, list(actors.values()), boxes):
                if kind == "actor":
                    ax, ay = actors[obstacle].a[:2]
                    sugg = (f"At scene time {scene_t:.1f}s {obstacle} is at ({ax:.1f}, {ay:.1f}). Keep the whole arm at least {need + 0.1:.1f} m "
                            f"from them (mind their walking path), or approach {obstacle} from the other side so the arm does not cross them.")
                elif kind == "prop":
                    sugg = f"Route the camera around '{obstacle}' (pass above it or stay on the near side) or move the rail so the arm does not reach across it."
                else:
                    sugg = "Raise the camera; the arm cannot go below the floor. Keep camera height above 0.3 m."
                note((kind, obstacle), Violation(shot_id=shot.id, kind=kind, t=round(float(t), 2),
                     detail=f"The {link} comes within {d * 100:.1f} cm of {obstacle} (needs {need * 100:.1f} cm).", suggestion=sugg,
                     link=link, obstacle=obstacle))
        qs.append(q.copy())
        links.append([[round(float(v), 3) for v in [rig.rail_x_start + q[0], rig.rail_y, 0.0]]] + [[round(float(v), 3) for v in f[:3, 3]] for f in frames])
        axes.append([[round(float(v), 3) for v in f[:3, 2]] for f in frames[:6]])
        cam = frames[-1]
        cam_pos.append([round(float(v), 3) for v in cam[:3, 3]])
        cam_fwd.append([round(float(v), 4) for v in cam[:3, 2]])
        cam_up.append([round(float(v), 4) for v in -cam[:3, 1]])

    qs = np.array(qs)
    peak_pct = 0.0
    if len(qs) > 2:
        vel = np.diff(qs, axis=0) / DT
        acc = np.diff(vel, axis=0) / DT
        if len(acc) > 2:  # 3-tap moving average: finite differences of IK samples are noisy
            acc = (acc[:-2] + acc[1:-1] + acc[2:]) / 3
        vel_ratio = np.abs(vel) / arm.VEL_LIMIT
        peak_pct = float(vel_ratio.max() * 100)
        if vel_ratio.max() > 1.0:
            k, j = np.unravel_index(np.argmax(vel_ratio), vel_ratio.shape)
            unit = "m/s" if j == 0 else "deg/s"
            scale = 1.0 if j == 0 else 180 / np.pi
            t0 = ts[max(np.searchsorted(ts, samples[k], side="right") - 1, 0)]
            t1 = ts[min(np.searchsorted(ts, samples[k], side="right"), len(ts) - 1)]
            note(("velocity",), Violation(shot_id=shot.id, kind="velocity", t=round(float(samples[k]), 2),
                 detail=f"The {arm.JOINT_NAMES[j]} axis needs {abs(vel[k, j]) * scale:.0f} {unit} (limit {arm.VEL_LIMIT[j] * scale:.0f}).",
                 suggestion=f"Give the move between t={t0:.1f}s and t={t1:.1f}s more time (about {vel_ratio.max():.1f}x as long), or sweep a smaller angle.",
                 link=arm.JOINT_NAMES[j]))
        if len(acc):
            acc_ratio = np.abs(acc) / arm.ACC_LIMIT
            if acc_ratio.max() > 1.0:
                k, j = np.unravel_index(np.argmax(acc_ratio), acc_ratio.shape)
                ratio = float(acc_ratio.max())
                t_hit = float(samples[min(k + 2, len(samples) - 1)])
                seg = min(max(np.searchsorted(ts, t_hit, side="right") - 1, 0), len(ts) - 2)
                t0, t1 = float(ts[seg]), float(ts[seg + 1])
                # acceleration scales with 1/time^2, so the segment must last sqrt(ratio) times longer (+15% margin)
                need = (t1 - t0) * np.sqrt(ratio) * 1.15
                note(("acceleration",), Violation(shot_id=shot.id, kind="acceleration", t=round(t_hit, 2),
                     detail=f"The {arm.JOINT_NAMES[j]} axis is asked to accelerate {ratio:.1f}x past its limit during the move from t={t0:.2f}s to t={t1:.2f}s.",
                     suggestion=f"Make that move take at least {need:.2f}s instead of {t1 - t0:.2f}s (shift every later keyframe by {need - (t1 - t0):.2f}s), "
                                f"or cut the angle it sweeps by about {100 * (1 - 1 / ratio):.0f}%. Extra keyframes in between do not help; time does.",
                     link=arm.JOINT_NAMES[j]))

    violations += list(found.values())
    motion = ShotMotion(
        shot_id=shot.id, dt=DT, duration=duration, start_t=shot.start_t, links=links, axes=axes, cam_pos=cam_pos, cam_fwd=cam_fwd, cam_up=cam_up,
        lens_mm=[round(float(v), 1) for v in lens], joints=[[round(float(v), 4) for v in q] for q in qs],
        peak_joint_speed_pct=round(peak_pct, 1), min_actor_gap_m=round(min_gap, 3) if np.isfinite(min_gap) else None, ok=not violations,
    )
    return motion, violations


def plan_motion(scene: Scene, shot_list: ShotList) -> RobotPlan:
    rig = arm.Rig(shot_list.rig.rail_x_start, shot_list.rig.rail_y)
    boxes = set_boxes(scene)
    shots, violations = [], []
    for shot in shot_list.shots:
        motion, v = plan_shot(shot, rig, scene, boxes)
        if motion:
            shots.append(motion)
        violations += v
    return RobotPlan(rig=RigPlacement(rail_x_start=rig.rail_x_start, rail_y=rig.rail_y), shots=shots, violations=violations)
