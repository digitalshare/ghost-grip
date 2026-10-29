"""A 6-axis camera arm (UR10-class DH) on a pedestal that rides a linear rail.

World frame is Z-up, metres. The rail runs along +X starting at ``rail_x_start``
at ``y = rail_y``. Configuration vector ``q = [s, th1..th6]`` where ``s`` is the
carriage position along the rail (0..RAIL_LENGTH).

The camera's optical axis is the flange +Z; its "right" is flange +X and "down"
is flange +Y (OpenCV convention), so roll is controlled by joint 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# UR10 standard DH parameters.
DH_D = np.array([0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922])
DH_A = np.array([0.0, -0.612, -0.5723, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

PEDESTAL_HEIGHT = 0.6
CAMERA_OFFSET = 0.08  # optical centre ahead of the flange
RAIL_LENGTH = 4.0
REACH = 1.3

DEG = np.pi / 180.0
JOINT_LOWER = np.array([0.0, -2 * np.pi, -2 * np.pi, -160 * DEG, -2 * np.pi, -2 * np.pi, -2 * np.pi])
JOINT_UPPER = np.array([RAIL_LENGTH, 2 * np.pi, 2 * np.pi, 160 * DEG, 2 * np.pi, 2 * np.pi, 2 * np.pi])
VEL_LIMIT = np.array([1.5, 120 * DEG, 120 * DEG, 150 * DEG, 200 * DEG, 200 * DEG, 200 * DEG])
ACC_LIMIT = np.array([2.5, 300 * DEG, 300 * DEG, 360 * DEG, 500 * DEG, 500 * DEG, 500 * DEG])
JOINT_NAMES = ["rail", "base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]

HOME = np.array([RAIL_LENGTH / 2, 0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


@dataclass(frozen=True)
class Rig:
    rail_x_start: float
    rail_y: float


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def forward(q: np.ndarray, rig: Rig) -> list[np.ndarray]:
    """Return 8 world transforms: arm base, joints 1..6 frames, camera."""
    base = np.eye(4)
    base[:3, 3] = [rig.rail_x_start + q[0], rig.rail_y, PEDESTAL_HEIGHT]
    frames = [base]
    t = base
    for i in range(6):
        t = t @ _dh(q[i + 1], DH_D[i], DH_A[i], DH_ALPHA[i])
        frames.append(t)
    cam = t.copy()
    cam[:3, 3] = t[:3, 3] + t[:3, 2] * CAMERA_OFFSET
    frames.append(cam)
    return frames


def camera_pose(q: np.ndarray, rig: Rig) -> np.ndarray:
    return forward(q, rig)[-1]


def look_rotation(pos: np.ndarray, look_at: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """Camera rotation (columns right, down, forward) for a world-up camera, with optional Dutch roll."""
    fwd = np.asarray(look_at, float) - np.asarray(pos, float)
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up/down
        right = np.array([0.0, -1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    if roll_deg:
        r = roll_deg * DEG
        right, down = np.cos(r) * right + np.sin(r) * down, -np.sin(r) * right + np.cos(r) * down
    return np.column_stack([right, down, fwd])


def rotvec(r: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    cos = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-7:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = np.sin(angle)
    if s < 1e-6:  # angle ~ pi
        axis = np.sqrt(np.clip((np.diag(r) + 1) / 2, 0, None))
        return axis * angle
    return v / (2 * s) * angle


def jacobian(frames: list[np.ndarray]) -> np.ndarray:
    p = frames[-1][:3, 3]
    j = np.zeros((6, 7))
    j[:3, 0] = [1.0, 0.0, 0.0]  # prismatic rail along world X
    for i in range(6):
        z, o = frames[i][:3, 2], frames[i][:3, 3]
        j[:3, i + 1] = np.cross(z, p - o)
        j[3:, i + 1] = z
    return j


ORIENT_WEIGHT = 0.4  # metres of position error that equal one radian of orientation error


def pose_error(q: np.ndarray, rig: Rig, target_p: np.ndarray, target_r: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    frames = forward(q, rig)
    cam = frames[-1]
    e = np.concatenate([target_p - cam[:3, 3], rotvec(target_r @ cam[:3, :3].T)])
    return e, frames


def solve_ik(target_p, target_r, rig: Rig, seed: np.ndarray, iters: int = 60, damping: float = 0.04):
    """Damped least squares over rail + 6 joints. Returns (q, pos_err_m, ang_err_rad)."""
    q = np.clip(np.array(seed, float), JOINT_LOWER, JOINT_UPPER)
    target_p = np.asarray(target_p, float)
    w = np.array([1, 1, 1, ORIENT_WEIGHT, ORIENT_WEIGHT, ORIENT_WEIGHT])
    for _ in range(iters):
        e, frames = pose_error(q, rig, target_p, target_r)
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        j = jacobian(frames) * w[:, None]
        ew = e * w
        dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(6), ew)
        step = np.linalg.norm(dq)
        if step > 0.4:
            dq *= 0.4 / step
        q = np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)
    e, _ = pose_error(q, rig, target_p, target_r)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def initial_seeds(target_p: np.ndarray, rig: Rig) -> list[np.ndarray]:
    s0 = float(np.clip(target_p[0] - rig.rail_x_start, 0.0, RAIL_LENGTH))
    yaw = float(np.arctan2(target_p[1] - rig.rail_y, target_p[0] - (rig.rail_x_start + s0)))
    seeds = []
    for s in (s0, np.clip(s0 - 0.6, 0, RAIL_LENGTH), np.clip(s0 + 0.6, 0, RAIL_LENGTH)):
        for base in (yaw + np.pi, yaw, yaw + np.pi / 2, yaw - np.pi / 2):
            for shoulder, elbow in ((-np.pi / 2, np.pi / 2), (-np.pi / 3, np.pi / 3), (-2 * np.pi / 3, 2 * np.pi / 3)):
                seeds.append(np.array([s, base, shoulder, elbow, -np.pi / 2, -np.pi / 2, 0.0]))
    return seeds


def solve_ik_global(target_p, target_r, rig: Rig):
    """Multi-seed IK for the first pose of a move; prefers converged, low-effort solutions."""
    best = None
    for seed in initial_seeds(np.asarray(target_p, float), rig):
        q, pe, ae = solve_ik(target_p, target_r, rig, seed, iters=80)
        score = pe + 0.3 * ae + (0.0 if pe < 0.01 and ae < 0.03 else 10.0) + 0.01 * np.linalg.norm(q[1:] - HOME[1:])
        if best is None or score < best[0]:
            best = (score, q, pe, ae)
    return best[1], best[2], best[3]
