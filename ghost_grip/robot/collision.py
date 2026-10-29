"""Capsule / box distance checks between the arm and the set (actors, props, walls, floor)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LINK_RADIUS = 0.075
CAMERA_RADIUS = 0.12
ACTOR_MARGIN = 0.25  # safety buffer around performers on top of their body radius
PROP_MARGIN = 0.03
LINK_NAMES = ["base column", "upper arm", "forearm", "wrist", "wrist", "flange", "camera mount"]


@dataclass(frozen=True)
class Capsule:
    name: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass(frozen=True)
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray


def segment_distance(p1, q1, p2, q2) -> float:
    """Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def segment_box_distance(a, b, box: Box, samples: int = 9) -> float:
    best = np.inf
    for u in np.linspace(0, 1, samples):
        p = a + (b - a) * u
        d = np.maximum(np.maximum(box.lo - p, 0), p - box.hi)
        best = min(best, float(np.linalg.norm(d)))
    return best


def arm_segments(frames: list[np.ndarray]) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    pts = [f[:3, 3] for f in frames]
    segs = []
    for i in range(len(pts) - 1):
        if np.linalg.norm(pts[i + 1] - pts[i]) > 1e-4:
            segs.append((LINK_NAMES[i], pts[i], pts[i + 1], LINK_RADIUS))
    cam = frames[-1]
    segs.append(("camera", cam[:3, 3] - cam[:3, 2] * 0.15, cam[:3, 3], CAMERA_RADIUS))
    return segs


def check_pose(frames, actors: list[Capsule], boxes: list[Box]):
    """Yield (kind, obstacle, link, distance, required) for every clearance breach in this pose."""
    for link, a, b, r in arm_segments(frames):
        if min(a[2], b[2]) < r:
            yield ("floor", "floor", link, float(min(a[2], b[2])), r)
        for actor in actors:
            need = r + actor.radius + ACTOR_MARGIN
            d = segment_distance(a, b, actor.a, actor.b)
            if d < need:
                yield ("actor", actor.name, link, d, need)
        for box in boxes:
            d = segment_box_distance(a, b, box)
            if d < r + PROP_MARGIN:
                yield ("prop", box.name, link, d, r + PROP_MARGIN)


def actor_gap(frames, actors: list[Capsule]) -> float:
    """Smallest surface-to-surface distance (m) between any arm part and any actor; inf with no actors."""
    best = np.inf
    for _, a, b, r in arm_segments(frames):
        for actor in actors:
            best = min(best, segment_distance(a, b, actor.a, actor.b) - r - actor.radius)
    return float(best)
