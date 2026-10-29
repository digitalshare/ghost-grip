"""Shared data contracts. The LLM-facing models are deliberately flat so schema-locked decoding stays reliable.

Set coordinates (metres, Z-up): x runs left→right across the set as seen from the camera side,
y runs from the open fourth wall (y=0) toward the back wall (y=depth), z is height above the floor.
The camera rail lies along x at y = rig.rail_y, usually just outside the fourth wall (y < 0.5).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Vec3(BaseModel):
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


class SetDims(BaseModel):
    width: float = Field(description="Set width along x in metres (3-10)")
    depth: float = Field(description="Set depth along y in metres (3-8)")
    height: float = Field(description="Ceiling height in metres (2.4-4)")


class Mark(BaseModel):
    t: float = Field(description="Scene time in seconds when the actor reaches this mark")
    x: float
    y: float


class Actor(BaseModel):
    name: str
    kind: Literal["human", "entity"] = Field(default="human", description="'entity' for ghosts, apparitions and other supernatural figures")
    x: float = Field(description="Starting mark x (scene time 0)")
    y: float = Field(description="Starting mark y (scene time 0)")
    height: float = Field(description="Standing height in metres; ~1.0 if crouching or sitting")
    path: list[Mark] = Field(default_factory=list, description="Later marks in time order if the actor moves; empty if they stay put")

    def position(self, t: float) -> tuple[float, float]:
        """Where the actor stands at scene time t (linear between marks, holding at the ends)."""
        marks = [(0.0, self.x, self.y)] + sorted(((m.t, m.x, m.y) for m in self.path if m.t > 0), key=lambda m: m[0])
        if t <= marks[0][0] or len(marks) == 1:
            return marks[0][1], marks[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(marks, marks[1:]):
            if t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-6)
                return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        return marks[-1][1], marks[-1][2]


class Prop(BaseModel):
    name: str
    x: float = Field(description="Centre x")
    y: float = Field(description="Centre y")
    width: float
    depth: float
    height: float


class Beat(BaseModel):
    t: float = Field(description="Seconds from the start of the scene")
    description: str
    focus: str = Field(description="Name of the actor or prop the audience should look at")


class Scene(BaseModel):
    title: str
    logline: str
    mood: str = Field(description="Short phrase, e.g. 'creeping dread then jolt'")
    set: SetDims
    actors: list[Actor]
    props: list[Prop]
    beats: list[Beat]
    duration_s: float


class Keyframe(BaseModel):
    t: float = Field(description="Seconds from the start of this shot; first keyframe is 0")
    pos: Vec3 = Field(description="Camera position in set coordinates")
    look_at: Vec3 = Field(description="Point the camera aims at")
    lens_mm: float = Field(description="Focal length, 14-135")
    roll_deg: float = Field(description="Dutch angle in degrees, 0 for level")


class Shot(BaseModel):
    id: str = Field(description="Short id like S1")
    name: str
    technique: str = Field(description="Technique name from the library")
    intent: str = Field(description="What the move does to the audience")
    reference_title: str = Field(description="Film and moment this shot cites")
    reference_url: str = Field(description="Source URL found by the reference scout")
    start_t: float = Field(default=0.0, description="Scene time in seconds at which this shot begins")
    keyframes: list[Keyframe]


class RigPlacement(BaseModel):
    rail_x_start: float = Field(description="x where the 4 m rail begins")
    rail_y: float = Field(description="y of the rail line; arm reach is ~1.3 m from it")


class ShotList(BaseModel):
    director_notes: str
    rig: RigPlacement
    shots: list[Shot]


class Clip(BaseModel):
    """A playable clip of the reference moment, verified to exist and be embeddable via YouTube oEmbed."""
    video_id: str
    title: str
    channel: str
    url: str
    thumbnail: str = ""
    start_s: int = 0
    matches_moment: bool = False  # False: a clip from the same film, not necessarily the cited moment


class ScoutReference(BaseModel):
    """What the Reference Scout reports. Deliberately has no clip field: clips are found and verified in code, never by the LLM."""
    film: str
    moment: str
    technique: str
    url: str
    why: str


class Reference(ScoutReference):
    clip: Clip | None = None


class References(BaseModel):
    references: list[ScoutReference]


class Violation(BaseModel):
    shot_id: str
    kind: Literal["unreachable", "joint_limit", "velocity", "acceleration", "actor", "prop", "floor", "bad_input"]
    t: float
    detail: str
    suggestion: str
    link: str = ""  # arm part (or axis) at fault, for highlighting
    obstacle: str = ""  # actor / prop involved, if any


class ShotMotion(BaseModel):
    shot_id: str
    dt: float
    duration: float
    start_t: float = 0.0
    links: list[list[list[float]]]  # [sample][point][xyz] arm points for rendering
    axes: list[list[list[float]]] = []  # [sample][joint 1..6][xyz] joint rotation axes for rendering housings
    cam_pos: list[list[float]]
    cam_fwd: list[list[float]]
    cam_up: list[list[float]]
    lens_mm: list[float]
    joints: list[list[float]]
    peak_joint_speed_pct: float
    min_actor_gap_m: float | None = None
    ok: bool


class RobotPlan(BaseModel):
    rig: RigPlacement
    shots: list[ShotMotion]
    violations: list[Violation]

    @property
    def feasible(self) -> bool:
        return not self.violations
