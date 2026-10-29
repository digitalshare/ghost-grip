import numpy as np

from ghost_grip.robot import arm
from ghost_grip.robot.planner import plan_motion
from ghost_grip.schemas import Actor, Keyframe, Prop, RigPlacement, Scene, SetDims, Shot, ShotList, Vec3

RIG = arm.Rig(rail_x_start=0.0, rail_y=-0.4)


def make_scene(actors=None, props=None) -> Scene:
    return Scene(
        title="t", logline="l", mood="dread", set=SetDims(width=5, depth=5, height=3),
        actors=actors or [], props=props or [], beats=[], duration_s=10,
    )


def kf(t, pos, look, lens=35.0, roll=0.0) -> Keyframe:
    return Keyframe(t=t, pos=Vec3(x=pos[0], y=pos[1], z=pos[2]), look_at=Vec3(x=look[0], y=look[1], z=look[2]), lens_mm=lens, roll_deg=roll)


def shot_list(*keyframes, rig=RIG) -> ShotList:
    shot = Shot(id="S1", name="test", technique="push-in", intent="", reference_title="", reference_url="", keyframes=list(keyframes))
    return ShotList(director_notes="", rig=RigPlacement(rail_x_start=rig.rail_x_start, rail_y=rig.rail_y), shots=[shot])


def test_fk_ik_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = arm.HOME + rng.normal(0, 0.3, 7)
        q[0] = rng.uniform(0.5, 3.5)
        target = arm.camera_pose(q, RIG)
        q2, pe, ae = arm.solve_ik_global(target[:3, 3], target[:3, :3], RIG)
        assert pe < 1e-3 and ae < 1e-2
        np.testing.assert_allclose(arm.camera_pose(q2, RIG)[:3, 3], target[:3, 3], atol=1e-3)


def test_look_rotation_is_level():
    r = arm.look_rotation(np.zeros(3), np.array([0.0, 1.0, 0.0]))
    assert abs(r[2, 0]) < 1e-9  # camera "right" is horizontal
    np.testing.assert_allclose(r[:, 2], [0, 1, 0])


def test_unreachable_pose_is_flagged():
    plan = plan_motion(make_scene(), shot_list(kf(0, (2, 3.5, 1.5), (2, 5, 1.5)), kf(2, (2.2, 3.5, 1.5), (2, 5, 1.5))))
    assert any(v.kind == "unreachable" for v in plan.violations)


def test_gentle_push_in_is_feasible():
    plan = plan_motion(make_scene(actors=[Actor(name="Mara", x=2.0, y=3.0, height=1.7)]),
                       shot_list(kf(0, (2.0, 0.2, 1.5), (2.0, 3.0, 1.55)), kf(4, (2.0, 0.6, 1.5), (2.0, 3.0, 1.55))))
    assert plan.violations == [], plan.violations
    assert plan.shots[0].ok and len(plan.shots[0].links) == 101


def test_collision_with_actor_is_flagged():
    plan = plan_motion(make_scene(actors=[Actor(name="Mara", x=2.0, y=0.8, height=1.7)]),
                       shot_list(kf(0, (2.0, 0.3, 1.5), (2.0, 3, 1.5)), kf(3, (2.0, 0.6, 1.5), (2.0, 3, 1.5))))
    assert any(v.kind == "actor" and "Mara" in v.detail for v in plan.violations)


def test_collision_with_prop_is_flagged():
    plan = plan_motion(make_scene(props=[Prop(name="closet", x=2.0, y=0.55, width=1.0, depth=0.4, height=2.2)]),
                       shot_list(kf(0, (2.0, 0.2, 1.5), (2.0, 3, 1.5)), kf(3, (2.0, 0.5, 1.5), (2.0, 3, 1.5))))
    assert any(v.kind == "prop" for v in plan.violations)


def test_whip_pan_too_fast_is_flagged():
    plan = plan_motion(make_scene(), shot_list(kf(0, (2.0, 0.3, 1.5), (0.0, 3, 1.5)), kf(0.3, (2.0, 0.3, 1.5), (4.0, 3, 1.5))))
    assert any(v.kind in ("velocity", "acceleration") for v in plan.violations)


def test_full_turn_ik_flip_is_not_a_speed_violation():
    # a slow 90 degree pan: any 2*pi jump between IK samples must be unwrapped, not reported as thousands of deg/s
    plan = plan_motion(make_scene(), shot_list(kf(0, (2.0, 0.3, 1.5), (0.5, 2.5, 1.5)), kf(4, (2.0, 0.3, 1.5), (3.5, 2.5, 1.5))))
    assert plan.shots[0].peak_joint_speed_pct < 100, plan.violations
    joints = np.array(plan.shots[0].joints)
    assert np.abs(np.diff(joints[:, 1:], axis=0)).max() < 0.5


def test_actor_walking_into_the_move_is_flagged():
    from ghost_grip.schemas import Mark
    # Mara starts 2 m away and walks onto the camera path; a static check at her start mark would miss it
    walker = Actor(name="Mara", x=4.5, y=0.7, height=1.7, path=[Mark(t=2.0, x=2.0, y=0.7)])
    sl = shot_list(kf(0, (2.0, 0.2, 1.5), (2.0, 3, 1.5)), kf(3, (2.0, 0.3, 1.5), (2.0, 3, 1.5)))
    plan = plan_motion(make_scene(actors=[walker]), sl)
    hits = [v for v in plan.violations if v.kind == "actor"]
    assert hits and hits[0].obstacle == "Mara" and hits[0].t >= 1.0 and hits[0].link


def test_actor_position_interpolates_marks():
    from ghost_grip.schemas import Mark
    a = Actor(name="Leo", x=0, y=1, height=1.4, path=[Mark(t=4, x=4, y=1)])
    assert a.position(-1) == (0, 1) and a.position(2) == (2.0, 1.0) and a.position(9) == (4, 1)
