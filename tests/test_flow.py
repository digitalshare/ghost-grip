from ghost_grip import config
from ghost_grip.crew.flow import GhostGripFlow
from ghost_grip.llm import _strict_schema
from ghost_grip.schemas import RigPlacement, RobotPlan, ShotList, Violation

V = Violation(shot_id="S1", kind="velocity", t=1.0, detail="too fast", suggestion="slow down")


def flow_with(violations, revision=0) -> GhostGripFlow:
    f = GhostGripFlow()
    f.state.plan = RobotPlan(rig=RigPlacement(rail_x_start=0, rail_y=0), shots=[], violations=violations)
    f.state.revision = revision
    return f


def test_gate_revises_on_violations():
    f = flow_with([V])
    assert f.gate() == "revise"
    assert f.state.revision == 1


def test_gate_approves_feasible_plan():
    f = flow_with([])
    assert f.gate() == "approved" and f.state.approved


def test_gate_stops_after_max_revisions():
    f = flow_with([V], revision=config.MAX_REVISIONS)
    assert f.gate() == "approved" and not f.state.approved


def test_strict_schema_requires_every_field():
    schema = _strict_schema(ShotList)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"director_notes", "rig", "shots"}
    for d in schema["$defs"].values():
        assert d["additionalProperties"] is False and set(d["required"]) == set(d["properties"])


def test_strict_schema_keeps_fields_named_title():
    from ghost_grip.schemas import Scene
    schema = _strict_schema(Scene)
    assert "title" in schema["properties"] and "title" in schema["required"]


def test_tidy_strips_stray_separators():
    from ghost_grip.llm import _tidy
    assert _tidy({"name": ",Maya", "props": [{"name": " ,Wardrobe "}], "n": 2}) == {"name": "Maya", "props": [{"name": "Wardrobe"}], "n": 2}
