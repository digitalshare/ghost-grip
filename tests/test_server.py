from fastapi.testclient import TestClient

from ghost_grip import config, server
from ghost_grip.crew.flow import GhostGripState
from ghost_grip.robot.planner import plan_motion
from tests.test_robot import kf, make_scene, shot_list
from ghost_grip.schemas import Actor


def saved_run(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    scene = make_scene(actors=[Actor(name="Mara", x=2.0, y=3.0, height=1.7)])
    bad = shot_list(kf(0, (2.0, 0.3, 1.5), (0.0, 3, 1.5)), kf(0.3, (2.0, 0.3, 1.5), (4.0, 3, 1.5)))
    good = shot_list(kf(0, (2.0, 0.2, 1.5), (2.0, 3.0, 1.55)), kf(4, (2.0, 0.6, 1.5), (2.0, 3.0, 1.55)))
    plan = plan_motion(scene, good)
    st = GhostGripState(id="0123abcd-0000-0000-0000-000000000000", scene_text="x", scene=scene, shot_list=good, plan=plan, revision=1,
                        rounds=[{"round": 0, "violations": [v.model_dump() for v in plan_motion(scene, bad).violations], "shot_list": bad.model_dump()},
                                {"round": 1, "violations": [], "shot_list": good.model_dump()}])
    (tmp_path / f"{st.id}.json").write_text(st.model_dump_json())
    return st.id


def test_round_replay_reproduces_rejection(tmp_path, monkeypatch):
    rid = saved_run(tmp_path, monkeypatch)
    c = TestClient(server.app)
    r0 = c.get(f"/api/runs/{rid}/rounds/0").json()
    assert not r0["feasible"] and r0["violations"][0]["link"]
    assert c.get(f"/api/runs/{rid}/rounds/1").json()["feasible"]
    assert c.get(f"/api/runs/{rid}/rounds/7").status_code == 404


def test_program_csv_has_one_row_per_sample(tmp_path, monkeypatch):
    rid = saved_run(tmp_path, monkeypatch)
    text = TestClient(server.app).get(f"/api/runs/{rid}/program.csv").text.strip().splitlines()
    assert text[0].startswith("shot,t_s,scene_t_s,rail_m,j1_deg")
    assert len(text) == 1 + 101 and all(len(row.split(",")) == 14 for row in text)
