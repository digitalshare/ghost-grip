import httpx

from ghost_grip import clips
from ghost_grip.crew.flow import GhostGripState
from ghost_grip.schemas import References, ScoutReference
from ghost_grip.llm import _strict_schema

SHINING = ScoutReference(film="The Shining (1980)", moment="the twins at the end of the hallway", technique="Steadicam", url="https://example.com/a", why="")

OEMBED = {
    "CMbI7DmLCNI": {"title": "The Shining (1980) - Come Play With Us Scene (2/7) | Movieclips", "author_name": "Movieclips"},
    "JHuwLvm4XZA": {"title": "Reactors Reaction To Seeing The Grady Twins | The Shining (1980) Reaction", "author_name": "Mixed Reactions"},
    "PRIVATEVIDx": None,  # oEmbed answers 401 for videos that are private or refuse embedding
}


def fake_world(monkeypatch, tmp_path, urls):
    monkeypatch.setattr(clips, "CACHE_PATH", tmp_path / "clips.json")
    monkeypatch.setattr(clips.youcom, "search", lambda q, **kw: [{"url": u, "title": "", "snippet": "", "thumbnail": ""} for u in urls])

    def get(url, params=None, timeout=None):
        vid = params["url"].split("v=")[1]
        meta = OEMBED.get(vid)
        return httpx.Response(200, json=meta) if meta else httpx.Response(401)

    monkeypatch.setattr(clips.httpx, "get", get)


def test_official_clip_beats_reaction_video(monkeypatch, tmp_path):
    fake_world(monkeypatch, tmp_path, ["https://www.youtube.com/watch?v=JHuwLvm4XZA", "https://www.youtube.com/watch?v=CMbI7DmLCNI"])
    clip = clips.find_clip(SHINING)
    assert clip and clip.video_id == "CMbI7DmLCNI" and clip.channel == "Movieclips"


def test_unembeddable_and_non_youtube_results_are_dropped(monkeypatch, tmp_path):
    fake_world(monkeypatch, tmp_path, ["https://screenrant.com/shining-twins", "https://youtu.be/PRIVATEVIDx", "https://www.youtube.com/watch?v=JHuwLvm4XZA"])
    assert clips.find_clip(SHINING) is None  # only a reaction video survives oEmbed, and it scores too low


def test_result_is_cached(monkeypatch, tmp_path):
    fake_world(monkeypatch, tmp_path, ["https://www.youtube.com/watch?v=CMbI7DmLCNI"])
    assert clips.find_clip(SHINING)
    monkeypatch.setattr(clips.youcom, "search", lambda q, **kw: (_ for _ in ()).throw(AssertionError("should hit the cache")))
    assert clips.find_clip(SHINING).video_id == "CMbI7DmLCNI"


def test_video_id_parsing():
    assert clips.video_id("https://www.youtube.com/watch?v=CMbI7DmLCNI") == ("CMbI7DmLCNI", 0)
    assert clips.video_id("https://youtu.be/CMbI7DmLCNI?t=42") == ("CMbI7DmLCNI", 42)
    assert clips.video_id("https://m.youtube.com/watch?v=CMbI7DmLCNI&t=1m05s") == ("CMbI7DmLCNI", 65)
    assert clips.video_id("https://www.youtube.com/@movieclips") is None
    assert clips.video_id("https://example.com/watch?v=CMbI7DmLCNI") is None


def test_llm_never_sees_a_clip_field_and_old_runs_still_load():
    props = _strict_schema(References)["$defs"]["ScoutReference"]["properties"]
    assert "clip" not in props and set(props) == {"film", "moment", "technique", "url", "why"}
    old = {"id": "x", "references": [{"film": "Jaws", "moment": "m", "technique": "t", "url": "https://u", "why": "w"}]}
    assert GhostGripState.model_validate(old).references[0].clip is None
