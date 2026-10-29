"""Find a playable clip of each reference moment: You.com search restricted to YouTube, every candidate
verified (and its channel read) through YouTube's oEmbed endpoint, then scored to prefer official clip channels.

Nothing is downloaded: the UI plays clips through YouTube's own embedded player.
"""
from __future__ import annotations

import json
import re
import threading
from urllib.parse import parse_qs, urlparse

import httpx

from ghost_grip import config, youcom
from ghost_grip.schemas import Clip, ScoutReference

OEMBED = "https://www.youtube.com/oembed"
OFFICIAL = ("movieclips", "warner bros", "universal pictures", "paramount", "sony pictures", "lionsgate", "a24", "blumhouse",
            "focus features", "neon", "rotten tomatoes", "netflix", "screen bites", "shudder", "20th century", "searchlight",
            "mgm", "new line", "hbo", "fandango", "zero media", "the criterion")
NOT_THE_FILM = ("reaction", "reacts", "review", "explained", "ranking", "top 10", "top ten", "parody", "fan made", "fanmade",
                "edit", "tiktok", "#shorts", "breakdown", "theory", "recap", "vs ")
STOPWORDS = {"the", "a", "an", "of", "and", "in", "on"}

CACHE_PATH = config.CACHE_DIR / "clips.json"
_lock = threading.Lock()


def video_id(url: str) -> tuple[str, int] | None:
    """(id, start seconds) for youtube.com/watch?v=… or youtu.be/… URLs; None for anything else."""
    u = urlparse(url)
    host = (u.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    q = parse_qs(u.query)
    if host == "youtube.com" and u.path == "/watch":
        vid = (q.get("v") or [""])[0]
    elif host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
    else:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        return None
    t = (q.get("t") or q.get("start") or ["0"])[0]
    m = re.fullmatch(r"(?:(\d+)m)?(\d+)s?", t)
    start = int(m.group(1) or 0) * 60 + int(m.group(2)) if m else 0
    return vid, start


def oembed(vid: str) -> dict | None:
    """Title and channel if the video exists and allows embedding; None otherwise (oEmbed answers 400/401/404)."""
    try:
        r = httpx.get(OEMBED, params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"}, timeout=5)
    except httpx.HTTPError:
        return None
    return r.json() if r.status_code == 200 else None


def _title_words(film: str) -> set[str]:
    film = re.sub(r"\(\d{4}\)", "", film)
    return {w for w in re.findall(r"[a-z0-9']+", film.lower()) if w not in STOPWORDS}


def moment_overlap(moment: str, text: str) -> int:
    """How many distinctive words of the cited moment (names, places, objects) the clip's title or description shares."""
    words = {w for w in re.findall(r"[a-z]{4,}", moment.lower())} - {"scene", "shot", "camera", "slow", "with", "from", "that", "into", "down", "over", "never", "what", "behind", "follows"}
    words = {w[:-1] if w.endswith("s") else w for w in words}
    t = text.lower()
    return sum(1 for w in words if w in t)


def score(film: str, title: str, channel: str, moment: str = "", context: str = "") -> int:
    t, c = title.lower(), channel.lower()
    s = 2 * min(3, moment_overlap(moment, f"{title} {context}")) if moment else 0  # the cited moment matters most
    if any(o in c for o in OFFICIAL):
        s += 2
    words = _title_words(film)
    if words and all(w in t for w in words):
        s += 2
    if "scene" in t or "clip" in t:
        s += 1
    if any(b in t for b in NOT_THE_FILM):
        s -= 6
    return s


def _cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}


def find_clip(ref: ScoutReference, max_candidates: int = 5) -> Clip | None:
    key = f"{ref.film}|{ref.moment}".lower()
    with _lock:
        cached = _cache()
    if key in cached:
        return Clip.model_validate(cached[key]) if cached[key] else None
    film = re.sub(r"\s*\(\d{4}\)", "", ref.film).strip()
    year = (re.search(r"\((\d{4})\)", ref.film) or [None, ""])[1]
    best: tuple[int, Clip] | None = None
    seen: set[str] = set()
    # moment-specific queries first; the film-only query is a fallback whose clip is labelled "same film, not this moment"
    queries = (f"{film} {year} {ref.moment} scene clip", f"{film} {year} {ref.moment}", f"{film} {year} movie clip scene")
    for qi, query in enumerate(queries):
        if qi == 2 and best and best[0] >= 4:
            break
        hits = youcom.search(" ".join(query.split()), count=8, include_domains=["youtube.com"], highlights=False)
        tried = 0
        for h in hits:
            parsed = video_id(h["url"])
            if not parsed or parsed[0] in seen or tried >= max_candidates:
                continue
            seen.add(parsed[0]); tried += 1
            vid, start = parsed
            meta = oembed(vid)
            if not meta:
                continue
            context = f"{h['title']} {h['snippet']}"
            s = score(ref.film, meta.get("title", ""), meta.get("author_name", ""), ref.moment, context)
            clip = Clip(video_id=vid, title=meta.get("title", ""), channel=meta.get("author_name", ""),
                        url=f"https://www.youtube.com/watch?v={vid}", thumbnail=f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg", start_s=start,
                        matches_moment=moment_overlap(ref.moment, f"{meta.get('title', '')} {context}") > 0)
            if best is None or s > best[0]:
                best = (s, clip)
    found = best[1] if best and best[0] >= 3 else None
    with _lock:
        cached = _cache()
        cached[key] = found.model_dump() if found else None
        config.CACHE_DIR.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cached, indent=1))
    return found
