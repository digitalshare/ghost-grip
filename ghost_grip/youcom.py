"""You.com: grounding. Web Search (with highlights) and Contents, plus CrewAI tool wrappers."""
from __future__ import annotations

import httpx
from crewai.tools import tool

from ghost_grip import config

SEARCH_URL = "https://ydc-index.io/v1/search"
CONTENTS_URL = "https://ydc-index.io/v1/contents"


def _post(url: str, payload: dict) -> dict:
    if not config.YDC_API_KEY:
        raise RuntimeError("YDC_API_KEY is not set; copy .env.example to .env and fill it in.")

    def run() -> dict:
        r = httpx.post(url, json=payload, headers={"X-API-Key": config.YDC_API_KEY}, timeout=45)
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After") or r.headers.get("X-RateLimit-Reset")
            raise config.RetryableError(f"You.com {r.status_code}", float(ra) if ra and ra.replace(".", "").isdigit() else None)
        r.raise_for_status()
        return r.json()
    return config.with_backoff(run)


def search(query: str, count: int = 6, include_domains: list[str] | None = None, highlights: bool = True) -> list[dict]:
    """Return [{title, url, snippet, thumbnail}] (query-relevant highlights by default)."""
    payload: dict = {"query": query, "count": count}
    if highlights:
        payload["extraction"] = {"extraction_mode": "highlights"}
    if include_domains:
        payload["include_domains"] = include_domains
    data = _post(SEARCH_URL, payload)
    results = data.get("results", data)
    web = results.get("web", []) if isinstance(results, dict) else []
    out = []
    for item in web[:count]:
        snippets = (item.get("contents") or {}).get("highlights") or item.get("snippets") or [item.get("description", "")]
        out.append({"title": item.get("title", ""), "url": item.get("url", ""), "snippet": " … ".join(s for s in snippets if s)[:700],
                    "thumbnail": item.get("thumbnail_url", "")})
    return out


def contents(urls: list[str], max_chars: int = 4000) -> list[dict]:
    data = _post(CONTENTS_URL, {"urls": urls, "formats": ["markdown"]})
    items = data if isinstance(data, list) else data.get("results", data.get("output", []))
    return [{"url": it.get("url", ""), "title": it.get("title", ""), "markdown": (it.get("markdown") or "")[:max_chars]} for it in items]


@tool("search_film_references")
def search_film_references(query: str) -> str:
    """Search the live web (You.com) for real films, scenes and camera techniques. Returns titles, URLs and the passages that answer the query. Use specific queries like 'The Shining hallway Steadicam tracking shot'."""
    hits = search(query)
    if not hits:
        return "No results."
    return "\n\n".join(f"[{i + 1}] {h['title']}\nURL: {h['url']}\n{h['snippet']}" for i, h in enumerate(hits))


@tool("read_page")
def read_page(url: str) -> str:
    """Fetch the main text of one web page (You.com Contents) to confirm details of a shot before citing it."""
    pages = contents([url], max_chars=3500)
    return pages[0]["markdown"] if pages and pages[0]["markdown"] else "Could not read that page."
