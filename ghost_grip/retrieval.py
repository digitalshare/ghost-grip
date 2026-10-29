"""Technique retrieval: Parasail GritLM embeddings over the camera-move library, cosine top-k."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ghost_grip import config, llm

LIBRARY_PATH = Path(__file__).parent / "library" / "techniques.json"


def load_library() -> list[dict]:
    return json.loads(LIBRARY_PATH.read_text())


def _doc(t: dict) -> str:
    return f"{t['name']}. Effect: {t['effect']}. Move: {t['move']} Examples: {t['examples']}."


def _library_matrix(library: list[dict]) -> np.ndarray:
    docs = [_doc(t) for t in library]
    key = hashlib.sha256((config.PARASAIL_EMBED_MODEL + "\n".join(docs)).encode()).hexdigest()[:16]
    path = config.CACHE_DIR / f"techniques-{key}.npy"
    if path.exists():
        return np.load(path)
    matrix = np.array(llm.embed(docs), dtype=np.float32)
    config.CACHE_DIR.mkdir(exist_ok=True)
    np.save(path, matrix)
    return matrix


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)


def top_techniques(queries: list[str], k: int = 6) -> list[dict]:
    """Best techniques for a set of beat descriptions; each result carries its similarity and the beat it matched."""
    library = load_library()
    lib = _normalise(_library_matrix(library))
    q = _normalise(np.array(llm.embed(queries), dtype=np.float32))
    sims = q @ lib.T  # [beats, techniques]
    best = sims.max(axis=0)
    order = np.argsort(-best)[:k]
    return [{**library[i], "score": round(float(best[i]), 3), "matched_beat": queries[int(sims[:, i].argmax())]} for i in order]
