"""Prove each partner key works before a demo:  uv run python -m ghost_grip.smoke"""
from __future__ import annotations

import shutil
import subprocess

from pydantic import BaseModel

from ghost_grip import config, llm, youcom


class Ping(BaseModel):
    film: str
    year: int


def check(name: str, fn):
    try:
        print(f"✓ {name}: {fn()}")
    except Exception as e:
        print(f"✗ {name}: {type(e).__name__}: {str(e)[:300]}")


check("You.com search", lambda: youcom.search("The Shining hallway Steadicam shot", count=2)[0]["url"])
check("Parasail models", lambda: config.PARASAIL_MODEL in [m.id for m in llm.client().models.list().data] or "model id not in /models list!")
check("Parasail schema JSON", lambda: llm.structured(Ping, "Answer as JSON.", "Which film features the Overlook Hotel?").model_dump())
check("Parasail embeddings", lambda: f"dim={len(llm.embed(['dolly zoom'])[0])}")
check("One secret", lambda: "set" if config.ONE_SECRET else (_ for _ in ()).throw(RuntimeError("ONE_SECRET missing")))
check("Node for @withone/mcp", lambda: subprocess.run([shutil.which("node") or "node", "--version"], capture_output=True, text=True).stdout.strip())
