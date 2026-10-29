.PHONY: install run test smoke
install:
	uv sync
	npx -y @withone/mcp </dev/null >/dev/null 2>&1 || true

run:
	uv run uvicorn ghost_grip.server:app --port 8017

test:
	uv run pytest -q

smoke:
	uv run python -m ghost_grip.smoke
