"""One: the Producer agent delivers the approved plan through the crew's real apps (Gmail, Google Calendar, Discord).

The MCP wiring (`one_mcp_params`, `dropping_nulls`, `harden_execute`, RULES) follows the event's
one-hackathon skill: https://hackathon.withone.ai/sf-halloween-2026/skill.md ("CrewAI with One").
Every exact payload (Gmail raw, calendar body, call sheet text) is built by plain Python from the plan.
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Callable

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from ghost_grip import config, llm
from ghost_grip.crew.flow import AGENTS, GhostGripState

RULES = (
    "Before executing any One action, find it with find_one_actions and read the documentation it returns. "
    "Omit optional tool parameters entirely; never pass null. "
    "Never call execute_one_action with an empty dataJson when the action needs a body. "
    "If a call fails twice with the same error, change the request based on the error instead of repeating it."
)


def one_mcp_params() -> StdioServerParameters:
    env = {**os.environ}
    if config.ONE_NODE_DIR:  # One refuses Node < 22.23.4; put a patched Node first without touching the user's shell
        env["PATH"] = config.ONE_NODE_DIR + os.pathsep + env.get("PATH", "")
    return StdioServerParameters(command="npx", args=["-y", "@withone/mcp"], env=env)


def strip_nulls(value):
    # CrewAI fills every optional field with None, nested ones too (find_one_actions' `load` items)
    if isinstance(value, dict):
        return {k: strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [strip_nulls(v) for v in value]
    if hasattr(value, "model_dump"):
        return strip_nulls(value.model_dump())
    return value


def dropping_nulls(tools):
    for t in tools:
        inner = t._run
        object.__setattr__(t, "_run", (lambda f: lambda **kw: f(**strip_nulls(kw)))(inner))
    return tools


def harden_execute(tools):
    mcp_execute = next(t for t in tools if t.name == "execute_one_action")

    @tool("execute_one_action")
    def execute_one_action(platform: str, actionId: str, connectionKey: str, dataJson: str = "", pathVariablesJson: str = "",
                           queryParamsJson: str = "", headersJson: str = "", isFormData: bool = False, isFormUrlEncoded: bool = False) -> str:
        """Execute an API action on a connected platform via One. Pass request parts as JSON-encoded STRINGS. Omit or pass "" for parts the action does not need."""
        kwargs = {"platform": platform, "actionId": actionId, "connectionKey": connectionKey}
        for key, raw in (("data", dataJson), ("pathVariables", pathVariablesJson), ("queryParams", queryParamsJson), ("headers", headersJson)):
            if raw and raw.strip() and raw.strip() != "{}":
                kwargs[key] = json.loads(raw)
        if "headers" in kwargs:  # never let the model override One's own headers (connection key, action id, secret)
            kwargs["headers"] = {k: v for k, v in kwargs["headers"].items() if not k.lower().startswith("x-one-")}
        if isFormData:
            kwargs["isFormData"] = True
        if isFormUrlEncoded:
            kwargs["isFormUrlEncoded"] = True
        return mcp_execute._run(**kwargs)

    return [t for t in tools if t.name != "execute_one_action"] + [execute_one_action]


def call_sheet(state: GhostGripState, shoot_start: datetime) -> str:
    """Plain-text call sheet: every fact comes from the approved plan."""
    s, plan = state, state.plan
    by_shot = {m.shot_id: m for m in plan.shots}
    lines = [
        f"GHOST GRIP CALL SHEET — {s.scene.title}",
        f"Call: {shoot_start:%a %d %b %Y, %H:%M}   Set: {s.scene.set.width:.1f} x {s.scene.set.depth:.1f} m",
        f"Logline: {s.scene.logline}",
        "",
        f"RIG: 4 m rail along x from x={plan.rig.rail_x_start:.2f} at y={plan.rig.rail_y:.2f}; 6-axis arm on 0.6 m pedestal.",
        f"Simulator: {'ALL MOVES FEASIBLE' if plan.feasible else f'{len(plan.violations)} OPEN ISSUES — see below'} after {s.revision} revision round(s).",
        f"Rig program: {sum(len(m.joints) for m in plan.shots)} samples at 25 Hz (rail + 6 joints + lens), exported from Ghost Grip as CSV.",
        "",
        "TALENT MARKS: " + "; ".join(f"{a.name} at ({a.x:.1f}, {a.y:.1f})" for a in s.scene.actors),
        "",
        "SHOTS",
    ]
    for sh in s.shot_list.shots:
        m = by_shot.get(sh.id)
        lenses = sorted({round(k.lens_mm) for k in sh.keyframes})
        lines.append(f"{sh.id}  {sh.name} — {sh.technique}, {'/'.join(map(str, lenses))}mm, {sh.keyframes[-1].t:.1f}s"
                     + (f", peak axis load {m.peak_joint_speed_pct:.0f}%" if m else ""))
        lines.append(f"    Intent: {sh.intent}")
        lines.append(f"    Reference: {sh.reference_title} — {sh.reference_url}")
    if plan.violations:
        lines += ["", "OPEN RIG ISSUES"] + [f"- {v.shot_id} @ {v.t}s: {v.detail}" for v in plan.violations]
    lines += ["", "SAFETY: no one inside the arm envelope (1.3 m from the pedestal) while the rig is live. "
              f"Performers keep their marks; the plan keeps >= {0.8:.1f} m clearance.", "", "DIRECTOR'S NOTES", s.notes.strip()]
    return "\n".join(lines)


def connected_platforms(one_tools) -> set[str]:
    """Platforms with an operational connection, read straight from One's list tool (no LLM involved)."""
    lister = next(t for t in one_tools if t.name == "list_one_integrations")
    raw = lister._run()
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return {c["platform"] for c in data.get("connections", []) if c.get("state", "operational") == "operational"}
    except (ValueError, AttributeError, TypeError):
        return {p for p in ("gmail", "google-calendar", "slack", "discord") if p in str(raw)}


def run_producer(state: GhostGripState, shoot_start: datetime, to: str = "", slack_channel: str = "", discord_channel: str = "",
                 emit: Callable[[str, dict], None] = lambda k, d: None) -> str:
    """Deliver the call sheet through whichever requested apps are connected on One. Returns the producer's report."""
    if not config.ONE_SECRET:
        raise RuntimeError("ONE_SECRET is not set; see README → One setup.")
    sheet = call_sheet(state, shoot_start)
    subject = f"Call sheet: {state.scene.title} ({shoot_start:%d %b})"
    shoot_end = shoot_start + timedelta(hours=4)
    shots = ", ".join(f"{sh.id} {sh.technique}" for sh in state.shot_list.shots)
    status = "all moves feasible" if state.plan.feasible else f"{len(state.plan.violations)} open rig issues"

    @tool("build_gmail_raw")
    def build_gmail_raw() -> str:
        """Build the base64url `raw` value for Gmail's send action from the approved call sheet. Always use this; never hand-encode."""
        message = MIMEText(sheet)
        message["To"] = to
        message["Subject"] = subject
        return base64.urlsafe_b64encode(message.as_bytes()).decode()

    @tool("build_calendar_event")
    def build_calendar_event() -> str:
        """Return the exact JSON body for a Google Calendar insert-event action for this shoot block."""
        return json.dumps({
            "summary": f"Shoot: {state.scene.title} (Ghost Grip rig)",
            "description": sheet[:7000],
            "start": {"dateTime": shoot_start.astimezone().isoformat()},
            "end": {"dateTime": shoot_end.astimezone().isoformat()},
            "attendees": [{"email": to}],
        })

    @tool("build_slack_message")
    def build_slack_message() -> str:
        """Return the exact JSON body for Slack's post-message action (chat.postMessage) carrying the call sheet."""
        return json.dumps({
            "channel": slack_channel,
            "text": f"🎥 Call sheet: {state.scene.title} — {status} after {state.revision} revision(s)",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": f"🎥 Call sheet: {state.scene.title}"[:150]}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*{shots}*\nRig simulator: {status} after {state.revision} revision(s)."[:2900]}},
                {"type": "section", "text": {"type": "mrkdwn", "text": f"```{sheet[:2800]}```"}},
            ],
        })

    @tool("build_discord_message")
    def build_discord_message() -> str:
        """Return the exact JSON body for a Discord send-message action announcing the plan."""
        return json.dumps({"content": f"{config.TEAM_HANDLE} · 🎥 {state.scene.title} · {shots} · {status} after {state.revision} revision(s). Call sheet sent to the crew."[:1900]})

    emit("step", {"step": "produce", "status": "running", "label": "Producer is delivering the call sheet through One"})
    with MCPServerAdapter(one_mcp_params(), connect_timeout=120) as one_tools:
        one_tools = harden_execute(dropping_nulls(list(one_tools)))
        connected = connected_platforms(one_tools)
        cfg = AGENTS["producer"]
        producer = Agent(role=cfg["role"], goal=cfg["goal"], backstory=cfg["backstory"] + " " + RULES,
                         tools=[*one_tools, build_gmail_raw, build_calendar_event, build_slack_message, build_discord_message],
                         llm=llm.crew_llm(temperature=0.1), max_iter=25, verbose=True)
        tasks, skipped = [], []
        if slack_channel:
            if "slack" in connected:
                tasks.append(Task(agent=producer, expected_output="The Slack message ts and channel, or the exact error.", description=(
                    "Post the call sheet to Slack through One. Call list_one_integrations for the slack connection key, then "
                    "find_one_actions for the slack platform with intent 'send a message to a channel', read its documentation, "
                    "call build_slack_message and pass its output verbatim as dataJson.")))
            else:
                skipped.append("Slack (not connected on One)")
        if to:
            if "gmail" in connected:
                tasks.append(Task(agent=producer, expected_output="The Gmail message id, or the exact error.", description=(
                    "Email the call sheet via Gmail through One. Call list_one_integrations to get the gmail connection key, then "
                    "find_one_actions for the gmail platform with intent 'send an email', read its documentation, call build_gmail_raw, "
                    "and execute the send action with dataJson {\"raw\": \"<output of build_gmail_raw>\"} and pathVariablesJson {\"userId\": \"me\"}.")))
            else:
                skipped.append("Gmail (not connected on One)")
            if "google-calendar" in connected:
                tasks.append(Task(agent=producer, expected_output="The created event's id and htmlLink, or the exact error.", description=(
                    "Create the shoot block in Google Calendar through One. Find the google-calendar platform's 'create an event' action, "
                    "read its documentation, call build_calendar_event and pass its output verbatim as dataJson, with pathVariablesJson "
                    "{\"calendarId\": \"primary\"} (or the path variable name the documentation gives).")))
            else:
                skipped.append("Google Calendar (not connected on One)")
        if discord_channel:
            if "discord" in connected:
                tasks.append(Task(agent=producer, expected_output="The Discord message id, or the exact error.", description=(
                    "Post the announcement to Discord through One. Find the discord platform's 'send a message to a channel' action, read "
                    f"its documentation, call build_discord_message and pass its output verbatim as dataJson, with the channel id {discord_channel} "
                    "as the channelId PATH variable (never in the body).")))
            else:
                skipped.append("Discord (not connected on One)")
        if not tasks:
            raise RuntimeError("Nothing to send: " + ("; ".join(skipped) or "give a Slack channel or a crew email") +
                               f". Connected on One: {', '.join(sorted(connected)) or 'nothing'}.")
        result = Crew(agents=[producer], tasks=tasks, process=Process.sequential, verbose=True).kickoff()
    report = "\n".join(f"- {t.description.split('.')[0]}: {o.raw}" for t, o in zip(tasks, result.tasks_output))
    if skipped:
        report += "\n- Skipped: " + "; ".join(skipped)
    emit("one_result", {"report": report, "call_sheet": sheet})
    emit("step", {"step": "produce", "status": "done", "label": "Call sheet delivered"})
    return report
