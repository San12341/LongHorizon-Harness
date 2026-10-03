from __future__ import annotations

import json
import shlex
from pathlib import Path

from lh_harness import agent_logs
from lh_harness.adapters.hermes import HermesAdapter
from lh_harness.types import EpisodeBudget, EpisodeResult


# Real events captured from `hermes chat --format stream-json` (Hermes Agent,
# GLM-5.3 backend).  Field order mirrors the CLI output exactly.
_INIT = {
    "type": "system",
    "subtype": "init",
    "model": "GLM-5.3",
    "session_id": "20261003_183743_9093e0",
    "timestamp": 1791023864310,
}

_TOOL_USE = {
    "type": "tool_use",
    "name": "write_file",
    "input": {"content": "hello", "path": "h_hermes_test.txt"},
    "timestamp": 1791023872299,
}

_TOOL_RESULT = {
    "type": "tool_result",
    "name": "write_file",
    "output": '{"bytes_written": 5, "verified": true}',
    "duration_ms": 141,
    "is_error": False,
    "timestamp": 1791023872442,
}

_TEXT_CHUNK = {
    "type": "text",
    "text": "\n\ndone",
    "timestamp": 1791023874134,
}

_RESULT = {
    "type": "result",
    "session_id": "20261003_183743_9093e0",
    "exit_code": 0,
    "text": "done",
    "tokens": {"input": 180, "output": 50, "total": 39078, "cache_read": 38848},
    "duration_ms": 9907,
    "timestamp": 1791023874217,
}


def _raw(*records: dict) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def test_hermes_stream_detected() -> None:
    assert agent_logs.detect_format(_raw(_INIT, _TOOL_USE, _TOOL_RESULT, _TEXT_CHUNK, _RESULT)) == (
        agent_logs.HERMES_STREAM_JSON
    )


def test_hermes_visible_output_prefers_result_text() -> None:
    raw = _raw(_INIT, _TOOL_USE, _TOOL_RESULT, _TEXT_CHUNK, _RESULT)
    assert agent_logs.visible_output(raw) == "done"


def test_hermes_assistant_texts() -> None:
    raw = _raw(_INIT, _TOOL_USE, _TOOL_RESULT, _TEXT_CHUNK, _RESULT)
    assert agent_logs.assistant_texts(raw) == ["\n\ndone", "done"]


def test_hermes_tool_output_view() -> None:
    raw = _raw(_INIT, _TOOL_USE, _TOOL_RESULT, _RESULT)
    assert '"bytes_written": 5' in agent_logs.tool_output_view(raw)


def test_hermes_runtime_signals_clean_exit() -> None:
    raw = _raw(_INIT, _TOOL_USE, _TOOL_RESULT, _TEXT_CHUNK, _RESULT)
    assert agent_logs.runtime_event_view(raw) == ""


def test_hermes_runtime_signals_failed_exit() -> None:
    failed = dict(_RESULT, exit_code=1, error="provider auth failed")
    raw = _raw(_INIT, failed)
    assert "response.failed: provider auth failed" in agent_logs.runtime_event_view(raw)


def test_hermes_trajectory_steps() -> None:
    raw = _raw(_INIT, _TOOL_USE, _TOOL_RESULT, _TEXT_CHUNK, _RESULT)
    steps = agent_logs.parse_trajectory(raw)
    kinds = [step["kind"] for step in steps]
    assert kinds == ["session", "tool_use", "tool_result", "text", "result"]
    assert steps[0]["model"] == "GLM-5.3"
    assert steps[1]["name"] == "write_file"
    assert steps[2]["is_error"] is False
    assert steps[-1]["text"] == "\n\ndone"
    assert steps[-1]["input_tokens"] == 180


def test_hermes_stream_not_confused_with_claude_or_opencode() -> None:
    # Claude Code opens with a `system/init` carrying `message`; that stream
    # must keep resolving to the Claude parser.
    claude_like = (
        '{"type": "system", "subtype": "init", "message": "hi", "cwd": "/x"}\n'
        '{"type": "result", "subtype": "success", "is_error": false, "result": "answer"}'
    )
    assert agent_logs.detect_format(claude_like) == agent_logs.CLAUDE_STREAM_JSON
    # OpenCode events without the Hermes header must keep resolving to the
    # OpenCode parser.
    opencode_like = (
        '{"type": "step_start", "part": {"type": "step-start"}}\n'
        '{"type": "text", "part": {"type": "text", "text": "hi"}}'
    )
    assert agent_logs.detect_format(opencode_like) == agent_logs.OPENCODE_RUN_JSON


def test_hermes_adapter_builds_command(monkeypatch, tmp_path: Path) -> None:
    from lh_harness.utils import agent_cli

    monkeypatch.setattr(
        agent_cli, "resolve_hermes_binary", lambda **_: "/opt/Hermes Bin/hermes"
    )
    import lh_harness.adapters.hermes as hermes_module

    monkeypatch.setattr(
        hermes_module, "resolve_hermes_binary", lambda **_: "/opt/Hermes Bin/hermes"
    )
    adapter = HermesAdapter(
        model="GLM-5.3",
        workspace_path="/tmp/ws",
        prompt_dir=str(tmp_path),
        reasoning_effort="high",
    )
    template = adapter.command_template
    assert template.startswith(shlex.quote("/opt/Hermes Bin/hermes"))
    assert "chat" in template
    assert "--query-file" in template and shlex.quote("-") in template
    assert "--format" in template and "stream-json" in template
    assert "--yolo" in template
    assert "--in" in template and shlex.quote("/tmp/ws") in template
    assert "--model" in template and shlex.quote("GLM-5.3") in template
    assert "--reasoning" in template and shlex.quote("high") in template
    assert template.endswith("< {prompt_path}")


def test_hermes_adapter_omits_model_when_unset(monkeypatch, tmp_path: Path) -> None:
    import lh_harness.adapters.hermes as hermes_module

    monkeypatch.setattr(hermes_module, "resolve_hermes_binary", lambda **_: "hermes")
    adapter = HermesAdapter(workspace_path="/tmp/ws", prompt_dir=str(tmp_path))
    assert "--model" not in adapter.command_template


def test_hermes_adapter_ignores_credentials(monkeypatch, tmp_path: Path) -> None:
    import lh_harness.adapters.hermes as hermes_module

    monkeypatch.setattr(hermes_module, "resolve_hermes_binary", lambda **_: "hermes")
    adapter = HermesAdapter(
        api_key="sk-secret",
        base_url="http://example.internal",
        workspace_path="/tmp/ws",
        prompt_dir=str(tmp_path),
    )
    template = adapter.command_template
    assert "sk-secret" not in template
    assert "example.internal" not in template


def test_hermes_adapter_registered_in_cli_choices() -> None:
    from lh_harness.cli import _AGENT_CHOICES

    assert "hermes" in _AGENT_CHOICES
