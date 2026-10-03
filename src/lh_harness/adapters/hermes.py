from __future__ import annotations

import shlex

from ..agent_logs import visible_output as extract_hermes_visible_output
from ..agent_registry import normalise_reasoning_effort
from ..types import DEFAULT_HERMES_MODEL, DEFAULT_TMP_DIR, DEFAULT_WORKSPACE_PATH
from ..utils.agent_cli import resolve_hermes_binary
from .cli_agent import CommandAgentAdapter


class HermesAdapter(CommandAgentAdapter):
    """Run Hermes Agent headlessly through LongHorizon Harness.

    ``hermes chat --query-file - --format stream-json`` emits one JSON event
    per line on stdout (``system`` init, ``tool_use``/``tool_result`` pairs,
    incremental ``text`` chunks, and a final ``result`` carrying the whole
    reply), which ``agent_logs`` normalizes into the same views every other
    backend provides.  ``--yolo`` approves the permission prompts the harness
    already scopes for the episode.  Hermes authenticates itself through its
    own credential pool (``~/.hermes/auth.json``), so the harness never needs
    an API key or base URL for this backend.  Reasoning effort maps to the
    ``--reasoning`` flag Hermes exposes on ``hermes chat``.
    """

    def __init__(
        self,
        *,
        model: str | None = DEFAULT_HERMES_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        workspace_path: str = DEFAULT_WORKSPACE_PATH,
        prompt_dir: str = f"{DEFAULT_TMP_DIR}/prompts",
        role: str = "cli_executor",
        hidden_paths: tuple[str, ...] = (),
        reasoning_effort: str | None = None,
    ) -> None:
        # api_key/base_url are intentionally ignored: Hermes manages its own
        # providers and credentials, and exporting a harness key into its
        # environment could override the pool the user configured.
        del api_key, base_url
        normalized_model = (model or "").strip()
        effort = normalise_reasoning_effort(reasoning_effort)

        hermes_binary = resolve_hermes_binary() or "hermes"
        command_parts = [
            shlex.quote(hermes_binary),
            "chat",
            "--query-file",
            "-",
            "--quiet",
            "--format",
            "stream-json",
            "--yolo",
            # Hermes restores the recorded cwd of a resumed session by
            # default; pin the episode workspace explicitly instead.
            "--no-restore-cwd",
            "--in",
            shlex.quote(self._workspace_arg(workspace_path, role)),
        ]
        if normalized_model:
            command_parts.extend(["--model", shlex.quote(normalized_model)])
        if effort:
            command_parts.extend(["--reasoning", shlex.quote(effort)])

        super().__init__(
            # `-` makes Hermes read the prompt from stdin, keeping long
            # prompts off the command line and out of the process table.
            command_template=f"{' '.join(command_parts)} < {{prompt_path}}",
            prompt_dir=prompt_dir,
            workspace_path=workspace_path,
            visible_output_parser=extract_hermes_visible_output,
            hidden_paths=hidden_paths,
        )

    @staticmethod
    def _workspace_arg(workspace_path: str, role: str) -> str:
        """Pick the directory the Hermes process should start in.

        Manager and auditor roles receive the workspace as read-only
        reviewers, but they still launch from it so relative task paths in
        their prompts resolve; executor roles run where the files live.
        """

        del role
        return workspace_path
