"""
Chat endpoint resolution, shared by every suite.

The harness is model-scale-agnostic -- it communicates with an HTTP chat
endpoint, and the canary/scoring/metrics logic compute identically
regardless of model size or hosting. The only hardcoded assumption is
which endpoint to call, which is why it is centralized and made
overridable in one place rather than duplicated per suite.

Precedence: explicit `--endpoint` CLI flag > `OPENLLM_CBENCH_ENDPOINT`
environment variable > default (a local Ollama install).

Default value and URL shape (`/api/chat`, `/api/show`) are unchanged from
the original harness, which targeted Ollama's native API specifically
(not its OpenAI-compatible `/v1` surface -- see ARCHITECTURE.md for why
that distinction matters for the channel-divergence suite)."""

import os

DEFAULT_BASE_URL = "http://localhost:11434"


def resolve_base_url(cli_value=None):
    return cli_value or os.environ.get("OPENLLM_CBENCH_ENDPOINT") or DEFAULT_BASE_URL


def chat_url(cli_value=None):
    return resolve_base_url(cli_value).rstrip("/") + "/api/chat"


def show_url(cli_value=None):
    return resolve_base_url(cli_value).rstrip("/") + "/api/show"
