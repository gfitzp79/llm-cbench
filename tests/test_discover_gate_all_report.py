"""
Regression test for `cbench discover --gate-all` swallowing caveat detail.

`_cmd_discover` imported `render_gate_report` alongside `run_gate` and
`to_registry_entry` -- the same trio `cbench gate` itself uses to print a
full report -- but only ever called two of the three. A model gated with
caveats in a --gate-all batch printed nothing but "caveats found -- saved
to <path>": the same caveat detail `cbench gate --model <tag>` prints for
one model at a time (architecture, tool-call check, channel-separation
state, the actual caveat text) was silently dropped, forcing a separate
per-tag re-run just to learn what was wrong. This is the reason the model
catalogue documents caveats at all -- see verified.json's own docstring on
why they're recorded per model.

Found via a static undefined-name/unused-import sweep (pyflakes) after
fixing an unrelated NameError crash in `cbench assess`; render_gate_report
being imported-but-unused was the same shape of "split apart, half the
wiring forgotten" bug.

`--gate-all`'s own real code paths (endpoint I/O, run_gate's live model
calls, registry file writes) are exercised elsewhere; this test patches
those at the module level they're imported from -- since the CLI's own
imports are local/lazy inside the function -- and asserts only on what
_cmd_discover itself decides to print.
"""

from unittest.mock import patch

from openllm_cbench import cli


def _model(name):
    return {"name": name, "size": 5_000_000_000, "modified_at": "2026-01-01T00:00:00Z",
            "architecture": "llama", "params_b": 8, "quant": "Q4_K_M"}


def test_gate_all_prints_full_report_when_caveats_found(tmp_path, capsys):
    caveat_result = {
        "model": "flaky:1b",
        "clean": False,
        "show_info_ok": True,
        "architecture": "llama",
        "param_size": "1B",
        "quant": "Q4_K_M",
        "capabilities": ["completion", "tools"],
        "tool_call_ok": False,
        "tool_call_detail": "tool call malformed -- missing required argument",
        "caveats": ["tool-call wellformedness failed"],
    }
    registry_file = tmp_path / "models.json"

    with patch("openllm_cbench.core.discover.list_local_models", return_value=[_model("flaky:1b")]), \
         patch("openllm_cbench.core.registry.load_registry", return_value={"models": {}}), \
         patch("openllm_cbench.core.gate.run_gate", return_value=caveat_result), \
         patch("openllm_cbench.core.gate.to_registry_entry", return_value={}), \
         patch("openllm_cbench.core.registry.save_entry", return_value=str(registry_file)):
        rc = cli._cmd_discover(["--gate-all"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "caveats found" in out
    # The regression: this detail used to be dropped entirely on a
    # non-clean result. render_gate_report() always emits this exact
    # tool-call-check line -- its presence proves the full report was
    # actually printed, not just the terse status.
    assert "tool call malformed -- missing required argument" in out
