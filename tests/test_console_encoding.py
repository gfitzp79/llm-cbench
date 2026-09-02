"""
Regression coverage for core/console.py:ensure_utf8_stdio().

Found live (2026-09-04): `cbench gate --model deepseek-r1:14b` crashed
with UnicodeEncodeError on a real Windows console. DeepSeek's Modelfile
sets a `stop` token containing U+FF5C ('｜', part of its
`<｜User｜>`-style special tokens); the console's cp1252 codepage
can't encode it, and a plain `print()` raised instead of degrading.

These tests reproduce the actual failure mode -- a stdout stream forced
to cp1252, the same non-ASCII character that crashed the real run -- by
running a subprocess with PYTHONIOENCODING=cp1252 rather than asserting
against an encoding-agnostic pytest-captured stream, which would not
exercise the bug at all.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"

TROUBLESOME_CHAR = "｜"  # the exact character that crashed deepseek-r1:14b's gate check


def _run_snippet(code, encoding="cp1252"):
    import os
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": encoding},
    )


def test_the_bug_reproduces_without_the_fix():
    """Control: confirms this test harness actually exercises the failure
    mode. A bare print() of the troublesome character on a stream forced
    to cp1252, with no fix applied, must still crash -- if this test ever
    starts passing on its own, the regression test below has stopped
    proving anything."""
    result = _run_snippet(f"print('{TROUBLESOME_CHAR}')")
    assert result.returncode != 0
    assert "UnicodeEncodeError" in result.stderr


def test_ensure_utf8_stdio_fixes_it():
    result = _run_snippet(
        "from openllm_cbench.core.console import ensure_utf8_stdio; "
        "ensure_utf8_stdio(); "
        f"print('{TROUBLESOME_CHAR}')"
    )
    assert result.returncode == 0
    assert "UnicodeEncodeError" not in result.stderr


def test_render_gate_report_style_output_does_not_crash():
    """Closer to the actual failure: a Modelfile-style sampling-param dict
    containing the troublesome character, printed the way
    cli.py's _cmd_gate() actually prints a gate report."""
    result = _run_snippet(
        "from openllm_cbench.core.console import ensure_utf8_stdio; "
        "ensure_utf8_stdio(); "
        f"params = {{'stop': '<{TROUBLESOME_CHAR}User{TROUBLESOME_CHAR}>'}}; "
        "print(f'Modelfile sampling parameters: `{params}`')"
    )
    assert result.returncode == 0
    assert "UnicodeEncodeError" not in result.stderr


def test_cli_gate_help_still_works_under_a_restrictive_encoding():
    """Sanity check that the fix doesn't break the common, ASCII-only
    case -- cbench gate --help under a forced cp1252 stream."""
    result = subprocess.run(
        [sys.executable, "-m", "openllm_cbench.cli", "gate", "--help"],
        capture_output=True, text=True, timeout=15,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "cp1252"},
    )
    assert result.returncode == 0
    assert "--model" in result.stdout
