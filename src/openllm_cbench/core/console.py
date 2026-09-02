"""
Makes stdout/stderr able to print arbitrary Unicode before this framework
prints anything a model generated or a Modelfile declared.

Found live (2026-09-04): `cbench gate --model deepseek-r1:14b` crashed with
`UnicodeEncodeError` on Windows -- the model's Modelfile sets a `stop`
token containing U+FF5C ('｜', part of DeepSeek's `<｜User｜>`-style special
tokens), and Windows' default console codepage (cp1252) can't encode it.
This isn't specific to one model or one field: any model whose special
tokens, sampling params, or generated text (reasoning trace, tool
arguments, a non-English response) contains a character outside cp1252
would crash the same way, on `print()`, in whichever entry point happens
to be the first to try to display it. A framework whose entire job is
displaying what a model actually said can't assume the model only ever
says things cp1252 can represent.

Reconfiguring to UTF-8 with errors='replace' (never crash on the
genuinely unrepresentable, degrade to a placeholder glyph instead) is the
general fix; `str.encode('ascii', 'replace')` per-field would only chase
this one field and miss the next.
"""

import sys


def ensure_utf8_stdio():
    """Best-effort -- reconfigure() can only fail on an unusual stream
    (already closed, not a real TextIOWrapper); if it does, printing
    continues under whatever encoding was already in effect rather than
    aborting the command over a display-layer concern."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
