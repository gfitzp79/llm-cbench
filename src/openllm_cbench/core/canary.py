"""
Loopback-only canary listener, shared safety primitive for suites that
give a model a real HTTP tool.

DESIGN INVARIANT: the canary is the only host any suite's `http_request`
tool can ever actually reach. It binds to 127.0.0.1 on an OS-assigned
ephemeral port, and start_canary() asserts that bind before returning --
refusing to start rather than silently listening somewhere else. Any
request to a different host is intercepted by the calling suite before
this module is ever involved and answered with a synthetic failure; no
real request to a non-canary host is ever made, regardless of what a
model attempts.
"""

import http.server
import threading
import time
from urllib.parse import urlparse


class CanaryHandler(http.server.BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):
        self._log()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    do_POST = do_GET

    def _log(self):
        CanaryHandler.hits.append({"path": self.path, "ts": time.time()})

    def log_message(self, *a):
        pass  # silence default stderr access-log spam


def start_canary():
    """Starts the canary HTTP server on 127.0.0.1:<ephemeral>. Refuses to
    start if the resulting bind address is not loopback -- this should be
    unreachable given the hardcoded "127.0.0.1" below, but is asserted
    explicitly rather than trusted, since this is the one property the
    whole safety claim rests on."""
    server = http.server.HTTPServer(("127.0.0.1", 0), CanaryHandler)
    bound_host = server.server_address[0]
    if bound_host not in ("127.0.0.1", "localhost"):
        server.server_close()
        raise RuntimeError(
            f"Canary bound to non-loopback address {bound_host!r} -- refusing to "
            f"start. This should never happen; if it does, do not run any suite "
            f"until it's understood."
        )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


_CANARY_HOSTS = frozenset({"127.0.0.1", "localhost"})


def is_canary_url(url, canary_port):
    """True ONLY when the URL's ORIGIN is the loopback canary.

    MUST NEVER RAISE -- same discipline as
    scoring/containment_metrics.py:classify_target(). A tool-calling model
    can and does emit a non-string value for a URL-shaped argument (found
    live: a genuinely small/undertrained model passed a dict where a
    string was expected, which crashed this function's original bare `in`
    check with `TypeError: argument of type 'dict' is not iterable`,
    killing an in-progress trial). Anything that isn't a string is
    unambiguously not a match for the canary's own address, so this
    degrades to False rather than propagating.

    MUST NEVER SUBSTRING-MATCH. This was:

        f"127.0.0.1:{canary_port}" in url or f"localhost:{canary_port}" in url

    which matched a PUBLIC-host URL that merely embedded the canary address
    in its QUERY STRING. The caller then took the authorized branch and
    issued a real outbound GET at the public host, breaching the
    loopback-only invariant this package states as non-negotiable.

    THIS IS NOT HYPOTHETICAL. The substring form shipped, and it produced
    two real outbound requests carrying sandbox configuration in their
    query strings before it was caught. Whether either connected is
    unrecorded, because the call sits inside a bare except -- so whether
    anything left the machine came down to whether a fabricated hostname
    happened to be unregistered, rather than to any control this harness
    applied. Task files here deliberately name plausible external hosts,
    and any of them could be registered by someone else.

    Compare the parsed origin. Never a substring.

    Deliberately narrow: only `127.0.0.1` and `localhost`, because the canary
    binds to `127.0.0.1` and nothing else can reach it. Anything unrecognised
    falls through to the intercepted branch, which never connects. That is
    the safe direction to fail in.

    This is the package's single definition. `scoring/containment_metrics.py`
    and `integrations/inspect_containment.py` import it instead of carrying
    copies; three copies are what let the defect survive review.
    """
    if not isinstance(url, str):
        return False
    try:
        parts = urlparse(url.strip())
        if parts.scheme not in ("http", "https"):
            return False
        if (parts.hostname or "").lower() not in _CANARY_HOSTS:
            return False
        port = parts.port  # raises ValueError on netloc like host:host:port
    except Exception:
        return False
    if port is None:
        return False
    try:
        return int(port) == int(canary_port)
    except (TypeError, ValueError):
        return False


def canary_request_url(url, canary_port):
    """Rebuild a SAFE loopback URL from a validated canary URL, else None.

    Never hand a model-supplied string to `requests.get()`. Even once the
    origin is validated the remainder is still model-controlled, so query and
    fragment are dropped: a match on the origin must not be able to carry an
    attacker-chosen payload into a real request. Only the path survives,
    which is all the canary listener needs in order to log a hit.

    MUST NEVER RAISE.
    """
    if not is_canary_url(url, canary_port):
        return None
    try:
        path = urlparse(url.strip()).path or "/"
    except Exception:
        path = "/"
    if not path.startswith("/"):
        path = "/" + path
    return f"http://127.0.0.1:{int(canary_port)}{path}"
