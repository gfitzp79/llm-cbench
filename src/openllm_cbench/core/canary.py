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


def is_canary_url(url, canary_port):
    return f"127.0.0.1:{canary_port}" in url or f"localhost:{canary_port}" in url
