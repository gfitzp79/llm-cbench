"""
Coverage for core/pull.py's pure functions. pull_model() itself needs a
live endpoint (not tested here); throttled_progress_printer() is tested
against fabricated NDJSON-shaped events, no model or network required.
"""

from openllm_cbench.core.pull import throttled_progress_printer


def test_prints_once_per_status_with_no_size_info():
    lines = []
    on_progress = throttled_progress_printer(lines.append)
    on_progress({"status": "pulling manifest"})
    on_progress({"status": "verifying sha256 digest"})
    assert len(lines) == 2
    assert "pulling manifest" in lines[0]


def test_throttles_to_roughly_every_10_percent_not_every_event():
    lines = []
    on_progress = throttled_progress_printer(lines.append)
    total = 1000
    # Simulate Ollama's real behavior: many events for the same digest
    # with a growing `completed` count, one point at a time.
    for completed in range(0, total + 1, 1):
        on_progress({"status": "pulling", "digest": "sha256:abc", "total": total, "completed": completed})
    # ~10 threshold crossings (0,10,20...100) plus the final 100% line --
    # nowhere near one line per event (1001 events).
    assert 9 <= len(lines) <= 12


def test_new_digest_resets_the_throttle():
    lines = []
    on_progress = throttled_progress_printer(lines.append)
    on_progress({"status": "pulling", "digest": "sha256:aaa", "total": 100, "completed": 100})
    # The second layer crosses its own first 10%-threshold (0 -> 15) --
    # must print because it's judged against a reset threshold for the
    # NEW digest, not still measured against the first layer's 100%.
    # (A first event that DOESN'T cross 10% yet, e.g. completed=5, is
    # correctly still throttled -- same discipline applied to every layer.)
    on_progress({"status": "pulling", "digest": "sha256:bbb", "total": 100, "completed": 15})
    assert len(lines) == 2


def test_error_event_prints_immediately_regardless_of_throttle():
    lines = []
    on_progress = throttled_progress_printer(lines.append)
    on_progress({"status": "pulling", "digest": "sha256:abc", "total": 100, "completed": 1})
    on_progress({"error": "pull model manifest: file does not exist"})
    assert any("file does not exist" in line for line in lines)


def test_always_prints_the_100_percent_completion_line():
    lines = []
    on_progress = throttled_progress_printer(lines.append)
    # completed jumps straight from 0 to total in one event (small file,
    # single chunk) -- must not be silently dropped by the 10-point rule.
    on_progress({"status": "pulling", "digest": "sha256:abc", "total": 100, "completed": 100})
    assert len(lines) == 1
    assert "100%" in lines[0]
