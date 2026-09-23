"""The automatic generation budget: who counts as reasoning, and the
precedence between a flag, the catalogue and the automatic value. The
suites' wiring of it is in test_catalogue_wiring.py."""

from openllm_cbench.core import budget
from openllm_cbench.core.budget import (
    REASONING_NUM_CTX, REASONING_NUM_PREDICT, model_reasons, resolve_budget,
)


def test_the_catalogue_measurement_decides_before_the_endpoint(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("asked the endpoint about a catalogued model")
    monkeypatch.setattr(budget, "_endpoint_reports_thinking", boom)
    assert model_reasons("m:1b", {"thinking": True}) is True
    assert model_reasons("m:1b", {"thinking": False}) is False


def test_reasoning_switched_off_is_not_reasoning():
    assert model_reasons("m:1b", {"thinking": True, "config_overrides": {"think": False}}) is False
    assert model_reasons("m:1b", {"thinking": True}, think=False) is False
    # A caller that knows reasoning is on (S2) is not overruled by the
    # catalogue's tool-calling setting.
    assert model_reasons("m:1b", {"thinking": True, "config_overrides": {"think": False}},
                         think=True) is True


def test_an_uncatalogued_model_asks_the_endpoint_unless_told_not_to(monkeypatch):
    monkeypatch.setattr(budget, "_endpoint_reports_thinking", lambda *a, **k: True)
    assert model_reasons("m:1b", None) is True
    assert model_reasons("m:1b", None, ask_endpoint=False) is None


def test_precedence_is_flag_then_catalogue_then_automatic():
    assert resolve_budget(None, None, {}, 8192, 2048, reasoning=True)[:2] == \
        (REASONING_NUM_CTX, REASONING_NUM_PREDICT)
    assert resolve_budget(None, None, {}, 8192, 2048, reasoning=False)[:2] == (8192, 2048)
    assert resolve_budget(None, None, {}, 8192, 2048, reasoning=None)[:2] == (8192, 2048)
    assert resolve_budget(None, None, {"num_predict": 4000}, 8192, 2048, reasoning=True)[:2] == \
        (REASONING_NUM_CTX, 4000)
    assert resolve_budget(9000, 3000, {"num_predict": 4000}, 8192, 2048, reasoning=True)[:2] == \
        (9000, 3000)


def test_the_window_always_has_room_for_the_reply():
    """Raising the reply budget without the window is the fix that does not
    take: the reply pushes the start of the conversation out, silently."""
    assert REASONING_NUM_CTX >= 2 * REASONING_NUM_PREDICT
