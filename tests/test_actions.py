"""Actions. The simulated executor is what stops a stray click during the demo."""
import pytest

from seentap import actions, config


def test_simulated_executor_records_instead_of_clicking():
    ex = actions.SimExecutor()
    ex.execute("click", 100, 200)
    assert ex.events == [{"verb": "click", "x": 100, "y": 200}]


@pytest.mark.parametrize("verb", config.VOCAB)
def test_every_verb_in_the_vocabulary_is_executable(verb):
    ex = actions.SimExecutor()
    if verb == "drop":
        ex.execute("drag", 0, 0)      # drop is only legal after a drag
    ex.execute(verb, 10, 20)
    assert ex.events[-1]["verb"] == verb


def test_unknown_verb_is_refused_not_guessed():
    with pytest.raises(ValueError):
        actions.SimExecutor().execute("banana", 0, 0)


def test_drag_then_drop_is_tracked_as_a_pair():
    ex = actions.SimExecutor()
    ex.execute("drag", 10, 10)
    assert ex.dragging
    ex.execute("drop", 90, 90)
    assert not ex.dragging


def test_drop_without_a_drag_is_refused():
    ex = actions.SimExecutor()
    with pytest.raises(actions.ActionError):
        ex.execute("drop", 0, 0)


def test_cancel_clears_a_pending_drag():
    ex = actions.SimExecutor()
    ex.execute("drag", 10, 10)
    ex.execute("cancel", 0, 0)
    assert not ex.dragging


def test_real_executor_keeps_the_corner_failsafe_armed():
    """Second line of defence behind the simulated desktop."""
    pyautogui = pytest.importorskip("pyautogui")
    actions.RealExecutor()
    assert pyautogui.FAILSAFE is True
