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
    # The permission gate is the host's business, not this assertion's.
    actions.RealExecutor(require_permission=False)
    assert pyautogui.FAILSAFE is True


def test_missing_accessibility_is_refused_rather_than_swallowed(monkeypatch):
    """Without it macOS discards every injected click in silence -- no error,
    no exception, nothing logged -- which looks exactly like gaze landing in
    the wrong place. It never prompts on its own either."""
    pytest.importorskip("pyautogui")
    monkeypatch.setattr(actions, "can_post_events", lambda: False)
    with pytest.raises(actions.PermissionError_, match="Accessibility"):
        actions.RealExecutor()


def test_permission_granted_constructs_normally(monkeypatch):
    pytest.importorskip("pyautogui")
    monkeypatch.setattr(actions, "can_post_events", lambda: True)
    assert actions.RealExecutor().dragging is False


def test_an_undetectable_permission_state_does_not_block(monkeypatch):
    """Not macOS, or Quartz missing: refusing to run would be worse."""
    pytest.importorskip("pyautogui")
    monkeypatch.setattr(actions, "can_post_events", lambda: None)
    assert actions.RealExecutor() is not None
