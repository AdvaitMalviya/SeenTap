"""The log is the only artefact the report's results come from, and the only
place the 'no video is retained' claim can be enforced."""
import pytest

from seentap import eventlog


def test_roundtrip_preserves_order_and_fields(tmp_path):
    p = tmp_path / "s.jsonl"
    with eventlog.EventLog(p) as log:
        for i in range(5):
            log.write("gaze", x=float(i), y=1.0, conf=0.9)
    rows = list(eventlog.read(p))
    assert [r["x"] for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(r["kind"] == "gaze" for r in rows)


def test_timestamps_are_monotonic_and_default_to_now(tmp_path):
    p = tmp_path / "s.jsonl"
    with eventlog.EventLog(p) as log:
        for _ in range(20):
            log.write("gaze", x=0.0, y=0.0)
    ts = [r["t"] for r in eventlog.read(p)]
    assert ts == sorted(ts)
    assert all(isinstance(t, float) for t in ts)


def test_explicit_timestamp_is_kept_verbatim(tmp_path):
    """Samples are stamped at the source, so the logger must not restamp them."""
    p = tmp_path / "s.jsonl"
    with eventlog.EventLog(p) as log:
        log.write("gaze", t=123.456, x=0.0, y=0.0)
    assert next(iter(eventlog.read(p)))["t"] == 123.456


@pytest.mark.parametrize("field", ["frame", "image", "audio", "pcm", "waveform"])
def test_raw_media_cannot_reach_disk(tmp_path, field):
    """The consent form says no imagery or audio is stored. Enforce it here."""
    p = tmp_path / "s.jsonl"
    with eventlog.EventLog(p) as log:
        with pytest.raises(eventlog.EthicsError):
            log.write("gaze", **{field: [1, 2, 3]})
    assert not list(eventlog.read(p))


def test_non_serialisable_payload_is_rejected_not_silently_dropped(tmp_path):
    p = tmp_path / "s.jsonl"
    with eventlog.EventLog(p) as log:
        with pytest.raises(TypeError):
            log.write("gaze", blob=object())


def test_now_is_monotonic():
    a, b = eventlog.now(), eventlog.now()
    assert b >= a
