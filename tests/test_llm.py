import json

import pytest

from extract.llm import LLMClient, LLMUnavailableError, RateLimiter, cache_key_for, _parse_json_with_repair


def test_cache_key_for_is_deterministic_and_content_derived():
    key1 = cache_key_for("chunk-hash-abc", "v1")
    key2 = cache_key_for("chunk-hash-abc", "v1")
    key3 = cache_key_for("chunk-hash-abc", "v2")
    assert key1 == key2
    assert key1 != key3


def test_complete_json_returns_cached_response_without_network_call(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = cache_key_for("hash1", "v1")
    (cache_dir / f"{key}.json").write_text(json.dumps([{"subject": "X"}]), encoding="utf-8")

    client = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(cache_dir))
    result = client.complete_json("any prompt", key)
    assert result == [{"subject": "X"}]


def test_complete_json_raises_when_no_cache_and_no_key(tmp_path):
    client = LLMClient(api_key=None, model="gemini-2.5-flash", cache_dir=str(tmp_path / "cache"))
    with pytest.raises(LLMUnavailableError):
        client.complete_json("any prompt", "missing-key")


def test_complete_json_writes_cache_after_provider_call(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    client = LLMClient(api_key="fake-key", model="gemini-2.5-flash", cache_dir=str(cache_dir))
    monkeypatch.setattr(client, "_call_provider", lambda prompt: '[{"subject": "Y"}]')
    result = client.complete_json("prompt", "new-key")
    assert result == [{"subject": "Y"}]
    assert (cache_dir / "new-key.json").exists()

    # second call must not need the provider again even if it would now raise
    monkeypatch.setattr(client, "_call_provider", lambda prompt: (_ for _ in ()).throw(RuntimeError("should not be called")))
    assert client.complete_json("prompt", "new-key") == [{"subject": "Y"}]


def test_parse_json_with_repair_handles_fenced_json():
    raw = '```json\n[{"a": 1}]\n```'
    assert _parse_json_with_repair(raw) == [{"a": 1}]


def test_parse_json_with_repair_handles_trailing_prose():
    raw = 'Here is the result:\n[{"a": 1}]\nHope that helps!'
    assert _parse_json_with_repair(raw) == [{"a": 1}]


def test_rate_limiter_enforces_minimum_interval(monkeypatch):
    # A controllable fake clock: fake_sleep advances the clock by exactly the
    # duration slept, so the mock stays internally consistent (sleeping
    # actually "passes time"), unlike a canned sequence of increasing values.
    clock = [0.0]
    sleeps = []

    def fake_monotonic():
        return clock[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("time.monotonic", fake_monotonic)
    monkeypatch.setattr("time.sleep", fake_sleep)

    limiter = RateLimiter(per_minute=60)  # min_interval = 1.0s

    # Sleep branch: two calls back-to-back with no real time passing other
    # than what the mocked sleep advances. _last_call starts at 0.0, so the
    # first call has elapsed=0 and must sleep the full min_interval; since
    # nothing but the fake sleep moved the clock, the second call is also
    # "immediate" relative to when the first call's sleep ended, so it must
    # sleep the full interval again.
    limiter.wait()
    limiter.wait()
    assert sleeps == [1.0, 1.0]

    # No-sleep branch: simulate the caller doing other work for longer than
    # min_interval between calls. The next .wait() should not sleep at all.
    sleeps.clear()
    clock[0] += 5.0
    limiter.wait()
    assert sleeps == []
