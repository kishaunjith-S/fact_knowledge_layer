import hashlib
import json
import time
from pathlib import Path

import httpx

# httpx has no Happy-Eyeballs: when getaddrinfo returns AAAA records first and
# the host's IPv6 route is a black hole, every request pays a full OS connect
# timeout (~2 min on Windows) before falling back to IPv4. Binding the local
# socket to an IPv4 address forces IPv4 directly -- Google's endpoint is dual
# stack, so IPv4 is always available -- turning 2-minute calls back into 2s.
_HTTP_TRANSPORT = httpx.HTTPTransport(local_address="0.0.0.0", retries=1)
_HTTP_TIMEOUT = httpx.Timeout(90.0, connect=10.0)


class LLMUnavailableError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, per_minute: int):
        self.min_interval = 60.0 / max(per_minute, 1)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def cache_key_for(content: str, prompt_version: str) -> str:
    return hashlib.sha256(f"{content}:{prompt_version}".encode("utf-8")).hexdigest()[:24]


def _parse_json_with_repair(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start_options = [i for i in (text.find("["), text.find("{")) if i != -1]
    end_options = [i for i in (text.rfind("]"), text.rfind("}")) if i != -1]
    if start_options and end_options:
        start, end = min(start_options), max(end_options)
        if end > start:
            return json.loads(text[start:end + 1])
    raise ValueError(f"could not parse JSON from response: {raw[:200]!r}")


class LLMClient:
    def __init__(self, api_key: str | None, model: str, cache_dir: str, rate_limiter: RateLimiter | None = None):
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limiter = rate_limiter or RateLimiter(per_minute=12)

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def complete_json(self, prompt: str, cache_key: str):
        cache_path = self._cache_path(cache_key)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        if not self.api_key:
            raise LLMUnavailableError(
                f"No cached response for '{cache_key}' and no API key is configured. "
                "Run ingestion once with GEMINI_API_KEY set to populate data/cache/, "
                "or use the committed cache from the starter corpus."
            )
        raw = self._call_provider(prompt)
        parsed = _parse_json_with_repair(raw)
        cache_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        return parsed

    # Retryable: 429 (rate limit) and 5xx (free-tier "high demand" spikes,
    # which are intermittent -- backing off and retrying gets calls through).
    _MAX_ATTEMPTS = 6
    _BACKOFF_SCHEDULE = (5, 15, 30, 60, 90)

    def _call_provider(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        last_error = "unknown error"
        for attempt in range(self._MAX_ATTEMPTS):
            self.rate_limiter.wait()
            try:
                with httpx.Client(transport=_HTTP_TRANSPORT, timeout=_HTTP_TIMEOUT) as client:
                    response = client.post(url, json=body)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                if response.status_code not in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(self._BACKOFF_SCHEDULE[min(attempt, len(self._BACKOFF_SCHEDULE) - 1)])
        raise RuntimeError(f"provider unavailable after {self._MAX_ATTEMPTS} attempts: {last_error}")
