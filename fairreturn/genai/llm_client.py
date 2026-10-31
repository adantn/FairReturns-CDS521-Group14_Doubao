"""Minimal client for any OpenAI-compatible chat-completions endpoint, with a disk cache and a mock mode.

Configuration (environment variables, or a .env file in the project root):
    LLM_BASE_URL   e.g. https://api.openai.com/v1, a provider's OpenAI-compatible URL,
                   or http://localhost:11434/v1 for a local Ollama model
    LLM_API_KEY    the key (any non-empty string for Ollama)
    LLM_MODEL      model name from the provider's documentation

backend='auto' uses the API when all three are set and falls back to 'mock' otherwise. The mock backend
never calls the network: each task supplies its own deterministic mock function, so the full pipeline can
be tested offline. Mock outputs are labelled as such everywhere and must not be reported.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fairreturn import config as cfg

_ENV_LOADED = False


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ (existing variables win)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(path or cfg.ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api_configured() -> bool:
    load_dotenv()
    return all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


class LLMClient:
    def __init__(self, backend: str = "auto", cache_file: Path | None = None, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 600, timeout: int = 90, max_retries: int = 5):
        load_dotenv()
        if backend == "auto":
            backend = "api" if api_configured() else "mock"
        if backend not in ("api", "mock"):
            raise ValueError("backend must be 'api', 'mock' or 'auto'")
        self.backend = backend
        self.model = "mock" if backend == "mock" else (model or os.environ.get("LLM_MODEL", ""))
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if backend == "api" and not (self.base_url and self.api_key and self.model):
            raise RuntimeError("backend='api' needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (env vars or .env file)")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.timeout, self.max_retries = timeout, max_retries
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.cache_file and self.cache_file.exists():
            for line in self.cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec
                    except json.JSONDecodeError:
                        continue

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"

    def describe(self) -> dict:
        return {"backend": self.backend, "model": self.model,
                "base_url": self.base_url if self.backend == "api" else ""}

    def _key(self, system: str, user: str) -> str:
        raw = json.dumps([self.backend, self.model, self.temperature, system, user], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _call_api(self, system: str, user: str) -> str:
        import requests
        url = self.base_url + "/chat/completions"
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                return r.json()["choices"][0]["message"]["content"] or ""
            except RuntimeError:
                raise
            except Exception as e:  # network errors, malformed JSON
                last = repr(e)
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last}")

    def chat(self, system: str, user: str, mock_fn: Callable[[str, str], str] | None = None,
             meta: dict | None = None) -> str:
        key = self._key(system, user)
        with self._lock:
            if key in self._cache:
                return self._cache[key]["reply"]
        if self.is_mock:
            if mock_fn is None:
                raise RuntimeError("mock backend needs a mock_fn for this task")
            reply = mock_fn(system, user)
        else:
            reply = self._call_api(system, user)
        rec = {"key": key, "backend": self.backend, "model": self.model, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
               "system": system, "user": user, "reply": reply, "meta": meta or {}}
        with self._lock:
            self._cache[key] = rec
            if self.cache_file:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return reply

    def chat_many(self, requests_: list[dict], workers: int = 4, progress_every: int = 50) -> list[str]:
        """requests_: list of dicts with keys system, user, mock_fn (optional), meta (optional)."""
        results: list[str | None] = [None] * len(requests_)
        done = [0]

        def run(i):
            r = requests_[i]
            results[i] = self.chat(r["system"], r["user"], r.get("mock_fn"), r.get("meta"))
            done[0] += 1
            if progress_every and done[0] % progress_every == 0:
                print(f"    {done[0]}/{len(requests_)} LLM calls done")

        n_workers = 1 if self.is_mock else max(1, workers)
        if n_workers == 1:
            for i in range(len(requests_)):
                run(i)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(run, range(len(requests_))))
        return [r or "" for r in results]


def extract_json(text: str, prefer: str = "array"):
    """Parse the first JSON array/object in an LLM reply (tolerates code fences and extra text)."""
    if not text:
        raise ValueError("empty reply")
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    order = (("[", "]"), ("{", "}")) if prefer == "array" else (("{", "}"), ("[", "]"))
    for opener, closer in order:
        start = t.find(opener)
        end = t.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in reply: {text[:120]!r}")
