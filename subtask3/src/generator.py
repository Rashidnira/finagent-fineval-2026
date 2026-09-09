"""Cached generation front-end (provider-agnostic).

The submitted system runs on the local_qwen3_32b provider: Qwen3-32B served
locally through Ollama as 4-bit GGUF weights, reasoning disabled, with no
model API call at any stage (see src/providers.py). Hosted API providers were explored during development but are not part of
the submitted system.

The provider fingerprint (provider name, model id, experiment namespace,
sampling config) is hashed into every cache key, so no provider can reuse or
overwrite another provider's cached generations.

Canonical-output rule (§8): the first successful cached response for a unique
(provider, model, generation config, system prompt, user prompt, config tag)
combination is canonical. Cached entries are never silently regenerated;
an intentional regeneration must use a new config_tag (a new experiment).
"""
import datetime
import hashlib
import json
from pathlib import Path

from src.providers import MissingCredentialsError, get_provider  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "llm"


def cache_key(fingerprint: dict, system: str, user: str, config_tag: str) -> str:
    payload = json.dumps({"fp": fingerprint, "system": system, "user": user,
                          "config": config_tag},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Generator:
    def __init__(self, provider: str = "local_qwen3_32b", **provider_kw):
        self.provider = get_provider(provider, **provider_kw)
        CACHE.mkdir(parents=True, exist_ok=True)

    def generate(self, *, experiment_id: str, system: str, user: str,
                 prompt_version: str, task_id: str, question: str = None,
                 submission_id: str = None, config_tag: str = "v1") -> dict:
        """Return the canonical (cached-first) response record."""
        fp = self.provider.config_fingerprint()
        key = cache_key(fp, system, user, config_tag)
        path = CACHE / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["from_cache"] = True
            return rec

        out = self.provider.generate(system, user)
        rec = {
            "cache_key": key,
            "experiment_id": experiment_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provider_fingerprint": fp,
            "model": out["model"],
            "served_model": out.get("served_model"),
            "prompt_version": prompt_version,
            "config_tag": config_tag,
            "system_prompt": system,
            "user_prompt": user,
            "user_prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "user_prompt_chars": len(user),
            "task_id": task_id,
            "submission_id": submission_id,
            "question": question,
            "raw_response": out["text"],
            "stop_reason": out["stop_reason"],
            "usage": out["usage"],
            "from_cache": False,
        }
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return rec
