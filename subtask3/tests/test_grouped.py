"""Grouped-generation tests: prompt construction and answer parsing (no API)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.grouped_runner import build_grouped_prompt, parse_grouped, row_uid

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def easy_group():
    df = pd.read_parquet(ROOT / "data/raw/public-00000-of-00001.parquet")
    return df[df.task_id == "MSFT_20200429"]


@pytest.fixture()
def expert_group():
    df = pd.read_parquet(ROOT / "data/manifests/official_finnlp_test.parquet")
    return df[df.task_id == "MSFT_20200429"]


def test_build_easy_prompt(easy_group):
    sys_text, user_text, uids, pvs = build_grouped_prompt(easy_group, "C", "easy")
    assert len(uids) == 4 and len(set(uids)) == 4
    assert user_text.startswith("Context:")
    assert all(f'id="{u}"' in sys_text for u in uids)
    for q in easy_group.question:
        assert q in user_text
    # evidence appears once, not four times
    assert user_text.count("Financial Statements:") == 1


def test_build_expert_prompt_uses_official_ids(expert_group):
    _, user_text, uids, _ = build_grouped_prompt(expert_group, "C", "expert")
    assert all(u.startswith("poly_") for u in uids)
    assert set(uids) == set(expert_group.id)


def test_parse_roundtrip(expert_group):
    _, _, uids, _ = build_grouped_prompt(expert_group, "C", "expert")
    text = "\n".join(
        f'<ANSWER_{i} id="{u}">\nAnswer: a{i}.\nFinancial Statements '
        f"Evidence: None.\n</ANSWER_{i}>" for i, u in enumerate(uids, 1))
    out = parse_grouped(text, uids)
    assert list(out) == uids
    assert out[uids[2]].startswith("Answer: a3")


def test_parse_positional_fallback(expert_group):
    _, _, uids, _ = build_grouped_prompt(expert_group, "C", "expert")
    text = "\n".join(
        f'<ANSWER_{i} id="WRONG_{i}">body {i}</ANSWER_{i}>'
        for i in range(1, 5))
    out = parse_grouped(text, uids)
    assert out[uids[0]] == "body 1" and out[uids[3]] == "body 4"


def test_parse_missing_answer_raises(expert_group):
    _, _, uids, _ = build_grouped_prompt(expert_group, "C", "expert")
    text = f'<ANSWER_1 id="{uids[0]}">only one</ANSWER_1>'
    with pytest.raises(ValueError, match="unparsed"):
        parse_grouped(text, uids)


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from src.providers import GoogleGemmaProvider, MissingCredentialsError
    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY"):
        GoogleGemmaProvider()


def test_anthropic_provider_disabled():
    from src.providers import AnthropicProviderDisabled, ProviderDisabledError
    with pytest.raises(ProviderDisabledError, match="DISABLED"):
        AnthropicProviderDisabled()


# --- OpenRouter competition provider (all local; no network calls) ----------

def _openrouter_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    from src.providers import OpenRouterGemmaProvider
    p = OpenRouterGemmaProvider()
    p.MIN_CALL_SPACING_S = 0    # no pacing sleeps inside tests
    return p


class _FakeResp:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload


def test_openrouter_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from src.providers import MissingCredentialsError, OpenRouterGemmaProvider
    with pytest.raises(MissingCredentialsError, match="OPENROUTER_API_KEY"):
        OpenRouterGemmaProvider()


def test_submitted_provider_is_the_default():
    """The submitted system ran Qwen3-32B locally, so that must be the default.

    The OpenRouter and direct-Google providers are development candidates kept
    for provenance; making one of them the default would contradict the paper's
    eligibility statement (model-based track, no model API calls).
    """
    import inspect
    from src.generator import Generator
    sig = inspect.signature(Generator.__init__)
    assert sig.parameters["provider"].default == "local_qwen3_32b"


def test_openrouter_model_is_pinned(monkeypatch):
    from src.providers import (ModelUnavailableError,
                               OPENROUTER_COMPETITION_MODEL,
                               OpenRouterGemmaProvider)
    assert OPENROUTER_COMPETITION_MODEL == "google/gemma-4-31b-it:free"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    with pytest.raises(ModelUnavailableError, match="pinned"):
        OpenRouterGemmaProvider(model="openrouter/auto")


def test_openrouter_payload_disables_fallbacks(monkeypatch):
    p = _openrouter_provider(monkeypatch)
    payload = p.build_payload("sys text", "user text")
    assert payload["model"] == "google/gemma-4-31b-it:free"
    assert payload["provider"] == {"allow_fallbacks": False}
    assert "models" not in payload          # no fallback-model list
    assert "test-key-not-real" not in str(payload)  # key never in payload
    assert payload["messages"][0]["content"].startswith("sys text")


def test_openrouter_cache_namespace_distinct_from_google(monkeypatch):
    from src.generator import cache_key
    from src.providers import COMPETITION_MODEL, GEMMA_GEN_CONFIG
    p = _openrouter_provider(monkeypatch)
    fp_or = p.config_fingerprint()
    fp_google = {"provider": "google_gemma", "model": COMPETITION_MODEL,
                 **GEMMA_GEN_CONFIG}
    assert fp_or["provider"] == "openrouter_gemma"
    assert fp_or["experiment_namespace"] == "openrouter_gemma4_31b"
    assert cache_key(fp_or, "s", "u", "v1") != cache_key(fp_google, "s", "u", "v1")


def test_openrouter_rejects_fallback_served_model(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _openrouter_provider(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp(200, {
        "model": "some/other-model",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}))
    with pytest.raises(ModelUnavailableError, match="not the pinned"):
        p.generate("s", "u")


def test_openrouter_429_stops_after_exactly_one_attempt(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _openrouter_provider(monkeypatch)
    calls = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: calls.append(1) or _FakeResp(429))
    with pytest.raises(ModelUnavailableError, match="429"):
        p.generate("s", "u")
    assert len(calls) == 1      # NO retry loop: one HTTP attempt, then STOP


def test_openrouter_5xx_still_retried_transiently(monkeypatch):
    import time

    import requests
    p = _openrouter_provider(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    responses = [_FakeResp(503), _FakeResp(200, {
        "model": "google/gemma-4-31b-it",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {}})]
    monkeypatch.setattr(requests, "post", lambda *a, **kw: responses.pop(0))
    out = p.generate("s", "u")
    assert out["text"] == "ok"          # transient 5xx retried, same model
    assert not responses                # both canned responses consumed


def test_openrouter_unrecoverable_error_stops_no_fallback(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _openrouter_provider(monkeypatch)
    calls = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: calls.append(1) or _FakeResp(402))
    with pytest.raises(ModelUnavailableError, match="STOP"):
        p.generate("s", "u")
    assert len(calls) == 1      # 402 is terminal: no retry, no model switch


def test_openrouter_success_roundtrip(monkeypatch):
    import requests
    p = _openrouter_provider(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp(200, {
        "model": "google/gemma-4-31b-it",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}}))
    out = p.generate("s", "u")
    assert out["text"] == "hello"
    assert out["model"] == "google/gemma-4-31b-it:free"
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 2}


# --- OpenRouter Gemma-4-26B-A4B candidate (all local; no network calls) -----

def _gemma26b_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    from src.providers import OpenRouterGemma26bProvider
    p = OpenRouterGemma26bProvider()
    p.MIN_CALL_SPACING_S = 0
    return p


def test_gemma26b_pinned_payload_and_registry(monkeypatch):
    from src.providers import (ModelUnavailableError, OpenRouterGemma26bProvider,
                               PROVIDERS)
    assert PROVIDERS["openrouter_gemma4_26b_a4b"] is OpenRouterGemma26bProvider
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    with pytest.raises(ModelUnavailableError, match="pinned"):
        OpenRouterGemma26bProvider(model="openrouter/auto")
    p = _gemma26b_provider(monkeypatch)
    payload = p.build_payload("sys text", "user text")
    assert payload["model"] == "google/gemma-4-26b-a4b-it:free"
    assert payload["provider"] == {"allow_fallbacks": False}
    assert "models" not in payload
    assert "test-key-not-real" not in str(payload)


def test_gemma26b_cache_namespace_separate(monkeypatch):
    from src.generator import cache_key
    from src.providers import COMPETITION_MODEL, GEMMA_GEN_CONFIG
    fps = [_gemma26b_provider(monkeypatch).config_fingerprint(),
           _openrouter_provider(monkeypatch).config_fingerprint(),
           _qwen_provider(monkeypatch).config_fingerprint(),
           {"provider": "google_gemma", "model": COMPETITION_MODEL,
            **GEMMA_GEN_CONFIG}]
    assert fps[0]["experiment_namespace"] == "openrouter_gemma4_26b_a4b"
    keys = {cache_key(fp, "s", "u", "v1") for fp in fps}
    assert len(keys) == 4                   # pairwise-distinct cache spaces


def test_gemma26b_rejects_other_model_and_429_stops_once(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _gemma26b_provider(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp(200, {
        "model": "google/gemma-4-31b-it",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}))
    with pytest.raises(ModelUnavailableError, match="not the pinned"):
        p.generate("s", "u")
    calls = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: calls.append(1) or _FakeResp(429))
    with pytest.raises(ModelUnavailableError, match="429"):
        p.generate("s", "u")
    assert len(calls) == 1


# --- OpenRouter Qwen3-32B candidate (all local; no network calls) -----------

def _qwen_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    from src.providers import OpenRouterQwen3Provider
    p = OpenRouterQwen3Provider()
    p.MIN_CALL_SPACING_S = 0
    return p


def test_qwen_model_is_pinned(monkeypatch):
    from src.providers import (ModelUnavailableError, OPENROUTER_QWEN_MODEL,
                               OpenRouterQwen3Provider)
    assert OPENROUTER_QWEN_MODEL == "qwen/qwen3-32b:free"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    with pytest.raises(ModelUnavailableError, match="pinned"):
        OpenRouterQwen3Provider(model="qwen/qwen3-235b-a22b:free")
    with pytest.raises(ModelUnavailableError, match="pinned"):
        OpenRouterQwen3Provider(model="openrouter/auto")


def test_qwen_payload_disables_fallbacks_and_reasoning(monkeypatch):
    p = _qwen_provider(monkeypatch)
    payload = p.build_payload("sys text", "user text")
    assert payload["model"] == "qwen/qwen3-32b:free"
    assert payload["provider"] == {"allow_fallbacks": False}
    assert payload["reasoning"] == {"enabled": False}
    assert "models" not in payload          # no fallback-model list
    assert "test-key-not-real" not in str(payload)


def test_qwen_cache_namespace_separate_from_gemma_and_google(monkeypatch):
    from src.generator import cache_key
    from src.providers import COMPETITION_MODEL, GEMMA_GEN_CONFIG
    q = _qwen_provider(monkeypatch)
    g = _openrouter_provider(monkeypatch)
    fp_q, fp_g = q.config_fingerprint(), g.config_fingerprint()
    fp_google = {"provider": "google_gemma", "model": COMPETITION_MODEL,
                 **GEMMA_GEN_CONFIG}
    assert fp_q["provider"] == "openrouter_qwen3_32b"
    assert fp_q["experiment_namespace"] == "openrouter_qwen3_32b"
    keys = {cache_key(fp, "s", "u", "v1") for fp in (fp_q, fp_g, fp_google)}
    assert len(keys) == 3                   # pairwise-distinct cache spaces


def test_qwen_rejects_fallback_served_model(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _qwen_provider(monkeypatch)
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp(200, {
        "model": "qwen/qwen2.5-72b-instruct",
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}))
    with pytest.raises(ModelUnavailableError, match="not the pinned"):
        p.generate("s", "u")


def test_qwen_429_stops_after_exactly_one_attempt(monkeypatch):
    import requests
    from src.providers import ModelUnavailableError
    p = _qwen_provider(monkeypatch)
    calls = []
    monkeypatch.setattr(requests, "post",
                        lambda *a, **kw: calls.append(1) or _FakeResp(429))
    with pytest.raises(ModelUnavailableError, match="429"):
        p.generate("s", "u")
    assert len(calls) == 1


def test_qwen_registered_and_gemma_unchanged():
    from src.providers import (OpenRouterGemmaProvider, OpenRouterQwen3Provider,
                               PROVIDERS)
    assert PROVIDERS["openrouter_qwen3_32b"] is OpenRouterQwen3Provider
    assert PROVIDERS["openrouter_gemma"] is OpenRouterGemmaProvider
    assert OpenRouterGemmaProvider.pinned_model == "google/gemma-4-31b-it:free"
    assert OpenRouterGemmaProvider.experiment_namespace == "openrouter_gemma4_31b"
    # Gemma sampling config untouched by the Qwen addition
    assert OpenRouterGemmaProvider.default_gen_config["temperature"] == 1.0
    assert OpenRouterGemmaProvider.extra_payload == {}


# --- Local self-hosted Qwen3-32B-AWQ candidate (vLLM; no network calls) -----

def test_local_qwen_pinned_no_key_needed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from src.providers import LocalQwen3Provider, ModelUnavailableError
    p = LocalQwen3Provider()            # constructs with no credentials at all
    assert p.model == "qwen3:32b"
    with pytest.raises(ModelUnavailableError, match="pinned"):
        LocalQwen3Provider(model="qwen3:235b")
    payload = p.build_payload("s", "u")
    assert payload["think"] is False            # no chain-of-thought
    assert payload["options"]["num_ctx"] == 32768
    assert payload["options"]["temperature"] == 0.7
    assert "Authorization" not in p._headers()


def test_local_qwen_base_url_and_namespace(monkeypatch):
    from src.generator import cache_key
    from src.providers import LocalQwen3Provider
    monkeypatch.delenv("LOCAL_OLLAMA_BASE", raising=False)
    p = LocalQwen3Provider()
    assert p.api_base == "http://127.0.0.1:11500"
    monkeypatch.setenv("LOCAL_OLLAMA_BASE", "http://127.0.0.1:9999")
    assert p.api_base == "http://127.0.0.1:9999"
    fp = p.config_fingerprint()
    assert fp["experiment_namespace"] == "local_qwen3_32b"
    assert fp["quantization"] == "gguf-q4_k_m"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    fp_or_qwen = _qwen_provider(monkeypatch).config_fingerprint()
    assert cache_key(fp, "s", "u", "v1") != cache_key(fp_or_qwen, "s", "u", "v1")


def test_local_qwen_rejects_other_served_model(monkeypatch):
    import requests
    from src.providers import LocalQwen3Provider, ModelUnavailableError
    p = LocalQwen3Provider()
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp(200, {
        "model": "some-other-model",
        "message": {"content": "x"}, "done_reason": "stop"}))
    with pytest.raises(ModelUnavailableError, match="not the pinned"):
        p.generate("s", "u")


def test_local_qwen_native_api_roundtrip(monkeypatch):
    import requests
    from src.providers import LocalQwen3Provider
    p = LocalQwen3Provider()
    captured = {}
    def fake_post(url, **kw):
        captured["url"] = url
        return _FakeResp(200, {"model": "qwen3:32b",
                               "message": {"content": "hello"},
                               "done_reason": "stop",
                               "prompt_eval_count": 12, "eval_count": 3})
    monkeypatch.setattr(requests, "post", fake_post)
    out = p.generate("s", "u")
    assert captured["url"].endswith("/api/chat")    # native API, not /v1
    assert out["text"] == "hello"
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 3}


def test_local_gemma_pinned_payload_and_namespace(monkeypatch):
    from src.generator import cache_key
    from src.providers import (LocalGemmaProvider, LocalQwen3Provider,
                               ModelUnavailableError, PROVIDERS)
    assert PROVIDERS["local_gemma4_31b"] is LocalGemmaProvider
    p = LocalGemmaProvider()
    assert p.model == "hf.co/unsloth/gemma-4-31B-it-GGUF:Q4_K_M"
    with pytest.raises(ModelUnavailableError, match="pinned"):
        LocalGemmaProvider(model="gemma3:12b")
    payload = p.build_payload("s", "u")
    assert payload["think"] is False    # Gemma-4 thinks on Ollama>=0.32: disable
    assert payload["options"]["temperature"] == 1.0   # official Gemma sampling
    assert payload["options"]["top_k"] == 64
    fp = p.config_fingerprint()
    assert fp["experiment_namespace"] == "local_gemma4_31b"
    fp_q = LocalQwen3Provider().config_fingerprint()
    assert cache_key(fp, "s", "u", "v1") != cache_key(fp_q, "s", "u", "v1")


def test_nemotron_pinned_eligible_and_namespaced(monkeypatch):
    from src.generator import cache_key
    from src.providers import (LocalNemotronProvider, LocalQwen3Provider,
                               ModelUnavailableError, PROVIDERS)
    assert PROVIDERS["local_nemotron_49b"] is LocalNemotronProvider
    p = LocalNemotronProvider()
    assert "49B" in p.model                 # 49B total — <=70B, published count
    with pytest.raises(ModelUnavailableError, match="pinned"):
        LocalNemotronProvider(model="llama3.3:70b")   # 70.6B — refused
    payload = p.build_payload("s", "u")
    assert "think" not in payload
    assert payload["options"]["temperature"] == 0.0   # greedy per model card
    fp = p.config_fingerprint()
    assert fp["experiment_namespace"] == "local_nemotron_49b"
    fp_q = LocalQwen3Provider().config_fingerprint()
    assert cache_key(fp, "s", "u", "v1") != cache_key(fp_q, "s", "u", "v1")


def test_grouped_runner_experiment_namespace():
    from src.providers import EXPERIMENT_NAMESPACE
    exp_id = f"easy_C_{EXPERIMENT_NAMESPACE}_grouped_v1"
    assert exp_id == "easy_C_openrouter_gemma4_31b_grouped_v1"
    assert exp_id != "easy_C_gemma4_31b_grouped_v1"   # legacy id preserved
