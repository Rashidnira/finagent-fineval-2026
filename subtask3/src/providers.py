"""Generation providers.

Competition inference candidates (OpenRouter only, https://openrouter.ai/api/v1):
- openrouter_gemma:           google/gemma-4-31b-it:free      (ns openrouter_gemma4_31b)
- openrouter_gemma4_26b_a4b:  google/gemma-4-26b-a4b-it:free  (ns openrouter_gemma4_26b_a4b)
- openrouter_qwen3_32b:       qwen/qwen3-32b:free             (ns openrouter_qwen3_32b;
  :free variant not listed on OpenRouter as of 2026-08-19 — 404s)
Each candidate is pinned to its exact free model with fallbacks disabled and
a fully separate cache/experiment namespace. If the pinned model is
unavailable, 429-rate-limited (one attempt, then STOP), or errors
unrecoverably, generation STOPS; no other model and no paid routing is ever
used.

Legacy paths, retained for provenance only:
- google_gemma: the previous direct Gemini-API implementation
  (LEGACY_EXPLORATORY — not for competition inference; its prior cached
  results are preserved untouched).
- anthropic: DISABLED_FOR_COMPETITION_INFERENCE.
"""
import os
import re
import time

# --- OpenRouter competition configuration -----------------------------------
OPENROUTER_COMPETITION_MODEL = "google/gemma-4-31b-it:free"  # 30.7B dense — eligible
OPENROUTER_QWEN_MODEL = "qwen/qwen3-32b:free"                # 32.8B dense — eligible
                                                             # (NOT LISTED on
                                                             # OpenRouter 2026-08-19:
                                                             # 404, paid-only variant)
OPENROUTER_GEMMA_26B_MODEL = "google/gemma-4-26b-a4b-it:free"  # 25.2B MoE — eligible
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Distinct experiment/cache namespaces: each appears in experiment_ids and in
# the provider fingerprint hashed into every cache key, so no OpenRouter
# candidate can ever reuse (or overwrite) another candidate's cache/results
# or the legacy direct-Google ones.
EXPERIMENT_NAMESPACE = "openrouter_gemma4_31b"
QWEN_EXPERIMENT_NAMESPACE = "openrouter_qwen3_32b"
GEMMA_26B_EXPERIMENT_NAMESPACE = "openrouter_gemma4_26b_a4b"

# --- Legacy direct-Google constants (LEGACY_EXPLORATORY only) ---------------
COMPETITION_MODEL = "gemma-4-31b-it"          # 30.7B dense — verified eligible
FALLBACK_MODEL = "gemma-4-26b-a4b-it"         # unused; kept for provenance

# Official recommended Gemma sampling configuration. LEGACY_EXPLORATORY: this
# path was evaluated during development and is not used by the submitted
# system, which runs local_qwen3_32b. No thinking/reasoning mode.
GEMMA_GEN_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 4096,   # 4 answers ≤100 words + delimiters, with headroom
}


class MissingCredentialsError(RuntimeError):
    pass


class ProviderDisabledError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    """The pinned free model cannot serve this request. Callers must STOP:
    switching models (free or paid) is prohibited by competition policy."""


class OpenRouterPinnedProvider:
    """Base for pinned free-model OpenRouter candidates.

    Policy (enforced here, not just documented; identical for every subclass):
    - Model is PINNED. No auto-routing, no fallback models, no paid routing:
      the request carries provider.allow_fallbacks=false, and any response
      served by a different model raises ModelUnavailableError.
    - HTTP 429 STOPS IMMEDIATELY after exactly one attempt (no retry loop):
      free-tier accounts have an account-wide daily allowance for :free
      models and rejected attempts can count against it, so automatic 429
      retries burn quota for nothing. Only transient network failures and
      5xx upstream errors are retried — always on the same pinned model.
    - If the model is unavailable or returns an unrecoverable error (429,
      other 4xx incl. 402 payment-required), we raise ModelUnavailableError
      and STOP — never switch models.
    - The API key is read from OPENROUTER_API_KEY, sent only in the
      Authorization header, and never logged or echoed.

    Free-tier limits (observed 2026-08-19): ~20 requests/min for :free
    models; calls are paced >=6s apart (<=10/min).

    The system text is prepended to the user content (single user message)
    for parity with the legacy google_gemma provider: inline prepending is
    deterministic and endpoint-independent, and keeps prompts byte-identical
    across candidates.

    Subclasses set: name, pinned_model, experiment_namespace,
    default_gen_config, and optionally extra_payload.
    """

    name = None
    pinned_model = None
    experiment_namespace = None
    default_gen_config = None
    extra_payload = {}                  # provider-specific API params
    api_base = OPENROUTER_BASE          # OpenAI-compatible chat-completions base
    MIN_CALL_SPACING_S = 6
    MAX_ATTEMPTS = 8
    grouped_admission_limited = False   # no Gemini-style 16k/min input bucket

    def __init__(self, model: str = None, **gen_overrides):
        """gen_overrides adds or replaces sampling keys (e.g. seed=1, num_ctx=40960).

        Overrides are merged into gen_config, which config_fingerprint() spreads
        into the cache key, so an overridden run can never collide with, reuse
        or overwrite an un-overridden one. Passing no overrides leaves the
        fingerprint byte-identical to previous runs, so existing caches stay
        valid.
        """
        model = model or self.pinned_model
        if model != self.pinned_model:
            raise ModelUnavailableError(
                f"Only the pinned competition model "
                f"{self.pinned_model!r} is permitted; got {model!r}.")
        self._check_credentials()
        self.model = model
        self.gen_config = dict(self.default_gen_config)
        self.gen_config.update(gen_overrides)
        self._last_call = 0.0

    def _check_credentials(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise MissingCredentialsError(
                "OPENROUTER_API_KEY is required for the pinned free model "
                f"{self.pinned_model}.\n"
                "No paid API will be used. Set OPENROUTER_API_KEY and rerun.")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "finnlp2026-polyfiqa",
        }

    def config_fingerprint(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "experiment_namespace": self.experiment_namespace,
                **self.gen_config}

    def build_payload(self, system: str, user: str) -> dict:
        contents = f"{system.strip()}\n\n{user}" if system else user
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": contents}],
            "temperature": self.gen_config["temperature"],
            "top_p": self.gen_config["top_p"],
            "top_k": self.gen_config["top_k"],
            "max_tokens": self.gen_config["max_output_tokens"],
            # hard-disable OpenRouter fallback routing (free AND paid)
            "provider": {"allow_fallbacks": False},
            **self.extra_payload,
        }

    def _pace(self):
        wait = self.MIN_CALL_SPACING_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def generate(self, system: str, user: str) -> dict:
        import requests
        payload = self.build_payload(system, user)
        headers = self._headers()
        data = None
        for attempt in range(self.MAX_ATTEMPTS):
            self._pace()
            self._last_call = time.time()
            try:
                resp = requests.post(f"{self.api_base}/chat/completions",
                                     json=payload, headers=headers, timeout=300)
            except requests.RequestException as e:
                if attempt < self.MAX_ATTEMPTS - 1:
                    print(f"    [network] {type(e).__name__}; retrying "
                          f"(attempt {attempt + 1}/{self.MAX_ATTEMPTS})")
                    time.sleep(15 * (attempt + 1))
                    continue
                raise ModelUnavailableError(
                    "STOP: network failure reaching OpenRouter after "
                    f"{self.MAX_ATTEMPTS} attempts; no fallback permitted.")
            if resp.status_code == 429:
                # NO retry loop on 429: rejected attempts can count against
                # the account-wide daily :free allowance, so retrying burns
                # quota for nothing. STOP immediately; rerun later.
                # Surface the server's own reason (daily cap vs upstream
                # congestion) — it never contains the API key.
                try:
                    detail = resp.json().get("error", {}).get("message", "")
                except Exception:
                    detail = ""
                raise ModelUnavailableError(
                    f"STOP: {self.model} returned HTTP 429 (rate-limited). "
                    "429s are never auto-retried (they can consume the "
                    "account-wide free allowance). No paid or alternate "
                    f"model fallback is permitted; rerun later. "
                    f"Server detail: {detail[:300]!r}")
            if resp.status_code in (500, 502, 503, 504):
                if attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise ModelUnavailableError(
                    f"STOP: OpenRouter upstream error {resp.status_code} "
                    f"persisted for {self.MAX_ATTEMPTS} attempts.")
            if resp.status_code != 200:
                # 400/401/402/403/404: unrecoverable for this pinned model
                raise ModelUnavailableError(
                    f"STOP: unrecoverable HTTP {resp.status_code} for pinned "
                    f"model {self.model}. No fallback model is permitted.")
            data = resp.json()
            if "error" in data and not data.get("choices"):
                raise ModelUnavailableError(
                    f"STOP: OpenRouter error for pinned model {self.model}: "
                    f"{data['error'].get('message', 'unknown')!s:.300}")
            break

        served = data.get("model") or ""
        if served.split(":")[0] != self.model.split(":")[0]:
            raise ModelUnavailableError(
                f"STOP: response served by {served!r}, not the pinned "
                f"{self.model!r}; fallback/auto-routing output is rejected.")
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return {
            "text": (choice.get("message") or {}).get("content") or "",
            "model": self.model,
            "served_model": served,
            "stop_reason": choice.get("finish_reason"),
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
        }

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError(
            "OpenRouter has no count_tokens endpoint; use the legacy "
            "google_gemma provider for --count-tokens-only audits.")


class OpenRouterGemmaProvider(OpenRouterPinnedProvider):
    """Competition candidate: google/gemma-4-31b-it:free (30.7B dense)."""

    name = "openrouter_gemma"
    pinned_model = OPENROUTER_COMPETITION_MODEL
    experiment_namespace = EXPERIMENT_NAMESPACE
    default_gen_config = GEMMA_GEN_CONFIG


# Official Qwen3 recommended non-thinking sampling (model card); prompts are
# NOT tuned per model — only sampling and the reasoning switch differ.
QWEN_GEN_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_output_tokens": 4096,
}


class OpenRouterGemma26bProvider(OpenRouterPinnedProvider):
    """Competition candidate: google/gemma-4-26b-a4b-it:free (25.2B total MoE
    — eligible). Same official Gemma sampling config and identical prompts as
    the 31B candidate; verified listed on OpenRouter 2026-08-19 ($0/$0,
    262,144-token context)."""

    name = "openrouter_gemma4_26b_a4b"
    pinned_model = OPENROUTER_GEMMA_26B_MODEL
    experiment_namespace = GEMMA_26B_EXPERIMENT_NAMESPACE
    default_gen_config = GEMMA_GEN_CONFIG


class OpenRouterQwen3Provider(OpenRouterPinnedProvider):
    """Competition candidate: qwen/qwen3-32b:free (32.8B dense — eligible).

    Reasoning/thinking mode is disabled via the OpenRouter API parameter
    (an API config, not a prompt change): Qwen3 defaults to long
    chain-of-thought, which would eat the 4096-token output budget and break
    the fixed answer format shared with the Gemma candidate.
    """

    name = "openrouter_qwen3_32b"
    pinned_model = OPENROUTER_QWEN_MODEL
    experiment_namespace = QWEN_EXPERIMENT_NAMESPACE
    default_gen_config = QWEN_GEN_CONFIG
    extra_payload = {"reasoning": {"enabled": False}}


class LocalQwen3Provider(OpenRouterPinnedProvider):
    """Competition candidate: Qwen3-32B served by OUR OWN Ollama instance
    (self-hosted on the UNT midas server, H100 GPU 1; reached through an SSH
    tunnel). Model-based route: no third-party inference API involved.

    - Pinned model "qwen3:32b" (32.8B dense, Apache-2.0, eligible; GGUF
      Q4_K_M quantization — disclose in the system paper. Ollama was chosen
      over vLLM because the box's NVIDIA driver caps CUDA at 12.2, older
      than current vLLM wheels support).
    - Endpoint from LOCAL_OLLAMA_BASE (default http://127.0.0.1:11500);
      talks to Ollama's NATIVE /api/chat (not the OpenAI shim) because only
      the native API accepts think:false — Qwen3's thinking mode is disabled
      as an API/serving parameter, not a prompt change.
    - No credentials exist anywhere on this path; same pinning/served-model
      checks as every candidate; no pacing needed (dedicated local server).
    """

    name = "local_qwen3_32b"
    pinned_model = "qwen3:32b"
    experiment_namespace = "local_qwen3_32b"
    default_gen_config = QWEN_GEN_CONFIG
    MIN_CALL_SPACING_S = 0

    @property
    def api_base(self):
        return os.environ.get("LOCAL_OLLAMA_BASE", "http://127.0.0.1:11500")

    def _check_credentials(self):
        pass    # local server: no credentials exist, none can leak

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def config_fingerprint(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "experiment_namespace": self.experiment_namespace,
                "quantization": "gguf-q4_k_m", "think": False,
                **self.gen_config}

    def build_payload(self, system: str, user: str) -> dict:
        contents = f"{system.strip()}\n\n{user}" if system else user
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": contents}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.gen_config["temperature"],
                "top_p": self.gen_config["top_p"],
                "top_k": self.gen_config["top_k"],
                "num_predict": self.gen_config["max_output_tokens"],
                # Overridable; the default keeps the fingerprint of earlier runs.
                "num_ctx": self.gen_config.get("num_ctx", 32768),
            },
        }
        # Only present when explicitly requested, so unseeded runs keep their
        # original cache keys.
        if self.gen_config.get("seed") is not None:
            payload["options"]["seed"] = self.gen_config["seed"]
        return payload

    def generate(self, system: str, user: str) -> dict:
        import requests
        payload = self.build_payload(system, user)
        data = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                resp = requests.post(f"{self.api_base}/api/chat", json=payload,
                                     headers=self._headers(), timeout=900)
            except requests.RequestException as e:
                if attempt < self.MAX_ATTEMPTS - 1:
                    print(f"    [network] {type(e).__name__}; retrying "
                          f"(attempt {attempt + 1}/{self.MAX_ATTEMPTS})")
                    time.sleep(10 * (attempt + 1))
                    continue
                raise ModelUnavailableError(
                    "STOP: cannot reach the local Ollama server after "
                    f"{self.MAX_ATTEMPTS} attempts; is the SSH tunnel up?")
            if resp.status_code in (500, 502, 503, 504) \
                    and attempt < self.MAX_ATTEMPTS - 1:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise ModelUnavailableError(
                    f"STOP: local Ollama returned HTTP {resp.status_code} "
                    f"for pinned model {self.model}: {resp.text[:300]!r}")
            data = resp.json()
            break

        served = data.get("model") or ""
        if served != self.model:
            raise ModelUnavailableError(
                f"STOP: response served by {served!r}, not the pinned "
                f"{self.model!r}; output rejected.")
        return {
            "text": (data.get("message") or {}).get("content") or "",
            "model": self.model,
            "served_model": served,
            "stop_reason": data.get("done_reason"),
            "usage": {
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
            },
        }


class LocalGemmaProvider(LocalQwen3Provider):
    """Competition candidate: Gemma-4-31B-it served by OUR OWN Ollama
    instance (ungated Unsloth GGUF Q4_K_M conversion — the same 30.7B model
    as the validated benchmarks; quantization disclosed in the paper).
    Same self-hosted endpoint and policy as the local Qwen candidate."""

    name = "local_gemma4_31b"
    pinned_model = "hf.co/unsloth/gemma-4-31B-it-GGUF:Q4_K_M"
    experiment_namespace = "local_gemma4_31b"
    default_gen_config = GEMMA_GEN_CONFIG

    def config_fingerprint(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "experiment_namespace": self.experiment_namespace,
                "quantization": "gguf-q4_k_m", "think": False,
                **self.gen_config}
    # payload inherited from LocalQwen3Provider, INCLUDING think:false —
    # Gemma-4 is a thinking model on Ollama >=0.32 (verified 2026-08-20:
    # without think:false its output lands in message.thinking, content empty)


# NVIDIA model-card recommendation for non-reasoning use: greedy decoding.
NEMOTRON_GEN_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 40,
    "max_output_tokens": 4096,
}


class LocalNemotronProvider(LocalQwen3Provider):
    """Competition candidate: Llama-3.3-Nemotron-Super-49B-v1 (NVIDIA;
    49B total parameters — unambiguously <=70B, published count) served by
    our own Ollama. bartowski GGUF Q4_K_M — a straight quantization of the
    official checkpoint (disclose quantization in the paper). Reasoning mode
    is prompt-toggled in this model and we never enable it; the Ollama
    think parameter does not apply (removed from the payload)."""

    name = "local_nemotron_49b"
    pinned_model = ("hf.co/bartowski/"
                    "nvidia_Llama-3_3-Nemotron-Super-49B-v1-GGUF:Q4_K_M")
    experiment_namespace = "local_nemotron_49b"
    default_gen_config = NEMOTRON_GEN_CONFIG

    def config_fingerprint(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "experiment_namespace": self.experiment_namespace,
                "quantization": "gguf-q4_k_m", **self.gen_config}

    def build_payload(self, system: str, user: str) -> dict:
        payload = super().build_payload(system, user)
        payload.pop("think", None)
        return payload


class GoogleGemmaProvider:
    """LEGACY_EXPLORATORY — NOT for competition inference.

    Previous direct Gemini-API implementation, preserved (with its cached
    results under cache/llm and prior experiment outputs) for provenance and
    exploratory comparison only. Competition inference uses openrouter_gemma.

    Zero-cost Gemma via the Gemini Developer API (google-genai SDK).

    Free-tier constraint (observed 2026-08-19): input-token quota for
    gemma-4-31b is 16,000 tokens/minute (GenerateContentInputTokensPerModel
    PerMinute-FreeTier). Our requests are ~15.3-16.0k tokens, so exactly one
    generation fits per minute: calls are proactively paced >=62s apart and
    429s honor the server's retryDelay before retrying.

    Note: the system text is prepended to the user content instead of using
    `system_instruction`, because Gemma endpoints on the Gemini API have
    historically rejected developer instructions; inline prepending is
    deterministic and endpoint-independent.
    """

    name = "google_gemma"
    MIN_CALL_SPACING_S = 62
    MAX_ATTEMPTS = 8
    grouped_admission_limited = True    # Gemini free-tier 16k/min input bucket

    def __init__(self, model: str = COMPETITION_MODEL):
        if not os.environ.get("GEMINI_API_KEY"):
            raise MissingCredentialsError(
                "GEMINI_API_KEY is required for the free Gemma 4 API endpoint.\n"
                "No paid API will be used.\n"
                "Set the free Google AI Studio key as GEMINI_API_KEY and rerun "
                "the specified command."
            )
        from google import genai
        from google.genai import types
        self.model = model
        # Disable SDK-internal 429 retries: rejected attempts appear to count
        # against the free-tier input-token bucket, so rapid internal retries
        # self-exhaust the quota. Our wrapper owns all pacing.
        self.client = genai.Client(http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1)))
        self.gen_config = dict(GEMMA_GEN_CONFIG)
        self._last_call = 0.0

    def config_fingerprint(self) -> dict:
        return {"provider": self.name, "model": self.model, **self.gen_config}

    @staticmethod
    def _retry_delay_s(exc) -> float:
        m = re.search(r"retry in ([\d.]+)s", str(exc))
        return float(m.group(1)) if m else 70.0

    def _pace(self):
        wait = self.MIN_CALL_SPACING_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def generate(self, system: str, user: str) -> dict:
        from google.genai import errors, types
        contents = f"{system.strip()}\n\n{user}" if system else user
        resp = None
        for attempt in range(self.MAX_ATTEMPTS):
            self._pace()
            try:
                self._last_call = time.time()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**self.gen_config),
                )
                break
            except errors.APIError as e:
                if e.code == 429 and attempt < self.MAX_ATTEMPTS - 1:
                    # sleep past a full bucket window, never less than 75s:
                    # rejected attempts seem to count against the bucket, so
                    # sub-minute sleeps re-collide forever
                    delay = min(max(self._retry_delay_s(e) + 10, 75), 150)
                    print(f"    [rate-limit] 429; sleeping {delay:.0f}s "
                          f"(attempt {attempt + 1}/{self.MAX_ATTEMPTS})")
                    time.sleep(delay)
                elif e.code in (500, 502, 503, 504) and attempt < self.MAX_ATTEMPTS - 1:
                    time.sleep(15 * (attempt + 1))
                else:
                    raise
        text = resp.text or ""
        um = resp.usage_metadata
        return {
            "text": text,
            "model": self.model,
            "stop_reason": str(
                resp.candidates[0].finish_reason) if resp.candidates else None,
            "usage": {
                "input_tokens": getattr(um, "prompt_token_count", None),
                "output_tokens": getattr(um, "candidates_token_count", None),
            },
        }

    def count_tokens(self, text: str) -> int:
        resp = self.client.models.count_tokens(model=self.model, contents=text)
        return resp.total_tokens


class AnthropicProviderDisabled:
    """DISABLED_FOR_COMPETITION_INFERENCE.

    Historical scaffolding (see git history: src/generator.py pre-migration).
    claude-sonnet-5 has an undisclosed parameter count and is ineligible under
    the 70B rule; it must never generate competition answers.
    """

    name = "anthropic_DISABLED"

    def __init__(self, *a, **kw):
        raise ProviderDisabledError(
            "The Anthropic provider is DISABLED_FOR_COMPETITION_INFERENCE "
            "(undisclosed parameter count; 70B rule). Use provider "
            "'google_gemma'.")


PROVIDERS = {
    "openrouter_gemma": OpenRouterGemmaProvider,       # COMPETITION candidate
    "openrouter_gemma4_26b_a4b": OpenRouterGemma26bProvider,  # COMPETITION candidate
    "openrouter_qwen3_32b": OpenRouterQwen3Provider,   # candidate (:free not listed)
    "local_qwen3_32b": LocalQwen3Provider,             # COMPETITION candidate (self-hosted)
    "local_gemma4_31b": LocalGemmaProvider,            # COMPETITION candidate (self-hosted)
    "local_nemotron_49b": LocalNemotronProvider,       # COMPETITION candidate (self-hosted)
    "google_gemma": GoogleGemmaProvider,               # LEGACY_EXPLORATORY only
    "anthropic": AnthropicProviderDisabled,            # DISABLED
}


def get_provider(name: str, **kw):
    return PROVIDERS[name](**kw)
