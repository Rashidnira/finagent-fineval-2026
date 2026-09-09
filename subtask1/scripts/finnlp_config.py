"""Central runtime config for the FinNLP-2026 shared-task pipeline.

Nothing in this repo should hardcode a host, port or model id. Everything reads
from here, and everything here is overridable by environment variable, so the
same code reproduces on a reviewer's machine with a different serving setup.

    export FINNLP_BASE_URL=http://my-host:8000/v1
    export FINNLP_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
"""
import os

# OpenAI-compatible endpoint. Default matches ./serve_vllm.sh on localhost.
BASE_URL = os.environ.get("FINNLP_BASE_URL", "http://127.0.0.1:8770/v1")
API_KEY = os.environ.get("FINNLP_API_KEY", "EMPTY")

# Must stay <= 70B total parameters to satisfy the shared-task rules.
MODEL = os.environ.get("FINNLP_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
MODEL_TOTAL_PARAMS_B = float(os.environ.get("FINNLP_MODEL_PARAMS_B", "32"))
PARAM_LIMIT_B = 70.0

MAX_MODEL_LEN = int(os.environ.get("FINNLP_MAX_MODEL_LEN", "4096"))
CONCURRENCY = int(os.environ.get("FINNLP_CONCURRENCY", "12"))
REQUEST_TIMEOUT = float(os.environ.get("FINNLP_TIMEOUT", "600"))

ROOT = os.environ.get("FINNLP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "hf")
OUT = os.path.join(ROOT, "out")

S1_DIR = os.path.join(DATA, "finnlp2026-subtask1-greek-ner")
S2_DIR = os.path.join(DATA, "finnlp2026-subtask2-japanese-icr")


def assert_rules_ok():
    """Fail loudly rather than submit an ineligible system."""
    if MODEL_TOTAL_PARAMS_B > PARAM_LIMIT_B:
        raise SystemExit(
            f"RULE VIOLATION: {MODEL} has {MODEL_TOTAL_PARAMS_B}B total params, "
            f"limit is {PARAM_LIMIT_B}B (MoE counts TOTAL, not active)."
        )


def client():
    """An OpenAI SDK client pointed at the configured endpoint."""
    from openai import OpenAI
    assert_rules_ok()
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT)


if __name__ == "__main__":
    assert_rules_ok()
    for k in ("BASE_URL", "MODEL", "MODEL_TOTAL_PARAMS_B", "MAX_MODEL_LEN", "CONCURRENCY", "ROOT"):
        print(f"{k:22s} = {globals()[k]}")
