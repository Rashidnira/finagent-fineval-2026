"""Table 8 - Model eligibility: the parameter count of every learned component.

WHAT THIS REPRODUCES
    Appendix C.2 of the paper. The shared task limits a submitted model to
    70 billion total parameters. This script reports the parameter count of
    each learned component in the pipeline and their sum.

    The BGE-M3 count is computed directly from the released weight file if
    those weights are present locally, rather than taken from a model card.
    Qwen3-32B is served as quantized GGUF weights, so its count is taken from
    the official model card; quantization changes the precision at which
    weights are stored, not how many there are.

HOW LONG        a few seconds, no GPU, no model download
WHAT IT NEEDS   nothing required; BGE-M3 weights are used if already cached
OUTPUT          reproduce/results/table_08_model_parameters.txt
"""
from __future__ import annotations

import glob
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce._common import Report                       # noqa: E402

LIMIT_B = 70.0
QWEN_PARAMS = 32_762_123_264  # general.parameter_count of the served GGUF file
QWEN_PARAMS_B = QWEN_PARAMS / 1e9   # 32.76B; displays as 32.8B, sums exactly
BGE_M3_KNOWN = 567_754_752    # verified from the released weights


def count_safetensors_params(path: str) -> int:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        size = 1
        for dim in meta["shape"]:
            size *= dim
        total += size
    return total


def find_bge_weights() -> str | None:
    patterns = [
        str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/model.safetensors"),
        str(ROOT / "models/bge-m3/model.safetensors"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    return None


def main() -> int:
    rep = Report("table_08_model_parameters")
    rep.head("Table 8 - Learned components and their total parameter counts")

    weights = find_bge_weights()
    if weights:
        bge = count_safetensors_params(weights)
        source = "computed from the released weight file"
    else:
        bge = BGE_M3_KNOWN
        source = ("previously verified value; the weight file was not found locally, so it "
                  "was not recomputed. Run any retrieval step once to download it, then "
                  "re-run this script to verify.")

    total_b = QWEN_PARAMS_B + bge / 1e9

    rep.table(["Component", "Role", "Architecture", "Total parameters"],
              [("Qwen3-32B", "Answer generation and rewriting", "Dense", f"{QWEN_PARAMS_B:.1f}B"),
               ("BGE-M3", "Dense passage retrieval", "Dense (XLM-R large)",
                f"{bge / 1e9:.2f}B"),
               ("Chunker and deterministic checks", "Chunking and validation",
                "No learned weights", "0"),
               ("Total", "", "", f"{total_b:.1f}B")])

    rep.p(f"Exact counts: Qwen3-32B {QWEN_PARAMS:,} "
          f"(general.parameter_count of the served GGUF file) + BGE-M3 {bge:,} ({source}) "
          f"= {QWEN_PARAMS + bge:,} parameters in total.")
    rep.p(f"Shared task limit: {LIMIT_B:.0f}B total parameters. "
          f"Our combined learned stack is {total_b:.1f}B, which is "
          f"{'within' if total_b <= LIMIT_B else 'ABOVE'} the limit.")
    rep.p("Participation track: model-based. The OpenRouter requirement applies to API-based "
          "agent and workflow solutions; this pipeline makes no model API calls and serves "
          "open-weight models on local hardware.")
    rep.save()
    return 0 if total_b <= LIMIT_B else 1


if __name__ == "__main__":
    raise SystemExit(main())
