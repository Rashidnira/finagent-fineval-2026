"""Record the exact computing environment used for the grid analysis.

    python scripts/record_grid_provenance.py

Queries the running server for its version and the served model's metadata,
combines that with the backend facts observed in the server log, and writes
reports/GRID_PROVENANCE.md for the appendix.

The grid ran in a different environment from the submitted system: a larger
context window, and the Vulkan backend of this Ollama build rather than the
CUDA backend used earlier. All four cells ran in this one environment, so
comparisons between cells are unaffected, but the grid is a new controlled
analysis and not a reproduction of the submitted runs.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/GRID_PROVENANCE.md"
BASE = os.environ.get("LOCAL_OLLAMA_BASE", "http://127.0.0.1:11500")

# Values recorded from the server that produced the analysis. They are used
# when no server is reachable, so the record can be regenerated from a fresh
# clone. Querying a live server overwrites them with what it reports.
RECORDED = {
    "version": "0.30.6",
    "architecture": "qwen3",
    "parameter_count": 32762123264,
    "parameter_size": "32.8B",
    "quantization_level": "Q4_K_M",
}

# Observed in the server log and via nvidia-smi while the grid was running.
# These come from the host and cannot be queried through the HTTP API.
BACKEND_OBSERVED = {
    "backend": "Vulkan (llama.cpp Vulkan backend)",
    "device": "NVIDIA H100 NVL",
    "layers offloaded to GPU": "65 / 65",
    "model buffer on device": "18,842 MiB",
    "KV cache on device": "10,240 MiB (40,960 cells, f16 K and V)",
    "CPU-mapped model buffer": "417 MiB",
    "GPU utilisation during generation": "98 %",
    "verification": "nvidia-smi showed the llama-server process holding "
                    "30,139 MiB on the device; no CPU fallback",
}


def main() -> int:
    live = True
    try:
        version = requests.get(f"{BASE}/api/version", timeout=20).json()["version"]
        show = requests.post(f"{BASE}/api/show", json={"model": "qwen3:32b"},
                             timeout=30).json()
        det = show.get("details", {})
        info = show.get("model_info", {})
        params = info.get("general.parameter_count")
    except Exception:
        live = False
        version = RECORDED["version"]
        det = {"parameter_size": RECORDED["parameter_size"],
               "quantization_level": RECORDED["quantization_level"]}
        info = {"general.architecture": RECORDED["architecture"]}
        params = RECORDED["parameter_count"]
        print(f"note: no server at {BASE}; using the recorded values.\n")

    L: list[str] = []

    def w(s: str = "") -> None:
        L.append(s)

    w("# Computing environment for the component grid")
    w()
    w(f"Recorded {datetime.date.today().isoformat()} "
      + ("from the running server." if live else
         "from values captured when the analysis was run; no server was "
         "reachable, so nothing was re-queried."))
    w()
    w("This environment differs from the one used for the submitted results: the "
      "context window is larger, and this Ollama build selects the Vulkan "
      "backend where the earlier runs used CUDA. All four cells of the grid ran "
      "in this single environment, so comparisons between cells are unaffected. "
      "The grid is a new controlled analysis and is not a reproduction of the "
      "submitted runs.")
    w()
    w("## Model")
    w()
    w("| Property | Value |")
    w("|---|---|")
    w(f"| Name | `qwen3:32b` |")
    w(f"| Architecture | {info.get('general.architecture', 'n/a')} |")
    w(f"| Parameter count, from the served file | "
      f"{int(params):,} |" if params else "| Parameter count | n/a |")
    w(f"| Reported size | {det.get('parameter_size', 'n/a')} |")
    w(f"| Quantization | {det.get('quantization_level', 'n/a')} |")
    w(f"| Server | Ollama {version} |")
    w()
    w("## Backend")
    w()
    w("| Property | Value |")
    w("|---|---|")
    for k, v in BACKEND_OBSERVED.items():
        w(f"| {k[0].upper() + k[1:]} | {v} |")
    w()
    w("## Decoding")
    w()
    w("| Setting | Value |")
    w("|---|---|")
    w("| Temperature | 0.7 |")
    w("| top-p | 0.8 |")
    w("| top-k | 20 |")
    w("| Maximum output tokens | 4096 |")
    w("| Context window | 40,960 |")
    w("| Reasoning mode | disabled |")
    w("| Seeds | 1, 2, 3, 4, 5 |")
    w()
    w("The submitted system used the same sampling settings with no seed and a "
      "32,768-token window. The window was raised for the grid because the "
      "largest grouped prompt with retrieved evidence is 32,668 tokens, which "
      "leaves no room for output at the smaller size.")
    w()
    w("## Data and code")
    w()
    w("| Item | Value |")
    w("|---|---|")
    w("| Dataset | PolyFiQA-Easy development split, 76 instances |")
    w("| Prompts | `src/retrieval/build_grid.py` |")
    w("| Generation | `scripts/run_grid.py`, `scripts/run_grid_all.py` |")
    w("| Scoring | `scripts/score_grid.py` |")
    w("| Post-processing | none: no rewriting, length guard or language guard |")
    w()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[saved] {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
