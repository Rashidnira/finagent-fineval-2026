# Computing environment for the component grid

Recorded 2026-09-09 from values captured when the analysis was run; no server was reachable, so nothing was re-queried.

This environment differs from the one used for the submitted results: the context window is larger, and this Ollama build selects the Vulkan backend where the earlier runs used CUDA. All four cells of the grid ran in this single environment, so comparisons between cells are unaffected. The grid is a new controlled analysis and is not a reproduction of the submitted runs.

## Model

| Property | Value |
|---|---|
| Name | `qwen3:32b` |
| Architecture | qwen3 |
| Parameter count, from the served file | 32,762,123,264 |
| Reported size | 32.8B |
| Quantization | Q4_K_M |
| Server | Ollama 0.30.6 |

## Backend

| Property | Value |
|---|---|
| Backend | Vulkan (llama.cpp Vulkan backend) |
| Device | NVIDIA H100 NVL |
| Layers offloaded to GPU | 65 / 65 |
| Model buffer on device | 18,842 MiB |
| KV cache on device | 10,240 MiB (40,960 cells, f16 K and V) |
| CPU-mapped model buffer | 417 MiB |
| GPU utilisation during generation | 98 % |
| Verification | nvidia-smi showed the llama-server process holding 30,139 MiB on the device; no CPU fallback |

## Decoding

| Setting | Value |
|---|---|
| Temperature | 0.7 |
| top-p | 0.8 |
| top-k | 20 |
| Maximum output tokens | 4096 |
| Context window | 40,960 |
| Reasoning mode | disabled |
| Seeds | 1, 2, 3, 4, 5 |

The submitted system used the same sampling settings with no seed and a 32,768-token window. The window was raised for the grid because the largest grouped prompt with retrieved evidence is 32,668 tokens, which leaves no room for output at the smaller size.

## Data and code

| Item | Value |
|---|---|
| Dataset | PolyFiQA-Easy development split, 76 instances |
| Prompts | `src/retrieval/build_grid.py` |
| Generation | `scripts/run_grid.py`, `scripts/run_grid_all.py` |
| Scoring | `scripts/score_grid.py` |
| Post-processing | none: no rewriting, length guard or language guard |

