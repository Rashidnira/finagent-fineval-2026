# Legacy development records

Nothing in this directory describes the submitted system, and no code reads any
of it. These files are kept only so the development history is inspectable.

- `config.yaml` — a working configuration file from an earlier phase, when a
  hosted-API route (Gemma via OpenRouter) was still being evaluated. That route
  was never used for the submission.
- `DEVELOPMENT_NOTES.md` — a phase checklist from the same period, including
  tasks that were later dropped or superseded.

**The submitted system** generated every answer with Qwen3-32B (32.8B,
Apache-2.0) served locally through Ollama, with BGE-M3 (568M) for retrieval, and
made no model API call at any stage. See `../README.md`, `../BEST_SYSTEM.md` and
Appendix C.2 of the paper.
