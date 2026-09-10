"""Shared helpers for the reproduction scripts.

Every reproduce/table_*.py script writes a plain-text report into
reproduce/results/ and also prints it, so results can be read either in the
terminal or from the files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Report:
    """Collects lines, prints them, and writes them to reproduce/results/."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.lines: list[str] = []

    def _emit(self, s: str = "") -> None:
        print(s, flush=True)
        self.lines.append(s)

    def head(self, title: str) -> None:
        self._emit("=" * 78)
        self._emit(title)
        self._emit("=" * 78)
        self._emit()

    def p(self, text: str = "") -> None:
        self._emit(text)
        self._emit()

    def pre(self, text: str) -> None:
        for line in text.splitlines():
            self._emit("    " + line)
        self._emit()

    def table(self, header: list[str], rows: list[tuple]) -> None:
        cols = len(header)
        widths = [len(str(header[i])) for i in range(cols)]
        for r in rows:
            for i in range(cols):
                widths[i] = max(widths[i], len(str(r[i])))
        fmt = "  ".join("{:<%d}" % w for w in widths)
        self._emit(fmt.format(*header))
        self._emit("  ".join("-" * w for w in widths))
        for r in rows:
            self._emit(fmt.format(*[str(x) for x in r]))
        self._emit()

    def save(self) -> Path:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / f"{self.name}.txt"
        out.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"[saved] {out.relative_to(ROOT)}")
        return out
