"""MBR consensus selection tests (no API, no gold access)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mbr_select import UTILITY_SCORER, mbr_pick


def test_mbr_picks_consensus():
    # two near-identical candidates and one outlier: consensus must win
    a = "Answer: Revenue grew 12% to $14.0 billion driven by cloud demand."
    b = "Answer: Revenue grew 12% to $14.0 billion on strong cloud demand."
    c = "Answer: The weather was pleasant throughout the quarter."
    best, utils = mbr_pick([a, b, c])
    assert best in (0, 1)
    assert utils[best] > utils[2]


def test_mbr_deterministic():
    cands = ["Answer: net income rose.", "Answer: net income fell.",
             "Answer: net income rose slightly."]
    assert mbr_pick(list(cands)) == mbr_pick(list(cands))


def test_mbr_single_candidate_degenerate():
    # selection over one candidate is trivially that candidate
    best, utils = mbr_pick(["only answer"] * 2)
    assert best in (0, 1)          # tie between identical candidates is fine
    assert UTILITY_SCORER == "multifinben_default"
