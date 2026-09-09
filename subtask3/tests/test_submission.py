"""Phase 10 submission-builder integrity tests (no API, no real predictions)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.submission import SubmissionError, build_submission, write_submission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def official():
    return pd.read_parquet(ROOT / "data/manifests/official_finnlp_test.parquet")


def _dummy_preds(official, **overrides):
    df = pd.DataFrame({"id": official.id,
                       "prediction": ["Answer: x.\nFinancial Statements "
                                      "Evidence: None."] * len(official)})
    for k, v in overrides.items():
        setattr(df, k, v)
    return df


def test_happy_path(official, tmp_path):
    sub = build_submission(_dummy_preds(official), official)
    assert len(sub) == 76
    assert list(sub.columns) == ["id", "prediction"]
    assert (sub.id.values == official.id.values).all()  # official row order
    write_submission(sub, tmp_path / "s.csv")
    back = pd.read_csv(tmp_path / "s.csv", dtype=str)
    assert len(back) == 76 and list(back.columns) == ["id", "prediction"]


def test_quoting_roundtrip(official, tmp_path):
    preds = _dummy_preds(official)
    tricky = 'Answer: revenue was $7,059M, "up 15%".\nNews: line two.'
    preds.loc[0, "prediction"] = tricky
    sub = build_submission(preds, official)
    write_submission(sub, tmp_path / "s.csv")
    back = pd.read_csv(tmp_path / "s.csv", dtype=str)
    assert back.loc[0, "prediction"] == tricky


def test_missing_id_rejected(official):
    preds = _dummy_preds(official).iloc[1:]
    with pytest.raises(SubmissionError, match="missing ids"):
        build_submission(preds, official)


def test_duplicate_id_rejected(official):
    preds = _dummy_preds(official)
    preds.loc[1, "id"] = preds.loc[0, "id"]
    with pytest.raises(SubmissionError, match="duplicate|missing"):
        build_submission(preds, official)


def test_extra_id_rejected(official):
    preds = pd.concat([_dummy_preds(official),
                       pd.DataFrame({"id": ["poly_999"], "prediction": ["x"]})])
    with pytest.raises(SubmissionError, match="extra ids"):
        build_submission(preds, official)


def test_empty_prediction_rejected(official):
    preds = _dummy_preds(official)
    preds.loc[5, "prediction"] = "   "
    with pytest.raises(SubmissionError, match="empty predictions"):
        build_submission(preds, official)


def test_fabricated_ids_never_accepted(official):
    """Official-ID rule: renumbered/reconstructed ids must fail."""
    preds = _dummy_preds(official)
    preds["id"] = [f"row_{i}" for i in range(len(preds))]
    with pytest.raises(SubmissionError):
        build_submission(preds, official)
