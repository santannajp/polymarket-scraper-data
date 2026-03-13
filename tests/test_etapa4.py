"""
Tests for Etapa 4 — Feature Pipeline.

Transformation functions (compute_price_std_7d, encode_categories,
impute_missing_values, validate_no_leakage, build_quality_report) are pure
pandas operations — tested directly with real DataFrames, no mocking needed.

run() mocks the DB connection and filesystem I/O.
"""

import json
import sys
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub C-extensions before importing any project module
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


if "psycopg2" not in sys.modules:
    _stub_module("psycopg2", connect=MagicMock(), OperationalError=Exception)
    _stub_module("psycopg2.extras", Json=lambda x: x, execute_values=MagicMock())

if "dotenv" not in sys.modules:
    _stub_module("dotenv", load_dotenv=MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_pipeline import (  # noqa: E402
    FEATURE_COLS,
    PRICE_STD_DEFAULT,
    SPREAD_DEFAULT,
    VOLUME_DEFAULT,
    build_quality_report,
    compute_price_std_7d,
    encode_categories,
    impute_missing_values,
    run,
    validate_no_leakage,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_RESOLVED = pd.Timestamp("2026-03-10 12:00:00", tz="UTC")
_SNAPSHOT = pd.Timestamp("2026-03-08 00:00:00", tz="UTC")  # 2 days before


def _make_df(**overrides) -> pd.DataFrame:
    """
    Returns a minimal, valid single-row DataFrame representing one ML sample.
    All values satisfy the anti-leakage invariants by default.
    """
    base = {
        "token_id":           ["TOKEN_YES"],
        "market_id":          ["MARKET_001"],
        "snapshot_date":      [_SNAPSHOT],
        "implied_prob":       [0.85],
        "spread_width":       [0.01],
        "time_to_expiry_days":[2.0],
        "price_std_1d":       [0.02],
        "price_std_7d_sql":   [0.018],
        "price_std_7d":       [0.018],  # pre-populated for unit tests of imputation
        "volume_total":       [50_000.0],
        "volume_1wk":         [5_000.0],
        "category":           ["Sports"],
        "outcome_index":      [0],
        "y":                  [1],
        "resolved_at":        [_RESOLVED],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_multi_token_df() -> pd.DataFrame:
    """Two tokens across three days — useful for rolling window tests."""
    rows = []
    tokens = ["TOKEN_A", "TOKEN_B"]
    for tok in tokens:
        for day_offset in range(10):
            rows.append({
                "token_id":           tok,
                "market_id":          "MARKET_001",
                "snapshot_date":      pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_offset),
                "implied_prob":       0.8,
                "spread_width":       0.01,
                "time_to_expiry_days": float(30 - day_offset),
                "price_std_1d":       float(day_offset + 1),   # 1..10
                "price_std_7d_sql":   0.0,
                "volume_total":       10_000.0,
                "volume_1wk":         1_000.0,
                "category":           "Sports",
                "outcome_index":      0,
                "y":                  1,
                "resolved_at":        pd.Timestamp("2026-02-01", tz="UTC"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_price_std_7d
# ---------------------------------------------------------------------------


class TestComputePriceStd7d:
    def test_column_created(self):
        df = _make_df()
        result = compute_price_std_7d(df)
        assert "price_std_7d" in result.columns

    def test_single_row_equals_price_std_1d(self):
        """With only one data point per token, rolling mean == the value itself."""
        df = _make_df(price_std_1d=[0.05])
        result = compute_price_std_7d(df)
        assert result["price_std_7d"].iloc[0] == pytest.approx(0.05)

    def test_rolling_window_of_seven(self):
        """Day 7 (index 6) should equal mean of values 1..7."""
        df = _make_multi_token_df()
        result = compute_price_std_7d(df)
        token_a = result[result["token_id"] == "TOKEN_A"].sort_values("snapshot_date")
        # Day index 6 → price_std_1d values 1,2,3,4,5,6,7 → mean = 4.0
        day7 = token_a.iloc[6]["price_std_7d"]
        assert day7 == pytest.approx(4.0)

    def test_computed_independently_per_token(self):
        """Token B's rolling window must not bleed into Token A's computation."""
        df = _make_multi_token_df()
        result = compute_price_std_7d(df)
        ta = result[result["token_id"] == "TOKEN_A"].sort_values("snapshot_date")
        tb = result[result["token_id"] == "TOKEN_B"].sort_values("snapshot_date")
        # Both tokens have identical price_std_1d, so values must be equal
        pd.testing.assert_series_equal(
            ta["price_std_7d"].reset_index(drop=True),
            tb["price_std_7d"].reset_index(drop=True),
        )

    def test_min_periods_one_handles_sparse_tokens(self):
        """Tokens with fewer than 7 days must still get a valid (non-NaN) value."""
        df = _make_df(price_std_1d=[0.03])
        result = compute_price_std_7d(df)
        assert not result["price_std_7d"].isna().any()

    def test_original_df_not_mutated(self):
        df = _make_df()
        original_cols = set(df.columns)
        compute_price_std_7d(df)
        assert set(df.columns) == original_cols  # no in-place mutation


# ---------------------------------------------------------------------------
# encode_categories
# ---------------------------------------------------------------------------


class TestEncodeCategories:
    def _encode(self, category: str) -> dict:
        df = _make_df(category=[category])
        result = encode_categories(df)
        return {
            "is_sports":   int(result["is_sports"].iloc[0]),
            "is_crypto":   int(result["is_crypto"].iloc[0]),
            "is_politics": int(result["is_politics"].iloc[0]),
            "is_other":    int(result["is_other"].iloc[0]),
        }

    def test_sports_detected(self):
        enc = self._encode("Sports")
        assert enc["is_sports"] == 1
        assert enc["is_other"] == 0

    def test_crypto_detected(self):
        enc = self._encode("Crypto")
        assert enc["is_crypto"] == 1
        assert enc["is_other"] == 0

    def test_politics_detected_by_affair(self):
        enc = self._encode("US-current-affairs")
        assert enc["is_politics"] == 1

    def test_politics_detected_by_election(self):
        enc = self._encode("US Election 2024")
        assert enc["is_politics"] == 1

    def test_politics_detected_by_politic(self):
        enc = self._encode("Politics")
        assert enc["is_politics"] == 1

    def test_unknown_category_maps_to_is_other(self):
        enc = self._encode("Entertainment")
        assert enc["is_other"] == 1
        assert enc["is_sports"] == 0
        assert enc["is_crypto"] == 0
        assert enc["is_politics"] == 0

    def test_null_category_maps_to_is_other(self):
        df = _make_df(category=[None])
        result = encode_categories(df)
        assert result["is_other"].iloc[0] == 1

    def test_case_insensitive(self):
        enc_lower = self._encode("sports")
        enc_upper = self._encode("SPORTS")
        assert enc_lower["is_sports"] == enc_upper["is_sports"] == 1

    def test_is_other_zero_when_named_category_matches(self):
        for cat in ["Sports", "Crypto", "Politics"]:
            enc = self._encode(cat)
            assert enc["is_other"] == 0, f"is_other should be 0 for '{cat}'"

    def test_all_four_columns_created(self):
        df = _make_df()
        result = encode_categories(df)
        for col in ("is_sports", "is_crypto", "is_politics", "is_other"):
            assert col in result.columns

    def test_original_df_not_mutated(self):
        df = _make_df()
        original_cols = set(df.columns)
        encode_categories(df)
        assert "is_sports" not in original_cols  # was not in original


# ---------------------------------------------------------------------------
# impute_missing_values
# ---------------------------------------------------------------------------


class TestImputeMissingValues:
    def test_spread_nan_filled_with_default(self):
        df = _make_df(spread_width=[None])
        result = impute_missing_values(df)
        assert result["spread_width"].iloc[0] == pytest.approx(SPREAD_DEFAULT)

    def test_price_std_1d_nan_filled_with_zero(self):
        df = _make_df(price_std_1d=[None])
        result = impute_missing_values(df)
        assert result["price_std_1d"].iloc[0] == pytest.approx(PRICE_STD_DEFAULT)

    def test_price_std_7d_nan_filled_with_zero(self):
        df = _make_df(price_std_7d=[None])
        result = impute_missing_values(df)
        assert result["price_std_7d"].iloc[0] == pytest.approx(PRICE_STD_DEFAULT)

    def test_volume_total_nan_filled_with_zero(self):
        df = _make_df(volume_total=[None])
        result = impute_missing_values(df)
        assert result["volume_total"].iloc[0] == pytest.approx(VOLUME_DEFAULT)

    def test_volume_1wk_nan_filled_with_zero(self):
        df = _make_df(volume_1wk=[None])
        result = impute_missing_values(df)
        assert result["volume_1wk"].iloc[0] == pytest.approx(VOLUME_DEFAULT)

    def test_non_null_values_not_changed(self):
        df = _make_df(spread_width=[0.05], volume_total=[99_000.0])
        result = impute_missing_values(df)
        assert result["spread_width"].iloc[0] == pytest.approx(0.05)
        assert result["volume_total"].iloc[0] == pytest.approx(99_000.0)

    def test_original_df_not_mutated(self):
        df = _make_df(spread_width=[None])
        impute_missing_values(df)
        assert pd.isna(df["spread_width"].iloc[0])  # original untouched


# ---------------------------------------------------------------------------
# validate_no_leakage
# ---------------------------------------------------------------------------


class TestValidateNoLeakage:
    def _valid_df(self) -> pd.DataFrame:
        df = _make_df()
        df = compute_price_std_7d(df)
        df = encode_categories(df)
        return impute_missing_values(df)

    def test_valid_data_passes_without_error(self):
        df = self._valid_df()
        validate_no_leakage(df)  # must not raise

    def test_raises_when_snapshot_equals_resolved_at(self):
        df = self._valid_df()
        # Make snapshot_date == resolved_at date
        df.loc[0, "snapshot_date"] = _RESOLVED
        with pytest.raises(AssertionError, match="leakage"):
            validate_no_leakage(df)

    def test_raises_when_snapshot_after_resolved_at(self):
        df = self._valid_df()
        df.loc[0, "snapshot_date"] = _RESOLVED + pd.Timedelta(days=1)
        with pytest.raises(AssertionError):
            validate_no_leakage(df)

    def test_raises_when_time_to_expiry_is_zero(self):
        df = self._valid_df()
        df.loc[0, "time_to_expiry_days"] = 0.0
        with pytest.raises(AssertionError, match="leakage"):
            validate_no_leakage(df)

    def test_raises_when_time_to_expiry_is_negative(self):
        df = self._valid_df()
        df.loc[0, "time_to_expiry_days"] = -1.0
        with pytest.raises(AssertionError):
            validate_no_leakage(df)

    def test_clips_implied_prob_below_minimum(self):
        df = self._valid_df()
        df.loc[0, "implied_prob"] = 0.0
        result = validate_no_leakage(df)
        assert result["implied_prob"].iloc[0] == pytest.approx(0.001)

    def test_clips_implied_prob_above_maximum(self):
        df = self._valid_df()
        df.loc[0, "implied_prob"] = 1.0
        result = validate_no_leakage(df)
        assert result["implied_prob"].iloc[0] == pytest.approx(0.999)

    def test_valid_prob_not_clipped(self):
        df = self._valid_df()
        df.loc[0, "implied_prob"] = 0.85
        result = validate_no_leakage(df)
        assert result["implied_prob"].iloc[0] == pytest.approx(0.85)

    def test_returns_dataframe(self):
        df = self._valid_df()
        result = validate_no_leakage(df)
        assert isinstance(result, pd.DataFrame)

    def test_original_df_not_mutated(self):
        df = self._valid_df()
        df.loc[0, "implied_prob"] = 0.0
        original_val = df["implied_prob"].iloc[0]
        validate_no_leakage(df)
        assert df["implied_prob"].iloc[0] == original_val  # original untouched


# ---------------------------------------------------------------------------
# build_quality_report
# ---------------------------------------------------------------------------


class TestBuildQualityReport:
    def _full_df(self) -> pd.DataFrame:
        df = _make_df()
        df = compute_price_std_7d(df)
        df = encode_categories(df)
        return impute_missing_values(df)

    def test_contains_required_keys(self):
        df = self._full_df()
        report = build_quality_report(df)
        for key in (
            "total_rows", "unique_markets", "unique_tokens",
            "date_range_start", "date_range_end",
            "y_positive_rate", "pct_spread_real",
            "y_by_category", "generated_at",
        ):
            assert key in report, f"Missing key: {key}"

    def test_total_rows_matches_dataframe(self):
        df = self._full_df()
        assert build_quality_report(df)["total_rows"] == len(df)

    def test_unique_markets_count(self):
        df = self._full_df()
        assert build_quality_report(df)["unique_markets"] == 1

    def test_unique_tokens_count(self):
        df = self._full_df()
        assert build_quality_report(df)["unique_tokens"] == 1

    def test_y_positive_rate_correct(self):
        # y=1 for one row → 100%
        df = self._full_df()
        assert build_quality_report(df)["y_positive_rate"] == pytest.approx(1.0)

    def test_pct_spread_real_zero_when_all_imputed(self):
        """When spread_width equals SPREAD_DEFAULT everywhere, pct_spread_real = 0."""
        df = self._full_df()
        df.loc[0, "spread_width"] = SPREAD_DEFAULT
        assert build_quality_report(df)["pct_spread_real"] == pytest.approx(0.0)

    def test_pct_spread_real_one_when_all_real(self):
        """When spread_width != SPREAD_DEFAULT everywhere, pct_spread_real = 1."""
        df = self._full_df()
        df.loc[0, "spread_width"] = 0.05  # distinct from default
        assert build_quality_report(df)["pct_spread_real"] == pytest.approx(1.0)

    def test_generated_at_is_iso_string(self):
        df = self._full_df()
        report = build_quality_report(df)
        # Must be parseable as ISO datetime
        datetime.fromisoformat(report["generated_at"])


# ---------------------------------------------------------------------------
# run() — integration (mocked DB + filesystem)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def _make_raw_df(self) -> pd.DataFrame:
        """Minimal raw DataFrame as if returned by pd.read_sql."""
        return pd.DataFrame({
            "token_id":           ["T_YES", "T_NO"],
            "market_id":          ["M1",    "M1"],
            "snapshot_date":      [pd.Timestamp("2026-03-08"), pd.Timestamp("2026-03-08")],
            "implied_prob":       [0.85, 0.15],
            "spread_width":       [0.01, 0.01],
            "time_to_expiry_days":[2.0,  2.0],
            "price_std_1d":       [0.02, 0.02],
            "price_std_7d_sql":   [0.018, 0.018],
            "volume_total":       [50_000.0, 50_000.0],
            "volume_1wk":         [5_000.0,  5_000.0],
            "category":           ["Sports", "Sports"],
            "outcome_index":      [0, 1],
            "y":                  [1, 0],
            "resolved_at":        [
                pd.Timestamp("2026-03-10 12:00:00", tz="UTC"),
                pd.Timestamp("2026-03-10 12:00:00", tz="UTC"),
            ],
        })

    def test_parquet_written(self, tmp_path):
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            df = run(output_dir=tmp_path)

        assert (tmp_path / "ml_dataset.parquet").exists()

    def test_meta_json_written(self, tmp_path):
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            run(output_dir=tmp_path)

        meta_path = tmp_path / "ml_dataset_meta.json"
        assert meta_path.exists()
        report = json.loads(meta_path.read_text())
        assert report["total_rows"] == 2

    def test_returns_dataframe_with_feature_cols(self, tmp_path):
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            df = run(output_dir=tmp_path)

        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_connection_closed_on_success(self, tmp_path):
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            run(output_dir=tmp_path)

        conn.close.assert_called_once()

    def test_connection_closed_on_db_error(self, tmp_path):
        """Connection must be closed even when read_sql raises."""
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", side_effect=RuntimeError("DB down")):
            with pytest.raises(RuntimeError):
                run(output_dir=tmp_path)

        conn.close.assert_called_once()

    def test_empty_result_returns_early(self, tmp_path):
        conn = MagicMock()
        empty = pd.DataFrame()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=empty):
            df = run(output_dir=tmp_path)

        assert df.empty
        assert not (tmp_path / "ml_dataset.parquet").exists()

    def test_parquet_can_be_read_back(self, tmp_path):
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            original = run(output_dir=tmp_path)

        reloaded = pd.read_parquet(tmp_path / "ml_dataset.parquet")
        assert len(reloaded) == len(original)
        assert set(reloaded.columns) == set(original.columns)

    def test_y_balance_is_fifty_fifty(self, tmp_path):
        """Binary market: one YES win + one NO loss → y mean = 0.5."""
        conn = MagicMock()
        with patch("feature_pipeline.get_db_connection", return_value=conn), \
             patch("feature_pipeline.pd.read_sql", return_value=self._make_raw_df()):
            df = run(output_dir=tmp_path)

        assert df["y"].mean() == pytest.approx(0.5)
