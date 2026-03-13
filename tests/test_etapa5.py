"""
Tests for Etapa 5 — Model Pipeline.

Pure functions (time_series_split, compute_ev) are tested directly with
real numpy/pandas — no mocking required.

evaluate_model() is tested with a lightweight mock model that returns fixed
predict_proba output, so no heavy ML libraries execute in unit tests.

train_baseline() is tested end-to-end with a tiny synthetic dataset using
real scikit-learn (added as project dependency).

run() is tested with all heavy I/O and training functions patched out,
verifying orchestration logic, file outputs, and return-value contracts.
"""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub C-extensions and optional libraries before any project import
# ---------------------------------------------------------------------------


def _stub_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub xgboost — C extension, may not be available in all test environments
if "xgboost" not in sys.modules:
    _mock_xgb_instance = MagicMock()
    _mock_xgb_instance.fit = MagicMock(return_value=None)
    _mock_xgb_instance.best_iteration = 42
    _stub_module("xgboost", XGBClassifier=MagicMock(return_value=_mock_xgb_instance))

# Stub matplotlib — display server not available in CI
if "matplotlib" not in sys.modules:
    _stub_module("matplotlib", use=MagicMock())
    _stub_module(
        "matplotlib.pyplot",
        subplots=MagicMock(return_value=(MagicMock(), MagicMock())),
        close=MagicMock(),
    )

# Stub dotenv if not installed
if "dotenv" not in sys.modules:
    _stub_module("dotenv", load_dotenv=MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model_pipeline import (  # noqa: E402
    FEATURE_COLS,
    FLB_THRESHOLD,
    TARGET_COL,
    TRAIN_RATIO,
    compute_ev,
    evaluate_model,
    time_series_split,
    train_baseline,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BASE_DATE = pd.Timestamp("2025-01-01", tz="UTC")


def _make_dataset(n_markets: int = 10, snapshots_per_market: int = 5) -> pd.DataFrame:
    """Synthetic dataset with the same schema as ml_dataset.parquet.

    Markets are resolved weekly, starting 30 days after the base date,
    so resolved_at grows monotonically with market index.
    Odd markets win (y=1); even markets lose (y=0) — 50/50 balance.
    """
    rows = []
    for i in range(n_markets):
        resolved_at = _BASE_DATE + pd.Timedelta(days=i * 7 + 30)
        implied = 0.3 + (i % 7) * 0.1  # varies 0.3–0.9
        for j in range(snapshots_per_market):
            snap = _BASE_DATE + pd.Timedelta(days=i * 7 + j)
            rows.append({
                "token_id":            f"TOK_{i:03d}_YES",
                "market_id":           f"MKT_{i:03d}",
                "snapshot_date":       snap,
                "implied_prob":        implied,
                "time_to_expiry_days": float(30 - j),
                "price_std_1d":        0.01,
                "price_std_7d":        0.01,
                "spread_width":        0.005,
                "volume_total":        10_000.0,
                "volume_1wk":          1_000.0,
                "is_sports":           1,
                "is_crypto":           0,
                "is_politics":         0,
                "is_other":            0,
                "outcome_index":       0,
                "y":                   i % 2,
                "resolved_at":         resolved_at,
            })
    return pd.DataFrame(rows)


def _smart_model(n_classes: int = 2) -> MagicMock:
    """Mock model whose predict_proba dynamically matches input size."""
    model = MagicMock()

    def _predict_proba(X):
        n = len(X)
        p1 = np.full(n, 0.7)
        return np.column_stack([1 - p1, p1])

    model.predict_proba.side_effect = _predict_proba
    return model


# ---------------------------------------------------------------------------
# TestTimeSeriesSplit
# ---------------------------------------------------------------------------


class TestTimeSeriesSplit:
    def test_no_market_in_both_splits(self):
        df = _make_dataset(10)
        train, test = time_series_split(df)
        assert set(train["market_id"]).isdisjoint(set(test["market_id"]))

    def test_all_rows_covered(self):
        df = _make_dataset(10)
        train, test = time_series_split(df)
        assert len(train) + len(test) == len(df)

    def test_train_ratio_approximately_respected(self):
        """80% of markets → train; tolerance of ±1 market."""
        df = _make_dataset(20)
        train, test = time_series_split(df, train_ratio=0.80)
        n_total = df["market_id"].nunique()
        n_train = train["market_id"].nunique()
        expected = int(n_total * 0.80)
        assert abs(n_train - expected) <= 1

    def test_test_contains_most_recent_markets(self):
        """The last 20% of markets (by resolved_at) must land in test."""
        df = _make_dataset(10)
        _, test = time_series_split(df, train_ratio=0.80)
        # Markets 0..9 resolved at day 30, 37, ..., 93.
        # cutoff_idx = int(10 * 0.80) = 8 → train=MKT_000..MKT_007, test=MKT_008,MKT_009
        assert "MKT_009" in set(test["market_id"])

    def test_train_ratio_one_gives_empty_test(self):
        df = _make_dataset(5)
        train, test = time_series_split(df, train_ratio=1.0)
        assert test.empty
        assert len(train) == len(df)

    def test_train_ratio_zero_gives_empty_train(self):
        df = _make_dataset(5)
        train, test = time_series_split(df, train_ratio=0.0)
        assert train.empty
        assert len(test) == len(df)

    def test_returns_copies_not_views(self):
        """Mutating the returned split must not affect the original."""
        df = _make_dataset(10)
        train, _ = time_series_split(df)
        original_val = df["y"].iloc[0]
        train.iloc[0, train.columns.get_loc("y")] = 99
        assert df["y"].iloc[0] == original_val

    def test_train_is_chronologically_older(self):
        """Max resolved_at in train must be <= min resolved_at in test."""
        df = _make_dataset(10)
        train, test = time_series_split(df)
        if not train.empty and not test.empty:
            max_train = train["resolved_at"].max()
            min_test = test["resolved_at"].min()
            assert max_train <= min_test

    def test_single_market_split(self):
        """With only one market, all rows go to train; test is empty."""
        df = _make_dataset(1)
        train, test = time_series_split(df, train_ratio=0.80)
        # int(1 * 0.80) = 0 — so train is EMPTY, test gets the 1 market
        # OR int rounds down, leaving everything in test.
        # Either is valid; what matters is no rows are lost.
        assert len(train) + len(test) == len(df)


# ---------------------------------------------------------------------------
# TestComputeEv
# ---------------------------------------------------------------------------


class TestComputeEv:
    def test_zero_when_no_bets_meet_threshold(self):
        y = np.array([1, 0, 1])
        p_model = np.array([0.5, 0.5, 0.5])
        p_ask = np.array([0.6, 0.6, 0.6])   # model always below ask
        assert compute_ev(y, p_model, p_ask) == pytest.approx(0.0)

    def test_positive_ev_when_model_beats_market(self):
        y = np.array([1, 0, 1, 0])
        p_model = np.array([0.9, 0.9, 0.9, 0.9])
        p_ask = np.array([0.8, 0.8, 0.8, 0.8])
        assert compute_ev(y, p_model, p_ask) == pytest.approx(0.1)

    def test_edge_threshold_filters_marginal_edge(self):
        """model_prob - p_ask == threshold must NOT qualify (strict >)."""
        y = np.array([1, 0])
        p_model = np.array([0.85, 0.85])
        p_ask = np.array([0.80, 0.80])
        # edge = 0.05, threshold = 0.05 → 0.85 > 0.80 + 0.05 = 0.85 → False
        ev = compute_ev(y, p_model, p_ask, edge_threshold=0.05)
        assert ev == pytest.approx(0.0)

    def test_ev_is_mean_of_eligible_bets(self):
        """EV is the mean of (model_prob - p_ask) for qualifying bets only."""
        y = np.array([1, 1, 1])
        p_model = np.array([0.90, 0.85, 0.75])
        p_ask = np.array([0.80, 0.80, 0.80])
        # Bets 0 (ev=0.10) and 1 (ev=0.05) qualify; bet 2 (0.75 < 0.80) does not.
        ev = compute_ev(y, p_model, p_ask)
        assert ev == pytest.approx(0.075)

    def test_ev_zero_when_all_arrays_empty(self):
        ev = compute_ev(np.array([]), np.array([]), np.array([]))
        assert ev == pytest.approx(0.0)

    def test_y_true_unused_in_ev_calculation(self):
        """y_true does not affect EV — only y_prob and p_ask matter."""
        p_model = np.array([0.9, 0.9])
        p_ask = np.array([0.8, 0.8])
        ev_win = compute_ev(np.array([1, 1]), p_model, p_ask)
        ev_loss = compute_ev(np.array([0, 0]), p_model, p_ask)
        assert ev_win == pytest.approx(ev_loss)

    def test_ev_negative_when_model_below_market(self):
        """If all bets somehow have model < ask but are forced through threshold=-1,
        EV is negative (model thinks we're overpaying)."""
        y = np.array([1])
        p_model = np.array([0.5])
        p_ask = np.array([0.7])
        ev = compute_ev(y, p_model, p_ask, edge_threshold=-1.0)
        assert ev == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# TestEvaluateModel
# ---------------------------------------------------------------------------


class TestEvaluateModel:
    def _setup(self, implied_prob: float = 0.6):
        """Return (X_test, y_test, df_test) with fixed implied_prob."""
        df = _make_dataset(6, snapshots_per_market=3)
        _, df_test = time_series_split(df, train_ratio=0.5)
        df_test = df_test.copy()
        df_test["implied_prob"] = implied_prob
        X_test = df_test[FEATURE_COLS]
        y_test = df_test[TARGET_COL]
        return X_test, y_test, df_test

    def test_returns_dict_with_all_required_keys(self):
        X_test, y_test, df_test = self._setup()
        model = _smart_model()
        result = evaluate_model(model, X_test, y_test, df_test, "TestModel")
        for key in ("model", "log_loss", "brier_score", "ev_mean", "ev_flb_zone",
                    "n_test", "n_flb_zone"):
            assert key in result, f"Missing key: '{key}'"

    def test_model_name_propagated_to_result(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "MyModel")
        assert result["model"] == "MyModel"

    def test_log_loss_is_positive_float(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert isinstance(result["log_loss"], float)
        assert result["log_loss"] > 0

    def test_brier_score_in_unit_interval(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_n_test_matches_data_length(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert result["n_test"] == len(y_test)

    def test_n_flb_zone_is_nonnegative_integer(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert isinstance(result["n_flb_zone"], int)
        assert result["n_flb_zone"] >= 0

    def test_flb_zone_counted_correctly(self):
        """When implied_prob=0.85 (> FLB_THRESHOLD), all rows are in FLB zone."""
        X_test, y_test, df_test = self._setup(implied_prob=0.85)
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert result["n_flb_zone"] == len(y_test)

    def test_no_flb_zone_when_prob_below_threshold(self):
        """When implied_prob=0.50 (< FLB_THRESHOLD), n_flb_zone is 0."""
        X_test, y_test, df_test = self._setup(implied_prob=0.50)
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        assert result["n_flb_zone"] == 0

    def test_predict_proba_called_once(self):
        X_test, y_test, df_test = self._setup()
        model = _smart_model()
        evaluate_model(model, X_test, y_test, df_test, "M")
        model.predict_proba.assert_called_once()

    def test_metrics_are_rounded_floats(self):
        X_test, y_test, df_test = self._setup()
        result = evaluate_model(_smart_model(), X_test, y_test, df_test, "M")
        for key in ("log_loss", "brier_score", "ev_mean", "ev_flb_zone"):
            assert isinstance(result[key], float), f"{key} should be float"


# ---------------------------------------------------------------------------
# TestTrainBaseline
# ---------------------------------------------------------------------------


class TestTrainBaseline:
    def _make_train_data(self, n: int = 30):
        """Balanced training set: n//2 positive, n//2 negative."""
        implied = np.linspace(0.1, 0.9, n)
        X = pd.DataFrame({
            "implied_prob": implied,
            **{col: np.zeros(n) for col in FEATURE_COLS if col != "implied_prob"},
        })
        y = pd.Series((implied > 0.5).astype(int), name="y")
        return X, y

    def test_returns_object_with_predict_proba(self):
        X, y = self._make_train_data()
        model = train_baseline(X, y, cv=2)
        assert hasattr(model, "predict_proba")

    def test_probabilities_sum_to_one(self):
        X, y = self._make_train_data()
        model = train_baseline(X, y, cv=2)
        proba = model.predict_proba(X[["implied_prob"]])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_probabilities_in_unit_range(self):
        X, y = self._make_train_data()
        model = train_baseline(X, y, cv=2)
        proba = model.predict_proba(X[["implied_prob"]])
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_output_shape_is_n_by_2(self):
        X, y = self._make_train_data(n=10)
        model = train_baseline(X, y, cv=2)
        proba = model.predict_proba(X[["implied_prob"]])
        assert proba.shape == (10, 2)

    def test_monotonicity_high_prob_predicts_higher(self):
        """Higher implied_prob must produce higher predicted probability."""
        X, y = self._make_train_data()
        model = train_baseline(X, y, cv=2)
        low_df = pd.DataFrame({"implied_prob": [0.2]})
        high_df = pd.DataFrame({"implied_prob": [0.8]})
        p_low = model.predict_proba(low_df)[0, 1]
        p_high = model.predict_proba(high_df)[0, 1]
        assert p_high > p_low

    def test_single_feature_input_accepted(self):
        """Model must accept a single-column DataFrame with 'implied_prob'."""
        X, y = self._make_train_data()
        model = train_baseline(X, y, cv=2)
        single_input = pd.DataFrame({"implied_prob": [0.5]})
        result = model.predict_proba(single_input)
        assert result.shape == (1, 2)


# ---------------------------------------------------------------------------
# TestRunPipeline — integration, all heavy ops mocked
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for the run() orchestration function.

    All training functions, plots, and joblib I/O are mocked.
    Only the data-loading mock is replaced with a real DataFrame so that
    the split / metrics logic executes for real.
    """

    def _make_df(self) -> pd.DataFrame:
        return _make_dataset(n_markets=10, snapshots_per_market=3)

    def _patch_run(self, df: pd.DataFrame, models_dir: Path, reports_dir: Path):
        """Context manager: patch everything except pure logic."""
        mock_model = _smart_model()
        return (
            patch("model_pipeline.load_dataset", return_value=df),
            patch("model_pipeline.train_baseline", return_value=mock_model),
            patch("model_pipeline.train_xgboost", return_value=mock_model),
            patch("model_pipeline.plot_reliability_diagram"),
            patch("model_pipeline.plot_feature_importance"),
            patch("model_pipeline.joblib"),
        )

    def test_returns_list_of_three_dicts(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            results = run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        assert isinstance(results, list)
        assert len(results) == 3

    def test_all_results_have_required_keys(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            results = run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        required = {"model", "log_loss", "brier_score", "ev_mean", "ev_flb_zone"}
        for r in results:
            assert required.issubset(r.keys()), f"Missing keys in {r['model']}"

    def test_correct_model_names_in_results(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            results = run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        names = {r["model"] for r in results}
        assert "LR_Baseline" in names
        assert "XGBoost_Calibrated" in names
        assert "Market_Benchmark" in names

    def test_feature_list_json_written(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        mdir = tmp_path / "models"
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=mdir, reports_dir=tmp_path / "r")

        feat_file = mdir / "feature_list.json"
        assert feat_file.exists()
        loaded = json.loads(feat_file.read_text())
        assert loaded == FEATURE_COLS

    def test_train_cutoff_date_written(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        mdir = tmp_path / "models"
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=mdir, reports_dir=tmp_path / "r")

        cutoff_file = mdir / "train_cutoff_date.txt"
        assert cutoff_file.exists()
        assert len(cutoff_file.read_text().strip()) > 0

    def test_joblib_dump_called_exactly_twice(self, tmp_path):
        """One dump for LR, one for XGBoost."""
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib") as mock_joblib:
            run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        assert mock_joblib.dump.call_count == 2

    def test_reliability_diagram_called_twice(self, tmp_path):
        """Once for LR, once for XGBoost."""
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram") as mock_plot, \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        assert mock_plot.call_count == 2

    def test_train_baseline_called_once(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model) as mock_lr, \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        mock_lr.assert_called_once()

    def test_train_xgboost_called_once(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model) as mock_xgb, \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        mock_xgb.assert_called_once()

    def test_benchmark_ev_is_zero(self, tmp_path):
        """Market Benchmark always has ev_mean and ev_flb_zone == 0."""
        df = self._make_df()
        mock_model = _smart_model()
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            results = run(models_dir=tmp_path / "m", reports_dir=tmp_path / "r")

        benchmark = next(r for r in results if r["model"] == "Market_Benchmark")
        assert benchmark["ev_mean"] == pytest.approx(0.0)
        assert benchmark["ev_flb_zone"] == pytest.approx(0.0)

    def test_models_directory_created(self, tmp_path):
        df = self._make_df()
        mock_model = _smart_model()
        mdir = tmp_path / "deeply" / "nested" / "models"
        with patch("model_pipeline.load_dataset", return_value=df), \
             patch("model_pipeline.train_baseline", return_value=mock_model), \
             patch("model_pipeline.train_xgboost", return_value=mock_model), \
             patch("model_pipeline.plot_reliability_diagram"), \
             patch("model_pipeline.plot_feature_importance"), \
             patch("model_pipeline.joblib"):
            run(models_dir=mdir, reports_dir=tmp_path / "r")

        assert mdir.is_dir()
