"""
model_pipeline.py — Etapa 5

ML pipeline for Favorite-Longshot Bias (FLB) detection on Polymarket data.

Two models are trained and evaluated on strictly out-of-time test data:
  1. Logistic Regression baseline (Platt Scaling on implied_prob only) —
     detects and quantifies the FLB signal from prices alone.
  2. XGBoost Classifier (all features, isotonic calibration) —
     exploits all engineered features to improve probability estimates.

All splits are chronological by market resolved_at to prevent data leakage.
No market ever appears in both train and test.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "implied_prob",
    "time_to_expiry_days",
    "price_std_1d",
    "price_std_7d",
    "spread_width",
    "volume_total",
    "volume_1wk",
    "is_sports",
    "is_crypto",
    "is_politics",
    "outcome_index",
]

TARGET_COL: str = "y"

TAKER_FEE: float = 0.002      # 0.2% market-order taker fee (from CLOB API)
TRAIN_RATIO: float = 0.80     # outer train/test chronological split
INNER_TRAIN_RATIO: float = 0.80  # inner train/val split (for XGBoost early stopping)
FLB_THRESHOLD: float = 0.80   # implied_prob > 0.80 defines the "favorite" zone


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(parquet_path: Path) -> pd.DataFrame:
    """Load the ML feature dataset from a Parquet file.

    Args:
        parquet_path: Path to ml_dataset.parquet produced by feature_pipeline.py.

    Returns:
        DataFrame with all feature and metadata columns.
    """
    df = pd.read_parquet(parquet_path)
    logging.info(
        f"Loaded {len(df):,} rows "
        f"({df['market_id'].nunique():,} markets) from {parquet_path}"
    )
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically by market, preserving strict temporal order.

    Markets are sorted by their latest resolved_at date. The earliest
    ``train_ratio`` fraction of markets go to train; the rest to test.
    No market ever appears in both splits, ensuring clean out-of-time evaluation.

    Preventing per-snapshot leakage: splitting by market (not by snapshot_date)
    ensures the same market cannot appear as train (early snapshot) and test
    (late snapshot).

    Args:
        df:          Dataset with 'market_id' and 'resolved_at' columns.
        train_ratio: Fraction of markets (by resolved_at) assigned to train.

    Returns:
        (df_train, df_test) — non-overlapping DataFrames with .copy() applied.
    """
    market_dates = df.groupby("market_id")["resolved_at"].max().sort_values()
    cutoff_idx = int(len(market_dates) * train_ratio)
    train_ids = set(market_dates.iloc[:cutoff_idx].index)
    test_ids = set(market_dates.iloc[cutoff_idx:].index)
    df_train = df[df["market_id"].isin(train_ids)].copy()
    df_test = df[df["market_id"].isin(test_ids)].copy()
    logging.info(
        f"Split: train={len(df_train):,} rows ({len(train_ids)} markets), "
        f"test={len(df_test):,} rows ({len(test_ids)} markets)."
    )
    return df_train, df_test


# ---------------------------------------------------------------------------
# Expected Value computation
# ---------------------------------------------------------------------------


def compute_ev(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    p_ask: np.ndarray,
    edge_threshold: float = 0.0,
) -> float:
    """Compute mean Expected Value (EV) for bets where the model sees edge.

    A bet is placed when ``model_prob > p_ask + edge_threshold``.
    The EV per bet is defined as ``model_prob - p_ask``: the estimated
    probability edge over the market ask price.

    This is *expected* EV, not realised PnL. It quantifies how much the
    model believes it outprices the market, on average, for the selected bets.

    Args:
        y_true:         Observed binary outcomes (1=win, 0=loss). Not used in
                        EV computation; kept for signature uniformity.
        y_prob:         Model predicted win probabilities.
        p_ask:          Market ask price per outcome (cost to enter the bet).
        edge_threshold: Minimum required edge over ask to include a bet.
                        Default 0.0 means any predicted edge qualifies.

    Returns:
        Mean EV per eligible bet, or 0.0 if no bet meets the threshold.
    """
    mask = y_prob > (p_ask + edge_threshold)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_prob[mask] - p_ask[mask]))


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    model_name: str,
) -> Dict:
    """Evaluate a fitted model on the out-of-time test set.

    Uses ``implied_prob`` from df_test as a proxy for the market ask price.
    This is a conservative estimate: the real ask is slightly above the mid.

    Metrics:
      - log_loss:    Calibration quality; lower is better.
      - brier_score: Mean squared probability error; lower is better.
      - ev_mean:     Mean EV across all "would-bet" signals (no threshold).
      - ev_flb_zone: Mean EV restricted to the FLB zone (implied_prob > 0.80).

    Args:
        model:      Fitted sklearn-compatible estimator with predict_proba().
        X_test:     Feature matrix for the test set.
        y_test:     True binary labels.
        df_test:    Full test DataFrame (provides implied_prob as ask proxy).
        model_name: String identifier used in logging and result dict.

    Returns:
        Dict with model name and all computed metrics.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    p_ask = df_test["implied_prob"].values

    ll = log_loss(y_test, y_prob)
    bs = brier_score_loss(y_test, y_prob)
    ev = compute_ev(y_test.values, y_prob, p_ask)

    flb_mask = p_ask > FLB_THRESHOLD
    ev_flb = (
        compute_ev(y_test.values[flb_mask], y_prob[flb_mask], p_ask[flb_mask])
        if flb_mask.sum() > 0
        else 0.0
    )

    metrics = {
        "model":       model_name,
        "log_loss":    round(float(ll), 6),
        "brier_score": round(float(bs), 6),
        "ev_mean":     round(float(ev), 6),
        "ev_flb_zone": round(float(ev_flb), 6),
        "n_test":      int(len(y_test)),
        "n_flb_zone":  int(flb_mask.sum()),
    }
    logging.info(
        f"[{model_name}] log_loss={ll:.4f} brier={bs:.4f} "
        f"ev={ev:.5f} ev_flb={ev_flb:.5f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> Pipeline:
    """Train a Logistic Regression baseline with Platt Scaling calibration.

    Uses only ``implied_prob`` as input feature. This single-feature model
    directly tests whether the market's own prices are miscalibrated.
    If the Reliability Diagram shows the FLB pattern (curve above diagonal
    for high probabilities), the market systematically underprices favorites.

    Args:
        X_train: Training DataFrame — must contain 'implied_prob' column.
        y_train: Binary target labels.
        cv:      Cross-validation folds for Platt Scaling. Use cv=2 in tests.

    Returns:
        Fitted sklearn Pipeline (StandardScaler → CalibratedClassifierCV).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(
            LogisticRegression(max_iter=1000, random_state=42),
            method="sigmoid",
            cv=cv,
        )),
    ])
    pipeline.fit(X_train[["implied_prob"]], y_train)
    logging.info("LR baseline (Platt Scaling) trained.")
    return pipeline


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> CalibratedClassifierCV:
    """Train XGBoost with early stopping and isotonic calibration.

    XGBoost is fitted on X_train with early stopping monitored on X_val,
    preventing overfitting without a fixed n_estimators.

    Isotonic calibration (``cv='prefit'``) is applied on the val set after
    fitting. This corrects probability outputs in the tails — precisely the
    low/high range where the FLB pattern manifests.

    Args:
        X_train: Training features (FEATURE_COLS).
        y_train: Binary training labels.
        X_val:   Validation features (for early stopping and calibration).
        y_val:   Binary validation labels.

    Returns:
        CalibratedClassifierCV wrapping the fitted XGBClassifier.
    """
    from xgboost import XGBClassifier  # lazy import — avoid hard dep at module level

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(
        X_train[FEATURE_COLS],
        y_train,
        eval_set=[(X_val[FEATURE_COLS], y_val)],
        verbose=False,
    )
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
    calibrated.fit(X_val[FEATURE_COLS], y_val)
    best_iter = getattr(xgb, "best_iteration", "N/A")
    logging.info(
        f"XGBoost trained. Best iteration: {best_iter}. "
        f"Isotonic calibration applied on {len(y_val):,} val samples."
    )
    return calibrated


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
) -> None:
    """Save a Reliability Diagram (calibration curve) as PNG.

    A perfectly calibrated model lies on the diagonal (y=x). The FLB pattern
    appears as:
      - Points ABOVE the diagonal at high probabilities (favorites underpriced).
      - Points BELOW the diagonal at low probabilities (longshots overpriced).

    Args:
        y_true:     Observed binary outcomes.
        y_prob:     Model predicted probabilities.
        model_name: Legend label for the curve.
        save_path:  Output PNG path.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")  # headless rendering — no display required
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from sklearn.calibration import calibration_curve  # noqa: PLC0415

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "s-", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability Diagram — {model_name}")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Reliability diagram saved → {save_path}")


def plot_feature_importance(
    model: CalibratedClassifierCV,
    feature_names: List[str],
    save_path: Path,
) -> None:
    """Save XGBoost feature importance (gain) bar chart as PNG.

    Extracts the native ``gain`` importance from the underlying XGBoost
    booster. XGBoost names features as f0, f1, …, fn, so the indices are
    mapped back to the ordered ``feature_names`` list.

    Args:
        model:         Fitted CalibratedClassifierCV wrapping an XGBClassifier.
        feature_names: Feature names in training order (matches FEATURE_COLS).
        save_path:     Output PNG path.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # Unwrap calibrated model to access the raw XGBoost booster
    base_model = (
        model.calibrated_classifiers_[0].estimator
        if hasattr(model, "calibrated_classifiers_")
        else model
    )
    raw_scores: Dict[str, float] = base_model.get_booster().get_score(
        importance_type="gain"
    )
    # Map "f0" → feature_names[0], "f1" → feature_names[1], etc.
    named_scores = {
        feature_names[int(k[1:])]: v
        for k, v in raw_scores.items()
        if k[1:].isdigit() and int(k[1:]) < len(feature_names)
    }
    imp_series = pd.Series(named_scores).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(imp_series) * 0.5)))
    imp_series.plot(kind="barh", ax=ax)
    ax.set_xlabel("Gain")
    ax.set_title("XGBoost Feature Importance (Gain)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logging.info(f"Feature importance plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    data_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> List[Dict]:
    """Execute the full model training and evaluation pipeline.

    Steps:
      1. Load ml_dataset.parquet.
      2. Out-of-time outer split (80/20 by market resolved_at).
      3. Inner train/val split (80/20 within train, for XGBoost early stopping).
      4. Train LR baseline (Platt Scaling on implied_prob).
      5. Train XGBoost (all features, early stopping + isotonic calibration).
      6. Compute Market Benchmark metrics (raw implied_prob as predictor).
      7. Plot reliability diagrams + XGBoost feature importance.
      8. Serialize models and metadata to disk.
      9. Log comparison table.

    Args:
        data_path:   Path to ml_dataset.parquet (default: data/ml_dataset.parquet).
        models_dir:  Output directory for serialized models (default: models/).
        reports_dir: Output directory for plots (default: reports/).

    Returns:
        List of metric dicts — LR Baseline, XGBoost Calibrated, Market Benchmark.
    """
    parquet_path = data_path or (DATA_DIR / "ml_dataset.parquet")
    mdir = models_dir or MODELS_DIR
    rdir = reports_dir or REPORTS_DIR
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    df = load_dataset(parquet_path)

    # --- 2. Outer split ---
    df_train_full, df_test = time_series_split(df, TRAIN_RATIO)

    # --- 3. Inner split (for XGBoost early stopping and calibration) ---
    df_train, df_val = time_series_split(df_train_full, INNER_TRAIN_RATIO)

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL]
    X_val = df_val[FEATURE_COLS]
    y_val = df_val[TARGET_COL]
    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL]

    # --- 4. LR Baseline ---
    # Train on the full train set (no early stopping needed → more data = better)
    logging.info("Training LR baseline (Platt Scaling on implied_prob)...")
    lr = train_baseline(df_train_full, df_train_full[TARGET_COL])
    lr_metrics = evaluate_model(
        lr, df_test[["implied_prob"]], y_test, df_test, "LR_Baseline"
    )

    # --- 5. XGBoost ---
    logging.info("Training XGBoost with early stopping...")
    xgb = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb, X_test, y_test, df_test, "XGBoost_Calibrated")

    # --- 6. Market Benchmark (using raw implied_prob as the predictor) ---
    p_clipped = df_test["implied_prob"].clip(0.001, 0.999)
    benchmark_metrics = {
        "model":       "Market_Benchmark",
        "log_loss":    round(float(log_loss(y_test, p_clipped)), 6),
        "brier_score": round(float(brier_score_loss(y_test, p_clipped)), 6),
        "ev_mean":     0.0,   # market prices → zero EV by definition
        "ev_flb_zone": 0.0,
        "n_test":      int(len(y_test)),
        "n_flb_zone":  int((df_test["implied_prob"] > FLB_THRESHOLD).sum()),
    }

    all_metrics = [lr_metrics, xgb_metrics, benchmark_metrics]

    # --- 7. Plots ---
    lr_probs = lr.predict_proba(df_test[["implied_prob"]])[:, 1]
    plot_reliability_diagram(
        y_test.values, lr_probs, "LR Baseline",
        rdir / "reliability_diagram_lr.png",
    )

    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    plot_reliability_diagram(
        y_test.values, xgb_probs, "XGBoost Calibrated",
        rdir / "reliability_diagram_xgb.png",
    )

    try:
        plot_feature_importance(xgb, FEATURE_COLS, rdir / "feature_importance.png")
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Feature importance plot skipped: {exc}")

    # --- 8. Serialize models ---
    joblib.dump(lr, mdir / "lr_baseline.pkl")
    joblib.dump(xgb, mdir / "xgb_model.pkl")
    (mdir / "feature_list.json").write_text(
        json.dumps(FEATURE_COLS, indent=2)
    )
    (mdir / "train_cutoff_date.txt").write_text(
        str(df_train_full["resolved_at"].max())
    )
    logging.info(f"Models serialized → {mdir}")

    # --- 9. Comparison table ---
    logging.info("=== Model Comparison (out-of-time test set) ===")
    for m in all_metrics:
        logging.info(
            f"  {m['model']:25s} | log_loss={m['log_loss']:.4f} "
            f"brier={m['brier_score']:.4f} "
            f"ev={m['ev_mean']:.5f} ev_flb={m['ev_flb_zone']:.5f}"
        )

    return all_metrics


if __name__ == "__main__":
    run()
