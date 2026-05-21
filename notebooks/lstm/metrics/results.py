#!/usr/bin/env python
# coding: utf-8

"""
Results script for the LSTM model following Fischer and Krauss (2018).

Paper checked:
Fischer, T. and Krauss, C. (2018). Deep learning with long short-term memory
networks for financial market predictions. European Journal of Operational
Research, 270(2), 654-669.

Important alignment notes:
- The paper evaluates the model out-of-sample on the trading period.
- Stocks are ranked by P(class 1), then the top k are bought and the flop k
  are sold short, with k in {10, 50, 100, 150, 200}.
- Accuracy is computed on those top/flop stocks: top stocks are correct when
  target == 1, flop stocks are correct when target == 0.
- Financial metrics require the raw next-day stock return. The LSTM sequence
  dataset contains the binary target, so this script optionally merges
  next-day raw returns from data/processed/dataset_with_returns.csv.
"""

from __future__ import annotations

## 0. Imports and global paths

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_DATASET_PATH = SCRIPT_DIR / "data" / "dataset_lstm.csv"
DEFAULT_MODEL_PATH = SCRIPT_DIR / "lstm_model.h5"
DEFAULT_RAW_RETURNS_PATH = REPO_ROOT / "data" / "processed" / "dataset_with_returns.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_outputs"

DEFAULT_K_VALUES = (10, 50, 100, 150, 200)
TRADING_DAYS_PER_YEAR = 252


## 1. Paper-aligned split helpers

def parse_return_col(col: str) -> int:
    """Parse columns such as R_-239, R_0 into sortable integers."""
    return int(col.split("_", 1)[1])


def get_return_columns(dataset_path: Path) -> list[str]:
    header = pd.read_csv(dataset_path, nrows=0).columns
    cols = [col for col in header if col.startswith("R_")]
    return sorted(cols, key=parse_return_col)


def get_unique_dates(dataset_path: Path) -> list[pd.Timestamp]:
    dates = pd.read_csv(dataset_path, usecols=["date"], parse_dates=["date"])
    return sorted(dates["date"].dropna().unique())


def choose_test_dates(
    unique_dates: list[pd.Timestamp],
    train_days: int = 750,
    test_days: int = 250,
    split_mode: str = "paper",
) -> tuple[list[pd.Timestamp], pd.Timestamp | None]:
    """
    Return out-of-sample dates.

    split_mode="paper":
        Matches lstm-network.py and the paper's logic: first 750 dates are
        training; everything after that is out-of-sample. If the sequence
        dataset has exactly 1000 dates, this yields 250 test dates.

    split_mode="last":
        Uses the last test_days dates literally. This is useful for notebooks,
        but can leak training dates if the model was trained with the first
        train_days dates and the sequence dataset has fewer than 1000 dates.
    """
    if not unique_dates:
        raise ValueError("No dates found in dataset.")

    if split_mode == "paper":
        split_idx = min(train_days - 1, len(unique_dates) - 1)
        split_date = pd.Timestamp(unique_dates[split_idx])
        test_dates = [pd.Timestamp(d) for d in unique_dates if pd.Timestamp(d) > split_date]
        return test_dates, split_date

    if split_mode == "last":
        n = min(test_days, len(unique_dates))
        return [pd.Timestamp(d) for d in unique_dates[-n:]], None

    raise ValueError("split_mode must be either 'paper' or 'last'.")


def load_test_sequences(
    dataset_path: Path,
    return_cols: list[str],
    test_dates: Iterable[pd.Timestamp],
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Load only rows needed for the out-of-sample period."""
    test_date_set = set(pd.to_datetime(list(test_dates)))
    usecols = ["ticker", "date", "target", *return_cols]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        dataset_path,
        usecols=usecols,
        parse_dates=["date"],
        chunksize=chunksize,
    ):
        mask = chunk["date"].isin(test_date_set)
        if mask.any():
            chunks.append(chunk.loc[mask].copy())

    if not chunks:
        raise ValueError("No test rows were loaded. Check split settings.")

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df


## 2. LSTM predictions

def load_lstm_model(model_path: Path):
    """
    Import TensorFlow lazily so the file can still be imported for analysis
    without immediately starting TensorFlow.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)


def predict_lstm(
    model,
    test_df: pd.DataFrame,
    return_cols: list[str],
    batch_size: int = 2048,
) -> pd.DataFrame:
    X = test_df[return_cols].to_numpy(dtype=np.float32, copy=True)
    X = X.reshape((len(test_df), len(return_cols), 1))

    probs = model.predict(X, batch_size=batch_size, verbose=1)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"Expected model probabilities with shape (n, 2), got {probs.shape}.")

    out = test_df[["ticker", "date", "target"]].copy()
    out["prob_class_0"] = probs[:, 0]
    out["prob_class_1"] = probs[:, 1]
    out["pred_label"] = np.argmax(probs, axis=1).astype(int)
    out["correct_all_universe"] = out["pred_label"].eq(out["target"])
    return out


## 3. Raw future returns for financial metrics

def load_raw_return_features(raw_returns_path: Path, max_abs_raw_return: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """
    Load raw returns and create:
    - future_return: raw one-day return from t to t+1, used for portfolio PnL.
    - past_5d_return: raw cumulative return over the 5 days ending at t,
      used for the paper's transparent short-term reversal strategy.
    """
    raw = pd.read_csv(
        raw_returns_path,
        usecols=["date", "Name", "return_1d"],
        parse_dates=["date"],
    )
    raw = raw.rename(columns={"Name": "ticker", "return_1d": "raw_return_1d"})
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    quality = {
        "raw_rows": int(len(raw)),
        "raw_return_min_before_filter": float(raw["raw_return_1d"].min()),
        "raw_return_max_before_filter": float(raw["raw_return_1d"].max()),
        "max_abs_raw_return_filter": max_abs_raw_return,
        "n_filtered_raw_returns": 0,
    }

    if max_abs_raw_return and max_abs_raw_return > 0:
        invalid = raw["raw_return_1d"].abs() > max_abs_raw_return
        quality["n_filtered_raw_returns"] = int(invalid.sum())
        quality["share_filtered_raw_returns"] = float(invalid.mean())
        raw.loc[invalid, "raw_return_1d"] = np.nan
    else:
        quality["share_filtered_raw_returns"] = 0.0

    raw["future_return"] = raw.groupby("ticker")["raw_return_1d"].shift(-1)

    def cumulative_5d(x: pd.Series) -> pd.Series:
        return (1.0 + x).rolling(window=5, min_periods=5).apply(np.prod, raw=True) - 1.0

    raw["past_5d_return"] = raw.groupby("ticker", group_keys=False)["raw_return_1d"].apply(cumulative_5d)
    quality["future_return_missing_after_filter"] = int(raw["future_return"].isna().sum())
    quality["past_5d_return_missing_after_filter"] = int(raw["past_5d_return"].isna().sum())
    return raw[["ticker", "date", "future_return", "past_5d_return"]], quality


def merge_raw_return_features(predictions: pd.DataFrame, raw_features: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(raw_features, on=["ticker", "date"], how="left")
    missing = merged["future_return"].isna().mean()
    if missing > 0.01:
        print(f"WARNING: {missing:.2%} of prediction rows have no future_return after merge.")
    return merged


## 4. Top/flop k selection and classification metrics

def select_top_flop_by_probability(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []

    for date, group in predictions.groupby("date", sort=True):
        ranked = group.sort_values("prob_class_1", ascending=False).reset_index(drop=True)
        if len(ranked) < 2 * k:
            continue

        top = ranked.head(k).copy()
        top["side"] = "long"
        top["selected_pred_label"] = 1
        top["rank_in_side"] = np.arange(1, len(top) + 1)
        top["correct_selected"] = top["target"].eq(1)

        flop = ranked.tail(k).copy()
        flop = flop.sort_values("prob_class_1", ascending=True).reset_index(drop=True)
        flop["side"] = "short"
        flop["selected_pred_label"] = 0
        flop["rank_in_side"] = np.arange(1, len(flop) + 1)
        flop["correct_selected"] = flop["target"].eq(0)

        selected.extend([top, flop])

    if not selected:
        return pd.DataFrame()

    return pd.concat(selected, ignore_index=True)


def classification_metrics_for_selection(selected: pd.DataFrame, k: int, model_name: str = "LSTM") -> dict:
    if selected.empty:
        return {
            "model": model_name,
            "k": k,
            "dates_used": 0,
            "n_selected": 0,
            "combined_accuracy": np.nan,
            "long_accuracy": np.nan,
            "short_accuracy": np.nan,
        }

    long_mask = selected["side"].eq("long")
    short_mask = selected["side"].eq("short")
    successes = int(selected["correct_selected"].sum())
    n = int(len(selected))

    return {
        "model": model_name,
        "k": k,
        "dates_used": int(selected["date"].nunique()),
        "n_selected": n,
        "combined_accuracy": float(selected["correct_selected"].mean()),
        "long_accuracy": float(selected.loc[long_mask, "correct_selected"].mean()),
        "short_accuracy": float(selected.loc[short_mask, "correct_selected"].mean()),
        "binomial_p_value_vs_50pct": float(stats.binom.sf(successes - 1, n, 0.5)),
    }


def overall_classification_metrics(predictions: pd.DataFrame) -> dict:
    y = predictions["target"].astype(int).to_numpy()
    yhat = predictions["pred_label"].astype(int).to_numpy()
    accuracy = float((y == yhat).mean())

    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tp = int(((y == 1) & (yhat == 1)).sum())

    precision_1 = tp / (tp + fp) if (tp + fp) else np.nan
    recall_1 = tp / (tp + fn) if (tp + fn) else np.nan
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) else np.nan

    return {
        "model": "LSTM",
        "scope": "all_test_universe",
        "n": int(len(predictions)),
        "dates": int(predictions["date"].nunique()),
        "accuracy": accuracy,
        "baseline_majority_accuracy": float(max(y.mean(), 1.0 - y.mean())),
        "precision_class_1": float(precision_1),
        "recall_class_1": float(recall_1),
        "f1_class_1": float(f1_1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


## 5. Statistical tests from the paper

def pesaran_timmermann_test(actual: np.ndarray, forecast: np.ndarray) -> dict:
    """
    Pesaran-Timmermann directional accuracy test.

    H0: predictions and responses are independently distributed.
    The p-value is upper-tail, as commonly used for directional accuracy.
    """
    y = np.asarray(actual).astype(int)
    z = np.asarray(forecast).astype(int)
    if y.shape != z.shape:
        raise ValueError("actual and forecast must have the same shape.")

    n = y.size
    pyz = np.mean(y == z)
    py = np.mean(y == 1)
    pz = np.mean(z == 1)

    qy = py * (1.0 - py) / n
    qz = pz * (1.0 - pz) / n
    p_ind = py * pz + (1.0 - py) * (1.0 - pz)

    var_success = p_ind * (1.0 - p_ind) / n
    var_ind = ((2.0 * py - 1.0) ** 2) * qz
    var_ind += ((2.0 * pz - 1.0) ** 2) * qy
    var_ind += 4.0 * qy * qz
    denom = math.sqrt(max(var_success - var_ind, 0.0))

    if denom == 0.0:
        stat = np.nan
        p_value = np.nan
    else:
        stat = (pyz - p_ind) / denom
        p_value = 1.0 - stats.norm.cdf(stat)

    return {
        "test": "Pesaran-Timmermann",
        "n": int(n),
        "directional_accuracy": float(pyz),
        "expected_accuracy_independence": float(p_ind),
        "statistic": float(stat),
        "p_value": float(p_value),
    }


def newey_west_mean_test(values: pd.Series | np.ndarray, lags: int = 1) -> dict:
    """
    Newey-West t-test for whether mean(values) differs from zero.

    The paper reports Newey-West standard errors with one-lag correction.
    """
    x = pd.Series(values).dropna().astype(float).to_numpy()
    n = x.size
    if n == 0:
        return {"n": 0, "mean": np.nan, "standard_error": np.nan, "t_statistic": np.nan, "p_value": np.nan}

    mean = float(x.mean())
    u = x - mean
    gamma0 = float(np.dot(u, u) / n)
    long_run_var = gamma0

    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(u[lag:], u[:-lag]) / n)
        long_run_var += 2.0 * weight * gamma

    se = math.sqrt(max(long_run_var, 0.0) / n)
    t_stat = mean / se if se else np.nan
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
    return {
        "n": int(n),
        "mean": mean,
        "standard_error": float(se),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
    }


def diebold_mariano_error_test(errors_i: np.ndarray, errors_j: np.ndarray, lags: int = 1) -> dict:
    """
    Diebold-Mariano style test on 0/1 classification errors.

    The paper uses 0 for correct and 1 for incorrect classifications in the
    k=10 top/flop portfolio. Here d = error_i - error_j. A negative mean
    means method i has lower error than method j.
    """
    e_i = np.asarray(errors_i).astype(float)
    e_j = np.asarray(errors_j).astype(float)
    if e_i.shape != e_j.shape:
        raise ValueError("DM error vectors must have the same shape.")

    nw = newey_west_mean_test(e_i - e_j, lags=lags)
    stat = nw["t_statistic"]
    lower_tail_p = stats.norm.cdf(stat) if not np.isnan(stat) else np.nan
    upper_tail_p = 1.0 - stats.norm.cdf(stat) if not np.isnan(stat) else np.nan

    return {
        "test": "Diebold-Mariano-on-classification-errors",
        "n": nw["n"],
        "mean_error_diff_i_minus_j": nw["mean"],
        "statistic": stat,
        "p_value_i_more_accurate_than_j": float(lower_tail_p),
        "p_value_i_less_accurate_than_j": float(upper_tail_p),
    }


## 6. Financial metrics from long-short daily returns

def select_reversal_strategy(predictions: pd.DataFrame, k: int) -> pd.DataFrame:
    """Paper-inspired transparent benchmark: long 5-day losers, short 5-day winners."""
    if "past_5d_return" not in predictions.columns:
        return pd.DataFrame()

    selected: list[pd.DataFrame] = []
    df = predictions.dropna(subset=["past_5d_return"]).copy()

    for date, group in df.groupby("date", sort=True):
        ranked = group.sort_values("past_5d_return", ascending=True).reset_index(drop=True)
        if len(ranked) < 2 * k:
            continue

        top = ranked.head(k).copy()
        top["side"] = "long"
        top["selected_pred_label"] = 1
        top["rank_in_side"] = np.arange(1, len(top) + 1)
        top["correct_selected"] = top["target"].eq(1)

        flop = ranked.tail(k).copy()
        flop = flop.sort_values("past_5d_return", ascending=False).reset_index(drop=True)
        flop["side"] = "short"
        flop["selected_pred_label"] = 0
        flop["rank_in_side"] = np.arange(1, len(flop) + 1)
        flop["correct_selected"] = flop["target"].eq(0)

        selected.extend([top, flop])

    if not selected:
        return pd.DataFrame()

    return pd.concat(selected, ignore_index=True)


def portfolio_daily_returns(
    selected: pd.DataFrame,
    k: int,
    model_name: str,
    transaction_cost_bps_per_half_turn: float = 5.0,
) -> pd.DataFrame:
    if selected.empty or "future_return" not in selected.columns:
        return pd.DataFrame()

    rows = []
    half_turn_cost = transaction_cost_bps_per_half_turn / 10_000.0
    round_turn_cost_per_leg = 2.0 * half_turn_cost

    for date, group in selected.dropna(subset=["future_return"]).groupby("date", sort=True):
        longs = group[group["side"].eq("long")]
        shorts = group[group["side"].eq("short")]
        if len(longs) < k or len(shorts) < k:
            continue

        long_before = float(longs["future_return"].mean())
        short_before = float(-shorts["future_return"].mean())
        combined_before = long_before + short_before

        long_after = long_before - round_turn_cost_per_leg
        short_after = short_before - round_turn_cost_per_leg
        combined_after = long_after + short_after

        rows.append(
            {
                "model": model_name,
                "k": k,
                "date": date,
                "long_return_before_costs": long_before,
                "short_return_before_costs": short_before,
                "long_short_return_before_costs": combined_before,
                "long_return_after_costs": long_after,
                "short_return_after_costs": short_after,
                "long_short_return_after_costs": combined_after,
                "n_long": int(len(longs)),
                "n_short": int(len(shorts)),
            }
        )

    return pd.DataFrame(rows)


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdown = 1.0 - wealth / running_max
    return float(drawdown.max())


def cvar(returns: pd.Series | np.ndarray, alpha: float) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    var = r.quantile(alpha)
    return float(r[r <= var].mean())


def annualized_return(returns: pd.Series | np.ndarray) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    cumulative = float((1.0 + r).prod())
    if cumulative <= 0:
        return np.nan
    return cumulative ** (TRADING_DAYS_PER_YEAR / len(r)) - 1.0


def annualized_downside_deviation(returns: pd.Series | np.ndarray, mar_daily: float = 0.0) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.nan
    downside = np.minimum(r - mar_daily, 0.0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS_PER_YEAR))


def financial_summary(
    daily_returns: pd.DataFrame,
    return_col: str,
    model_name: str,
    k: int,
    cost_label: str,
    risk_free_annual: float = 0.0,
) -> dict:
    r = daily_returns[return_col].dropna().astype(float)
    if r.empty:
        return {"model": model_name, "k": k, "cost_label": cost_label, "n_days": 0}

    nw = newey_west_mean_test(r, lags=1)
    ann_ret = annualized_return(r)
    ann_std = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    excess_ann = ann_ret - risk_free_annual
    downside_ann = annualized_downside_deviation(r)

    return {
        "model": model_name,
        "k": k,
        "cost_label": cost_label,
        "n_days": int(r.size),
        "mean_daily_return": float(r.mean()),
        "newey_west_se_lag1": nw["standard_error"],
        "newey_west_t_lag1": nw["t_statistic"],
        "newey_west_p_lag1": nw["p_value"],
        "min": float(r.min()),
        "q1": float(r.quantile(0.25)),
        "median": float(r.median()),
        "q3": float(r.quantile(0.75)),
        "max": float(r.max()),
        "share_positive": float((r > 0.0).mean()),
        "std_daily": float(r.std(ddof=1)),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),
        "var_1pct": float(r.quantile(0.01)),
        "cvar_1pct": cvar(r, 0.01),
        "var_5pct": float(r.quantile(0.05)),
        "cvar_5pct": cvar(r, 0.05),
        "max_drawdown": max_drawdown(r),
        "annualized_return": float(ann_ret),
        "annualized_excess_return": float(excess_ann),
        "annualized_std": ann_std,
        "annualized_downside_dev": downside_ann,
        "annualized_sharpe": float(excess_ann / ann_std) if ann_std else np.nan,
        "annualized_sortino": float(excess_ann / downside_ann) if downside_ann else np.nan,
    }


## 7. Monkey trading benchmark from the paper

def monkey_trading_returns(
    predictions: pd.DataFrame,
    k: int = 10,
    repetitions: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random long-short portfolios. The paper uses 100,000 repetitions.
    Keep repetitions configurable because the full setting can be slow.
    """
    if repetitions <= 0 or "future_return" not in predictions.columns:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    by_date = [
        group["future_return"].dropna().to_numpy(dtype=float)
        for _, group in predictions.groupby("date", sort=True)
        if group["future_return"].notna().sum() >= 2 * k
    ]

    rows = []
    for rep in range(repetitions):
        daily = []
        for returns in by_date:
            picks = rng.choice(len(returns), size=2 * k, replace=False)
            long_idx = picks[:k]
            short_idx = picks[k:]
            daily.append(float(returns[long_idx].mean() - returns[short_idx].mean()))
        rows.append({"monkey_id": rep, "mean_daily_return": float(np.mean(daily))})

    return pd.DataFrame(rows)


## 8. Black-box pattern diagnostics

def selected_average_sequences(test_df: pd.DataFrame, selected: pd.DataFrame, return_cols: list[str], k: int) -> pd.DataFrame:
    """
    Average standardized input sequences for top/flop stocks.
    This mirrors the paper's idea of inspecting common patterns in traded stocks.
    """
    if selected.empty:
        return pd.DataFrame()

    keys = selected[["ticker", "date", "side"]].copy()
    enriched = keys.merge(test_df[["ticker", "date", *return_cols]], on=["ticker", "date"], how="left")
    long_avg = enriched[enriched["side"].eq("long")][return_cols].mean()
    short_avg = enriched[enriched["side"].eq("short")][return_cols].mean()

    return pd.DataFrame(
        {
            "k": k,
            "timestep": [parse_return_col(c) for c in return_cols],
            "return_col": return_cols,
            "avg_standardized_return_long": long_avg.to_numpy(),
            "avg_standardized_return_short": short_avg.to_numpy(),
        }
    )


## 9. Output helpers

def save_csv(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def write_run_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved {path}")


## 10. Main pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LSTM results following Fischer and Krauss (2018).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--raw-returns", type=Path, default=DEFAULT_RAW_RETURNS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--test-days", type=int, default=250)
    parser.add_argument("--split-mode", choices=["paper", "last"], default="paper")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--risk-free-annual", type=float, default=0.0)
    parser.add_argument(
        "--max-abs-raw-return",
        type=float,
        default=0.5,
        help=(
            "Financial-metric sanity filter for local raw returns. Values with "
            "abs(return_1d) above this threshold are treated as missing. Use 0 "
            "to disable."
        ),
    )
    parser.add_argument("--monkey-reps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-test-rows", type=int, default=0, help="Debug only: limit loaded test rows.")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset dates and return columns...")
    return_cols = get_return_columns(args.dataset)
    unique_dates = get_unique_dates(args.dataset)
    test_dates, split_date = choose_test_dates(
        unique_dates,
        train_days=args.train_days,
        test_days=args.test_days,
        split_mode=args.split_mode,
    )

    metadata = {
        "dataset": str(args.dataset),
        "model": str(args.model),
        "raw_returns": str(args.raw_returns),
        "split_mode": args.split_mode,
        "train_days": args.train_days,
        "requested_test_days": args.test_days,
        "unique_sequence_dates": len(unique_dates),
        "first_sequence_date": min(unique_dates),
        "last_sequence_date": max(unique_dates),
        "split_date": split_date,
        "actual_test_dates": len(test_dates),
        "first_test_date": min(test_dates) if test_dates else None,
        "last_test_date": max(test_dates) if test_dates else None,
        "return_columns": len(return_cols),
        "max_abs_raw_return_filter": args.max_abs_raw_return,
        "paper_note": (
            "Fischer and Krauss use rolling 1000-day study periods with 750 training "
            "days and 250 trading days. This script defaults to the same train-days "
            "split used in lstm-network.py. If the prepared sequence dataset has fewer "
            "than 1000 unique dates, the resulting out-of-sample period can be below "
            "250 dates."
        ),
    }
    write_run_metadata(args.output_dir / "run_metadata.json", metadata)

    print("Loading out-of-sample rows...")
    test_df = load_test_sequences(args.dataset, return_cols, test_dates, chunksize=args.chunksize)
    if args.limit_test_rows > 0:
        test_df = test_df.head(args.limit_test_rows).copy()
        print(f"DEBUG: limited test rows to {len(test_df)}.")

    print("Loading model and making predictions...")
    model = load_lstm_model(args.model)
    predictions = predict_lstm(model, test_df, return_cols, batch_size=args.batch_size)

    if args.raw_returns.exists():
        print("Merging raw future returns for financial metrics...")
        raw_features, raw_quality = load_raw_return_features(
            args.raw_returns,
            max_abs_raw_return=args.max_abs_raw_return,
        )
        save_csv(pd.DataFrame([raw_quality]), args.output_dir / "raw_return_quality_report.csv")
        predictions = merge_raw_return_features(predictions, raw_features)
    else:
        print("WARNING: raw returns file not found. Financial metrics will be skipped.")

    if not args.no_save_predictions:
        save_csv(predictions, args.output_dir / "lstm_predictions_test.csv")

    print("Computing classification metrics...")
    overall = overall_classification_metrics(predictions)
    classification_rows = []
    statistical_rows = []
    pattern_rows = []
    all_portfolio_returns = []

    pt_all = pesaran_timmermann_test(
        predictions["target"].to_numpy(),
        predictions["pred_label"].to_numpy(),
    )
    pt_all.update({"model": "LSTM", "k": "all_test_universe"})
    statistical_rows.append(pt_all)

    for k in args.k_values:
        selected = select_top_flop_by_probability(predictions, k)
        save_csv(selected, args.output_dir / f"selected_lstm_k{k}.csv")

        classification_rows.append(classification_metrics_for_selection(selected, k, model_name="LSTM"))

        if not selected.empty:
            pt = pesaran_timmermann_test(
                selected["target"].to_numpy(),
                selected["selected_pred_label"].to_numpy(),
            )
            pt.update({"model": "LSTM", "k": k})
            statistical_rows.append(pt)

            pattern = selected_average_sequences(test_df, selected, return_cols, k)
            pattern_rows.append(pattern)

            daily = portfolio_daily_returns(
                selected,
                k=k,
                model_name="LSTM",
                transaction_cost_bps_per_half_turn=args.transaction_cost_bps,
            )
            all_portfolio_returns.append(daily)

    save_csv(pd.DataFrame([overall]), args.output_dir / "overall_classification_metrics.csv")
    save_csv(pd.DataFrame(classification_rows), args.output_dir / "top_flop_classification_metrics.csv")
    save_csv(pd.DataFrame(statistical_rows), args.output_dir / "statistical_tests.csv")

    if pattern_rows:
        save_csv(pd.concat(pattern_rows, ignore_index=True), args.output_dir / "top_flop_average_sequences.csv")

    print("Computing financial metrics when future_return is available...")
    financial_rows = []
    if all_portfolio_returns:
        portfolio_returns = pd.concat([x for x in all_portfolio_returns if not x.empty], ignore_index=True)
        save_csv(portfolio_returns, args.output_dir / "portfolio_daily_returns_lstm.csv")

        for (model_name, k), group in portfolio_returns.groupby(["model", "k"]):
            financial_rows.append(
                financial_summary(
                    group,
                    "long_short_return_before_costs",
                    model_name=model_name,
                    k=int(k),
                    cost_label="before_costs",
                    risk_free_annual=args.risk_free_annual,
                )
            )
            financial_rows.append(
                financial_summary(
                    group,
                    "long_short_return_after_costs",
                    model_name=model_name,
                    k=int(k),
                    cost_label="after_costs",
                    risk_free_annual=args.risk_free_annual,
                )
            )

    print("Computing short-term reversal benchmark when raw past_5d_return is available...")
    reversal_returns_all = []
    if "past_5d_return" in predictions.columns and "future_return" in predictions.columns:
        reversal_classification_rows = []
        for k in args.k_values:
            reversal_selected = select_reversal_strategy(predictions, k)
            save_csv(reversal_selected, args.output_dir / f"selected_short_term_reversal_k{k}.csv")
            reversal_classification_rows.append(classification_metrics_for_selection(reversal_selected, k, model_name="STR_5D"))

            reversal_daily = portfolio_daily_returns(
                reversal_selected,
                k=k,
                model_name="STR_5D",
                transaction_cost_bps_per_half_turn=args.transaction_cost_bps,
            )
            reversal_returns_all.append(reversal_daily)

        save_csv(
            pd.DataFrame(reversal_classification_rows),
            args.output_dir / "short_term_reversal_classification_metrics.csv",
        )

        if reversal_returns_all:
            reversal_returns = pd.concat([x for x in reversal_returns_all if not x.empty], ignore_index=True)
            save_csv(reversal_returns, args.output_dir / "portfolio_daily_returns_short_term_reversal.csv")

            for (model_name, k), group in reversal_returns.groupby(["model", "k"]):
                financial_rows.append(
                    financial_summary(
                        group,
                        "long_short_return_before_costs",
                        model_name=model_name,
                        k=int(k),
                        cost_label="before_costs",
                        risk_free_annual=args.risk_free_annual,
                    )
                )
                financial_rows.append(
                    financial_summary(
                        group,
                        "long_short_return_after_costs",
                        model_name=model_name,
                        k=int(k),
                        cost_label="after_costs",
                        risk_free_annual=args.risk_free_annual,
                    )
                )

    save_csv(pd.DataFrame(financial_rows), args.output_dir / "financial_summary.csv")

    print("Computing monkey trading benchmark if requested...")
    if args.monkey_reps > 0 and "future_return" in predictions.columns:
        monkey = monkey_trading_returns(
            predictions,
            k=10,
            repetitions=args.monkey_reps,
            seed=args.seed,
        )
        save_csv(monkey, args.output_dir / "monkey_trading_mean_returns.csv")
        if not monkey.empty:
            monkey_summary = pd.DataFrame(
                [
                    {
                        "k": 10,
                        "repetitions": args.monkey_reps,
                        "mean_of_mean_daily_returns": float(monkey["mean_daily_return"].mean()),
                        "std_of_mean_daily_returns": float(monkey["mean_daily_return"].std(ddof=1)),
                        "min_mean_daily_return": float(monkey["mean_daily_return"].min()),
                        "max_mean_daily_return": float(monkey["mean_daily_return"].max()),
                    }
                ]
            )
            save_csv(monkey_summary, args.output_dir / "monkey_trading_summary.csv")

    print("Done.")
    print(f"Outputs are in: {args.output_dir}")


if __name__ == "__main__":
    main()
