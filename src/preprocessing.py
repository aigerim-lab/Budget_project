from __future__ import annotations
import pandas as pd
import numpy as np

def paysim_to_windows(df: pd.DataFrame, window: str = "day") -> pd.DataFrame:
    """
    Convert PaySim raw transactions to aggregated windows.
    In PaySim: step ~ hour. So:
      day  = step//24
      week = step//(24*7)
    """
    df = df.copy()
    if window == "day":
        df["win_id"] = (df["step"] // 24).astype(int)
    elif window == "week":
        df["win_id"] = (df["step"] // (24 * 7)).astype(int)
    else:
        raise ValueError("window must be 'day' or 'week'")

    agg = df.groupby(["nameOrig", "win_id"]).agg(
        total_spent=("amount", "sum"),
        txn_count=("amount", "count"),
        avg_txn=("amount", "mean"),
        max_txn=("amount", "max"),
    ).reset_index()

    # a simple “velocity” feature: change of total_spent vs previous window for each user
    agg = agg.sort_values(["nameOrig", "win_id"])
    agg["prev_total_spent"] = agg.groupby("nameOrig")["total_spent"].shift(1)
    agg["spending_velocity"] = (agg["total_spent"] - agg["prev_total_spent"]).fillna(0.0)

    return agg

def make_risk_label(df_agg: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
    """
    Create supervised label for 'RISK' vs 'OK' using a percentile rule.
    This makes the project clearly 'trained ML' (supervised).
    """
    df = df_agg.copy()
    thr = float(np.percentile(df["total_spent"], percentile))
    df["risk"] = (df["total_spent"] >= thr).astype(int)
    df["risk_threshold"] = thr
    return df
