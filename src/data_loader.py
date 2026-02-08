from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_paysim_csv(path: Path) -> pd.DataFrame:
    # minimal columns for behavior
    cols = ["step", "type", "amount", "nameOrig"]
    return pd.read_csv(path, usecols=cols, low_memory=False)

def load_creditcard_csv(path: Path) -> pd.DataFrame:
    # typical columns: Time, Amount, Class + PCA cols
    return pd.read_csv(path, low_memory=False)

def load_personal_expenses_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalize names to lowercase
    df.columns = df.columns.str.strip().str.lower()
    return df
