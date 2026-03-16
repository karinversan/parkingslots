from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def build_splits(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
    split_policy: str = "group",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    if split_policy == "group" and "group_key" in df.columns:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, temp_idx = next(gss.split(df, groups=df["group_key"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)

        rel_val = val_size / (val_size + test_size)
        gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1)
        val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["group_key"]))
        val_df = temp_df.iloc[val_idx].reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        return train_df, val_df, test_df

    train_df, temp_df = train_test_split(
        df, train_size=train_size, random_state=seed, stratify=df["label"]
    )
    rel_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, train_size=rel_val, random_state=seed + 1, stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
