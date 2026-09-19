from __future__ import annotations

from pathlib import Path

import pandas as pd

KAMEI_FEATURES: tuple[str, ...] = (
    "ns", "nd", "nf", "entropy",
    "la", "ld", "lt",
    "fix",
    "ndev", "age", "nuc", "exp", "rexp", "sexp",
)
LABEL_COL = "buggy"

REQUIRED_COLUMNS = (*KAMEI_FEATURES, LABEL_COL)


def load_apachejit_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The expected columns {missing} are missing from {path.name}."
            f"Actual columns: {list(df.columns)}."
            "Check the file schema—the column names may differ from the canonical ones."
        )

    return df


def load_train_test(
    data_dir: str | Path = "../../../data/raw",
    train_filename: str = "apachejit_train.csv",
    test_filename: str = "apachejit_test_large.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_df = load_apachejit_csv(data_dir / train_filename)
    test_df = load_apachejit_csv(data_dir / test_filename)
    return train_df, test_df