from __future__ import annotations

from pathlib import Path

import pandas as pd

from bugit.paths import RAW_DATA_DIR

KAMEI_FEATURES: tuple[str, ...] = (
    "ns", "nd", "nf", "ent",
    "la", "ld",
    "fix",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
)
LABEL_COL = "buggy"

ID_COLUMNS: tuple[str, ...] = ("commit_id", "project", "year", "author_date")

REQUIRED_COLUMNS = (*KAMEI_FEATURES, LABEL_COL, *ID_COLUMNS)


def load_apachejit_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The expected columns {missing} are missing from {path.name}. "
            f"Actual columns: {list(df.columns)}. "
            "Check the file schema—the column names may differ from the canonical ones."
        )

    return df


def load_train_test(
    data_dir: str | Path = RAW_DATA_DIR,
    train_filename: str = "apachejit_train.csv",
    test_filename: str = "apachejit_test_large.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_df = load_apachejit_csv(data_dir / train_filename)
    test_df = load_apachejit_csv(data_dir / test_filename)
    return train_df, test_df