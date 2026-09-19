from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from bugit.data.load import KAMEI_FEATURES, LABEL_COL

DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "is_unbalance": True,
    "random_state": 42,
}


def train_baseline(
    train_df: pd.DataFrame,
    features: tuple[str, ...] = KAMEI_FEATURES,
    label: str = LABEL_COL,
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    params = {**DEFAULT_PARAMS, **(params or {})}
    model = lgb.LGBMClassifier(**params)
    model.fit(train_df[list(features)], train_df[label])
    return model


def save_model(
    model: lgb.LGBMClassifier,
    path: str | Path = "models/baseline_lgbm.txt",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))
    return path


if __name__ == "__main__":
    from bugit.data.load import load_train_test

    train_df, _ = load_train_test()
    model = train_baseline(train_df)
    saved_to = save_model(model)
    print(f"Model saved: {saved_to}")