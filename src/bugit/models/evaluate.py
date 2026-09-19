from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from bugit.data.load import KAMEI_FEATURES, LABEL_COL


def evaluate_baseline(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame,
    features: tuple[str, ...] = KAMEI_FEATURES,
    label: str = LABEL_COL,
) -> dict:
    X_test = test_df[list(features)]
    y_test = test_df[label]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),  # AUC-PR
    }

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"AUC-PR (average precision): {metrics['average_precision']:.3f}")

    return metrics


def plot_calibration(
    model: lgb.LGBMClassifier,
    test_df: pd.DataFrame,
    features: tuple[str, ...] = KAMEI_FEATURES,
    label: str = LABEL_COL,
    out_path: str | Path = "reports/figures/calibration.png",
) -> Path:
    X_test = test_df[list(features)]
    y_test = test_df[label]
    y_proba = model.predict_proba(X_test)[:, 1]

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual for buggy")
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    return out_path


if __name__ == "__main__":
    from bugit.data.load import load_train_test
    from bugit.models.train import train_baseline

    train_df, test_df = load_train_test()
    model = train_baseline(train_df)
    evaluate_baseline(model, test_df)
    fig_path = plot_calibration(model, test_df)
    print(f"Calibration Curve: {fig_path}")