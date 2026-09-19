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
from bugit.paths import FIGURES_DIR


def evaluate_baseline(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    features: tuple[str, ...] = KAMEI_FEATURES,
    label: str = LABEL_COL,
) -> dict:
    X_test = test_df[list(features)]
    y_test = test_df[label]
    y_proba = model.predict(X_test)
    y_pred = (y_proba >= 0.75).astype(int) #TODO 0.5

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
    }

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"AUC-PR (average precision): {metrics['average_precision']:.3f}")

    return metrics


def plot_calibration(
    model: lgb.Booster,
    test_df: pd.DataFrame,
    features: tuple[str, ...] = KAMEI_FEATURES,
    label: str = LABEL_COL,
    out_path: str | Path = FIGURES_DIR / "calibration.png",
) -> Path:
    X_test = test_df[list(features)]
    y_test = test_df[label]
    y_proba = model.predict(X_test)

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
    from bugit.data.load import load_apachejit_csv
    from bugit.models.train import load_model
    from bugit.paths import RAW_DATA_DIR

    test_df = load_apachejit_csv(RAW_DATA_DIR / "apachejit_test_large.csv")
    model = load_model()
    evaluate_baseline(model, test_df)
    fig_path = plot_calibration(model, test_df)
    print(f"Calibration curve: {fig_path}")