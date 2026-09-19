from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bugit.data.load import KAMEI_FEATURES, LABEL_COL, load_apachejit_csv, load_train_test
from bugit.paths import FIGURES_DIR, RAW_DATA_DIR

DEFAULT_LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "is_unbalance": False,
    "random_state": 42,
}

NAIVE_FEATURES: tuple[str, ...] = ("la", "ld")


def log_calibration_plot(y_true, y_proba, filename: str = "calibration.png") -> Path:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title("Calibration Curve")
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / filename
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def run_naive_baseline(
    experiment_name: str = "JIT_Defect_Prediction",
    run_name: str = "naive_logreg_la_ld",
) -> None:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        train_df, _ = load_train_test()
        test_df = load_apachejit_csv(RAW_DATA_DIR / "apachejit_test_large.csv")

        X_train = train_df[list(NAIVE_FEATURES)]
        y_train = train_df[LABEL_COL]

        X_test = test_df[list(NAIVE_FEATURES)]
        y_test = test_df[LABEL_COL]

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("num_features", len(NAIVE_FEATURES))
        mlflow.log_param("features", list(NAIVE_FEATURES))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ])

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auc_pr": average_precision_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        print(f"Run: {run_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        plot_path = log_calibration_plot(y_test, y_proba, filename="calibration_naive.png")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
        )


def run_lgbm_experiment(
    experiment_name: str = "JIT_Defect_Prediction",
    run_name: str = "lightgbm_baseline",
    params: dict | None = None,
) -> None:
    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        train_df, _ = load_train_test()
        test_df = load_apachejit_csv(RAW_DATA_DIR / "apachejit_test_large.csv")

        X_train = train_df[list(KAMEI_FEATURES)]
        y_train = train_df[LABEL_COL]

        X_test = test_df[list(KAMEI_FEATURES)]
        y_test = test_df[LABEL_COL]

        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_params(params)
        mlflow.log_param("num_features", len(KAMEI_FEATURES))
        mlflow.log_param("features", list(KAMEI_FEATURES))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auc_pr": average_precision_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        mlflow.log_metrics(metrics)

        print(f"Run: {run_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        plot_path = log_calibration_plot(y_test, y_proba, filename="calibration_lgbm.png")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        mlflow.lightgbm.log_model(
            lgb_model=model.booster_,
            name="model",
        )


if __name__ == "__main__":
    run_naive_baseline()