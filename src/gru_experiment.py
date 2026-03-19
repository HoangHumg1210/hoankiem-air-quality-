from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

from src.data_utils import (
    DataConfig,
    evaluate_by_horizon,
    evaluate_regression,
    inverse_y,
    prepare_dataset,
    set_seed,
)


@dataclass
class GRUExperimentConfig:
    gru_units: tuple[int, ...] = (128, 64)
    dense_units: Optional[int] = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    patience: int = 8
    min_lr: float = 1e-5
    step_to_plot: int = 1
    confidence_z: float = 1.96


def build_gru_model(
    input_shape: tuple[int, int],
    horizon: int,
    cfg: GRUExperimentConfig,
) -> Model:
    model = Sequential(name="gru_forecaster")
    model.add(Input(shape=input_shape))

    for idx, units in enumerate(cfg.gru_units):
        return_sequences = idx < len(cfg.gru_units) - 1
        model.add(GRU(units, return_sequences=return_sequences))
        if cfg.dropout > 0:
            model.add(Dropout(cfg.dropout))

    if cfg.dense_units is not None:
        model.add(Dense(cfg.dense_units, activation="relu"))
        if cfg.dropout > 0:
            model.add(Dropout(cfg.dropout))

    model.add(Dense(horizon, name="forecast"))
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_gru_model(
    artifacts: dict[str, Any],
    cfg: GRUExperimentConfig,
) -> tuple[Model, tf.keras.callbacks.History]:
    model = build_gru_model(
        input_shape=(artifacts["cfg"].lookback, artifacts["n_features"]),
        horizon=artifacts["cfg"].horizon,
        cfg=cfg,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, cfg.patience // 2),
            min_lr=cfg.min_lr,
            verbose=1,
        ),
    ]
    history = model.fit(
        artifacts["X_train_seq"],
        artifacts["y_train_seq"],
        validation_data=(artifacts["X_val_seq"], artifacts["y_val_seq"]),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def predict_split(
    model: Model,
    artifacts: dict[str, Any],
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    X_seq = artifacts[f"X_{split}_seq"]
    y_seq = artifacts[f"y_{split}_seq"]
    times = pd.DatetimeIndex(artifacts[f"{split}_times"])

    y_pred_scaled = model.predict(X_seq, verbose=0)
    y_pred = inverse_y(y_pred_scaled, artifacts["target_transformer"])
    y_true = inverse_y(y_seq, artifacts["target_transformer"])
    return y_true, y_pred, times


def build_forecast_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    times: pd.DatetimeIndex,
    step: int = 1,
    residual_std: Optional[float] = None,
    confidence_z: float = 1.96,
) -> pd.DataFrame:
    step_idx = step - 1
    frame = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(times),
            "actual": y_true[:, step_idx],
            "predicted": y_pred[:, step_idx],
        }
    )
    if residual_std is not None:
        half_width = confidence_z * residual_std
        frame["lower"] = np.clip(frame["predicted"] - half_width, 0, None)
        frame["upper"] = frame["predicted"] + half_width
    return frame


def evaluate_gru_predictions(
    model: Model,
    artifacts: dict[str, Any],
    split: str = "test",
    plot_step: int = 1,
    confidence_z: float = 1.96,
) -> dict[str, Any]:
    y_true, y_pred, times = predict_split(model, artifacts, split=split)
    metrics_all = evaluate_regression(y_true, y_pred, name=f"{split.upper()}(all horizons)")
    metrics_step1 = evaluate_regression(
        y_true[:, 0],
        y_pred[:, 0],
        name=f"{split.upper()}(step 1)",
    )
    by_horizon = evaluate_by_horizon(y_true, y_pred)

    residual_std = None
    if split == "test":
        y_true_val, y_pred_val, _ = predict_split(model, artifacts, split="val")
        residual_std = float(np.std(y_true_val[:, plot_step - 1] - y_pred_val[:, plot_step - 1]))

    forecast_frame = build_forecast_frame(
        y_true=y_true,
        y_pred=y_pred,
        times=times,
        step=plot_step,
        residual_std=residual_std,
        confidence_z=confidence_z,
    )

    metrics_frame = pd.DataFrame(
        [
            {
                "split": split,
                "scope": "all_horizons",
                **metrics_all,
            },
            {
                "split": split,
                "scope": f"step_{plot_step}",
                **metrics_step1,
            },
        ]
    )

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "times": times,
        "metrics_all": metrics_all,
        "metrics_step": metrics_step1,
        "metrics_frame": metrics_frame,
        "by_horizon": by_horizon,
        "forecast_frame": forecast_frame,
    }


def plot_training_history(history: tf.keras.callbacks.History) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    history_df = pd.DataFrame(history.history)

    history_df[["loss", "val_loss"]].plot(ax=axes[0], title="Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    history_df[["mae", "val_mae"]].plot(ax=axes[1], title="MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_forecast(
    forecast_frame: pd.DataFrame,
    title: str = "GRU forecast on test",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(forecast_frame["time"], forecast_frame["actual"], label="Actual (Test)", linewidth=2)
    ax.plot(forecast_frame["time"], forecast_frame["predicted"], label="GRU", linewidth=2)

    if {"lower", "upper"}.issubset(forecast_frame.columns):
        ax.fill_between(
            forecast_frame["time"],
            forecast_frame["lower"],
            forecast_frame["upper"],
            alpha=0.15,
            label="Approx. interval",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("PM2.5")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def run_gru_experiment(
    data_cfg: DataConfig,
    model_cfg: Optional[GRUExperimentConfig] = None,
) -> dict[str, Any]:
    if model_cfg is None:
        model_cfg = GRUExperimentConfig()

    set_seed(data_cfg.seed)
    artifacts = prepare_dataset(data_cfg)
    model, history = train_gru_model(artifacts, model_cfg)

    val_results = evaluate_gru_predictions(
        model,
        artifacts,
        split="val",
        plot_step=model_cfg.step_to_plot,
        confidence_z=model_cfg.confidence_z,
    )
    test_results = evaluate_gru_predictions(
        model,
        artifacts,
        split="test",
        plot_step=model_cfg.step_to_plot,
        confidence_z=model_cfg.confidence_z,
    )

    return {
        "artifacts": artifacts,
        "model": model,
        "history": history,
        "val_results": val_results,
        "test_results": test_results,
    }
