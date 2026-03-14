"""MLP sobre el escenario de ingenieria de variables sin GPA.

Objetivo:
- verificar si la ingenieria de variables tambien puede beneficiar a un MLP;
- comparar contra el mejor resultado previo de Random Forest con ingenieria_v1.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "Student_performance_data_clean.csv"
TARGET = "GradeClass"

MLP_GRID = {
    "optimizer": ["adam", "rmsprop", "sgd"],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "epochs": [50, 100, 150],
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    return df


def build_ingenieria_v1(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Age",
        "Gender",
        "Ethnicity",
        "ParentalEducation",
        "StudyTimeWeekly",
        "Absences",
        "Tutoring",
        "ParentalSupport",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering",
    ]
    x = df[base_cols].copy()
    x["StudyTimePerAbsence"] = x["StudyTimeWeekly"] / (x["Absences"] + 1.0)
    x["SupportTutoringInteraction"] = x["ParentalSupport"] * x["Tutoring"]
    x["AcademicEngagementScore"] = (
        x["StudyTimeWeekly"]
        + 2.0 * x["Tutoring"]
        + x["ParentalSupport"]
        - 0.6 * x["Absences"]
    )
    x["ActivityCount"] = (
        x["Extracurricular"] + x["Sports"] + x["Music"] + x["Volunteering"]
    )
    x["SupportActivityInteraction"] = x["ParentalSupport"] * (x["ActivityCount"] + 1.0)
    return x


def split_data(x: pd.DataFrame, y: np.ndarray):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(x_val, dtype=np.float32),
        np.asarray(x_test, dtype=np.float32),
        y_train.astype(np.int32),
        y_val.astype(np.int32),
        y_test.astype(np.int32),
    )


def build_optimizer(name: str, learning_rate: float):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)


def build_mlp(input_dim: int, num_classes: int, optimizer_name: str, learning_rate: float) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=build_optimizer(optimizer_name, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_mlp(x_train, x_val, y_train, y_val, num_classes: int):
    rows: List[Dict[str, object]] = []
    best_model = None
    best_cfg = None
    best_score = -np.inf
    best_loss = np.inf

    for cfg in ParameterGrid(MLP_GRID):
        set_seed()
        model = build_mlp(x_train.shape[1], num_classes, cfg["optimizer"], cfg["learning_rate"])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )
        pred_val = np.argmax(model.predict(x_val, verbose=0), axis=1)
        metrics = collect_metrics(y_val, pred_val)
        val_loss = float(np.min(history.history["val_loss"]))
        rows.append(
            {
                **cfg,
                "epochs_ran": len(history.history["loss"]),
                "val_accuracy": round(metrics["accuracy"], 4),
                "val_f1_macro": round(metrics["f1_macro"], 4),
                "val_recall_macro": round(metrics["recall_macro"], 4),
                "val_loss": round(val_loss, 4),
            }
        )
        if metrics["f1_macro"] > best_score or (
            np.isclose(metrics["f1_macro"], best_score) and val_loss < best_loss
        ):
            best_score = metrics["f1_macro"]
            best_loss = val_loss
            best_model = model
            best_cfg = dict(cfg)

    return best_model, best_cfg, pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "val_accuracy"], ascending=[False, False]
    )


def main() -> None:
    set_seed()
    df = load_df()
    x = build_ingenieria_v1(df)
    y = df[TARGET].to_numpy()
    num_classes = int(df[TARGET].nunique())

    print(f"Dataset: {len(df)} registros")
    print(f"Features ingenieria_v1: {x.shape[1]}")
    print(f"Distribucion de clases:\n{pd.Series(y).value_counts().sort_index()}\n")

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)
    model, cfg, tuning_df = tune_mlp(x_train, x_val, y_train, y_val, num_classes)

    pred_test = np.argmax(model.predict(x_test, verbose=0), axis=1)
    metrics = collect_metrics(y_test, pred_test)

    print("=== Mejor configuracion MLP sobre ingenieria_v1 ===")
    print(cfg)
    print(tuning_df.head(12).to_string(index=False))

    print("\n=== Resultados finales en prueba ===")
    print(pd.DataFrame([{**{k: round(v, 4) for k, v in metrics.items()}, "model": "MLP + ingenieria_v1"}]).to_string(index=False))

    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred_test, zero_division=0))

    tuning_df.to_csv(BASE_DIR / "tuning_mlp_ingenieria_v1.csv", index=False)
    pd.DataFrame(
        [{"model": "MLP + ingenieria_v1", **{k: round(v, 4) for k, v in metrics.items()}, "best_config": str(cfg)}]
    ).to_csv(BASE_DIR / "comparacion_mlp_ingenieria_v1.csv", index=False)

    print("\nArchivos generados:")
    print(BASE_DIR / "tuning_mlp_ingenieria_v1.csv")
    print(BASE_DIR / "comparacion_mlp_ingenieria_v1.csv")


if __name__ == "__main__":
    main()
