"""Corrida final para el reporte IEEE.

Compara:
- MLP principal con la arquitectura base del examen usando GPA.
- Random Forest comparativo usando GPA.

Exporta tablas, reportes de clasificacion, matrices de confusion e
importancia de variables para nutrir el reporte final.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "Student_performance_data_clean.csv"
TARGET = "GradeClass"

CONTINUOUS = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
CATEGORICAL = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]

MLP_GRID = {
    "optimizer": ["adam", "rmsprop", "sgd"],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "epochs": [50, 100, 150],
}

RF_GRID = {
    "n_estimators": [200, 400, 700],
    "max_depth": [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced", "balanced_subsample"],
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


def build_optimizer(name: str, learning_rate: float):
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)


def build_exam_mlp(input_dim: int, num_classes: int, optimizer_name: str, learning_rate: float) -> keras.Sequential:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(200, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(150, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(200, activation="relu"),
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


def prepare_data(df: pd.DataFrame):
    feature_columns = CONTINUOUS + CATEGORICAL
    x_df = df[feature_columns].copy()
    y = df[TARGET].astype(int).to_numpy()

    x_train_df, x_temp_df, y_train, y_temp = train_test_split(
        x_df, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val_df, x_test_df, y_val, y_test = train_test_split(
        x_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    preprocessor = ColumnTransformer(
        [
            ("continuous", StandardScaler(), CONTINUOUS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )

    x_train_mlp = np.asarray(preprocessor.fit_transform(x_train_df), dtype=np.float32)
    x_val_mlp = np.asarray(preprocessor.transform(x_val_df), dtype=np.float32)
    x_test_mlp = np.asarray(preprocessor.transform(x_test_df), dtype=np.float32)

    x_train_rf = x_train_df.copy()
    x_val_rf = x_val_df.copy()
    x_test_rf = x_test_df.copy()
    for frame in (x_train_rf, x_val_rf, x_test_rf):
        for col in CATEGORICAL:
            frame[col] = frame[col].astype(float)
        for col in CONTINUOUS:
            frame[col] = frame[col].astype(float)

    return (
        x_train_mlp,
        x_val_mlp,
        x_test_mlp,
        x_train_rf,
        x_val_rf,
        x_test_rf,
        y_train.astype(np.int32),
        y_val.astype(np.int32),
        y_test.astype(np.int32),
        preprocessor,
    )


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int], title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def tune_mlp(x_train, x_val, y_train, y_val, num_classes: int):
    rows: List[Dict[str, object]] = []
    best_model = None
    best_cfg = None
    best_score = -np.inf
    best_loss = np.inf

    for cfg in ParameterGrid(MLP_GRID):
        set_seed()
        model = build_exam_mlp(x_train.shape[1], num_classes, cfg["optimizer"], cfg["learning_rate"])
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]
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
        metrics_val = collect_metrics(y_val, pred_val)
        val_loss = float(np.min(history.history["val_loss"]))
        rows.append(
            {
                **cfg,
                "epochs_ran": len(history.history["loss"]),
                "val_accuracy": round(metrics_val["accuracy"], 4),
                "val_f1_macro": round(metrics_val["f1_macro"], 4),
                "val_recall_macro": round(metrics_val["recall_macro"], 4),
                "val_loss": round(val_loss, 4),
            }
        )
        if metrics_val["f1_macro"] > best_score or (
            np.isclose(metrics_val["f1_macro"], best_score) and val_loss < best_loss
        ):
            best_score = metrics_val["f1_macro"]
            best_loss = val_loss
            best_model = model
            best_cfg = dict(cfg)

    return best_model, best_cfg, pd.DataFrame(rows).sort_values(["val_f1_macro", "val_accuracy"], ascending=[False, False])


def tune_random_forest(x_train, x_val, y_train, y_val):
    rows: List[Dict[str, object]] = []
    best_model = None
    best_cfg = None
    best_score = -np.inf

    for cfg in ParameterGrid(RF_GRID):
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **cfg)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        metrics_val = collect_metrics(y_val, pred_val)
        rows.append(
            {
                **cfg,
                "val_accuracy": round(metrics_val["accuracy"], 4),
                "val_f1_macro": round(metrics_val["f1_macro"], 4),
                "val_recall_macro": round(metrics_val["recall_macro"], 4),
            }
        )
        if metrics_val["f1_macro"] > best_score:
            best_score = metrics_val["f1_macro"]
            best_model = model
            best_cfg = dict(cfg)

    return best_model, best_cfg, pd.DataFrame(rows).sort_values(["val_f1_macro", "val_accuracy"], ascending=[False, False])


def classification_report_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})


def main() -> None:
    set_seed()
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    df["GPA"] = df["GPA"].astype(float)

    (
        x_train_mlp,
        x_val_mlp,
        x_test_mlp,
        x_train_rf,
        x_val_rf,
        x_test_rf,
        y_train,
        y_val,
        y_test,
        _preprocessor,
    ) = prepare_data(df)
    labels = sorted(np.unique(y_test).tolist())
    num_classes = df[TARGET].nunique()

    mlp_model, mlp_cfg, mlp_search = tune_mlp(x_train_mlp, x_val_mlp, y_train, y_val, num_classes)
    rf_model, rf_cfg, rf_search = tune_random_forest(x_train_rf, x_val_rf, y_train, y_val)

    mlp_pred = np.argmax(mlp_model.predict(x_test_mlp, verbose=0), axis=1)
    rf_pred = rf_model.predict(x_test_rf)

    mlp_metrics = collect_metrics(y_test, mlp_pred)
    rf_metrics = collect_metrics(y_test, rf_pred)

    comparison_df = pd.DataFrame(
        [
            {"model": "MLP Principal", "best_config": mlp_cfg, **{k: round(v, 4) for k, v in mlp_metrics.items()}},
            {"model": "Random Forest", "best_config": rf_cfg, **{k: round(v, 4) for k, v in rf_metrics.items()}},
        ]
    )

    importance_df = (
        pd.DataFrame({"variable": x_train_rf.columns, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    mlp_search.to_csv(BASE_DIR / "hiperparametros_mlp_final.csv", index=False)
    rf_search.to_csv(BASE_DIR / "hiperparametros_random_forest_final.csv", index=False)
    classification_report_df(y_test, mlp_pred).to_csv(BASE_DIR / "classification_report_mlp_final.csv", index=False)
    classification_report_df(y_test, rf_pred).to_csv(BASE_DIR / "classification_report_random_forest_final.csv", index=False)
    comparison_df.to_csv(BASE_DIR / "comparacion_modelos_final.csv", index=False)
    importance_df.to_csv(BASE_DIR / "importancia_variables_random_forest_final.csv", index=False)

    save_confusion_matrix(y_test, mlp_pred, labels, "Matriz de confusion normalizada - MLP final", BASE_DIR / "confusion_matrix_mlp_final.png")
    save_confusion_matrix(y_test, rf_pred, labels, "Matriz de confusion normalizada - Random Forest final", BASE_DIR / "confusion_matrix_random_forest_final.png")

    print("\n=== Mejor configuracion MLP ===")
    print(mlp_cfg)
    print(mlp_search.head(12).to_string(index=False))
    print("\n=== Mejor configuracion Random Forest ===")
    print(rf_cfg)
    print(rf_search.head(12).to_string(index=False))
    print("\n=== Resultados finales en prueba ===")
    print(comparison_df.to_string(index=False))
    print("\nReporte de clasificacion MLP:")
    print(classification_report(y_test, mlp_pred, zero_division=0))
    print("\nReporte de clasificacion Random Forest:")
    print(classification_report(y_test, rf_pred, zero_division=0))
    print("\nArchivos generados:")
    for file_name in [
        "comparacion_modelos_final.csv",
        "hiperparametros_mlp_final.csv",
        "hiperparametros_random_forest_final.csv",
        "classification_report_mlp_final.csv",
        "classification_report_random_forest_final.csv",
        "confusion_matrix_mlp_final.png",
        "confusion_matrix_random_forest_final.png",
        "importancia_variables_random_forest_final.csv",
    ]:
        print(BASE_DIR / file_name)


if __name__ == "__main__":
    main()
