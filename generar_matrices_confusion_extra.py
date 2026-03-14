"""Genera variantes populares de matrices de confusion para los modelos finales.

Objetivo:
- producir varias visualizaciones para revision manual con el profesor;
- no modificar el reporte, solo exportar imagenes comparables.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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

MLP_CFG = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50,
}

RF_CFG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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
        for col in CATEGORICAL + CONTINUOUS:
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
    )


def fit_final_models(df: pd.DataFrame):
    (
        x_train_mlp,
        x_val_mlp,
        x_test_mlp,
        x_train_rf,
        _x_val_rf,
        x_test_rf,
        y_train,
        y_val,
        y_test,
    ) = prepare_data(df)

    x_train_full_mlp = np.concatenate([x_train_mlp, x_val_mlp], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    set_seed()
    mlp = build_exam_mlp(x_train_full_mlp.shape[1], int(df[TARGET].nunique()), MLP_CFG["optimizer"], MLP_CFG["learning_rate"])
    callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)]
    mlp.fit(
        x_train_full_mlp,
        y_train_full,
        epochs=MLP_CFG["epochs"],
        batch_size=MLP_CFG["batch_size"],
        verbose=0,
        callbacks=callbacks,
    )
    mlp_pred = np.argmax(mlp.predict(x_test_mlp, verbose=0), axis=1)

    x_train_full_rf = pd.concat([x_train_rf, _x_val_rf], axis=0)
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_CFG)
    rf.fit(x_train_full_rf, y_train_full)
    rf_pred = rf.predict(x_test_rf)

    return y_test, mlp_pred, rf_pred


def annotate_counts_and_percent(cm_counts: np.ndarray, cm_norm: np.ndarray) -> np.ndarray:
    annotations = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            annotations[i, j] = f"{cm_norm[i, j]:.2f}\n({cm_counts[i, j]})"
    return annotations


def save_heatmap(cm: np.ndarray, labels: list[int], title: str, path: Path, fmt: str = ".2f", annot=True, cmap="Blues", mask=None):
    plt.figure(figsize=(7.2, 5.6))
    sns.heatmap(cm, annot=annot, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, mask=mask, cbar=True)
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_counts_plus_percent(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], title: str, path: Path):
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    annot = annotate_counts_and_percent(cm_counts, cm_norm)
    plt.figure(figsize=(7.4, 5.8))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_error_only(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], title: str, path: Path):
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    mask = np.eye(len(labels), dtype=bool)
    plt.figure(figsize=(7.2, 5.6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Reds", xticklabels=labels, yticklabels=labels, mask=mask, cbar=True)
    plt.title(title)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_side_by_side(y_true: np.ndarray, pred_left: np.ndarray, pred_right: np.ndarray, labels: list[int], left_title: str, right_title: str, path: Path, normalize: str = "true"):
    cm_left = confusion_matrix(y_true, pred_left, labels=labels, normalize=normalize)
    cm_right = confusion_matrix(y_true, pred_right, labels=labels, normalize=normalize)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_left, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cm_right, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[0].set_title(left_title)
    axes[1].set_title(right_title)
    for ax in axes:
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_marginal_bars(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int], title: str, path: Path):
    cm_true = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_pred = confusion_matrix(y_true, y_pred, labels=labels, normalize="pred")
    recall_diag = np.diag(cm_true)
    precision_diag = np.diag(cm_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(labels, recall_diag, color="#457b9d")
    axes[0].set_title("Recall por clase")
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Clase")
    axes[0].set_ylabel("Valor")

    axes[1].bar(labels, precision_diag, color="#2a9d8f")
    axes[1].set_title("Precision por clase")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Clase")
    axes[1].set_ylabel("Valor")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    set_seed()
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    df["GPA"] = df["GPA"].astype(float)

    y_test, mlp_pred, rf_pred = fit_final_models(df)
    labels = sorted(np.unique(y_test).tolist())

    # Variantes para MLP
    save_heatmap(confusion_matrix(y_test, mlp_pred, labels=labels, normalize="true"), labels, "MLP - matriz normalizada por fila", BASE_DIR / "cm_mlp_normalizada_filas.png")
    save_heatmap(confusion_matrix(y_test, mlp_pred, labels=labels, normalize="pred"), labels, "MLP - matriz normalizada por columna", BASE_DIR / "cm_mlp_normalizada_columnas.png")
    save_heatmap(confusion_matrix(y_test, mlp_pred, labels=labels, normalize="all"), labels, "MLP - matriz normalizada global", BASE_DIR / "cm_mlp_normalizada_global.png")
    save_counts_plus_percent(y_test, mlp_pred, labels, "MLP - porcentajes y conteos", BASE_DIR / "cm_mlp_conteos_porcentajes.png")
    save_error_only(y_test, mlp_pred, labels, "MLP - errores fuera de la diagonal", BASE_DIR / "cm_mlp_solo_errores.png")
    save_marginal_bars(y_test, mlp_pred, labels, "MLP - precision y recall por clase", BASE_DIR / "cm_mlp_barras_precision_recall.png")

    # Variantes para RF
    save_heatmap(confusion_matrix(y_test, rf_pred, labels=labels, normalize="true"), labels, "Random Forest - matriz normalizada por fila", BASE_DIR / "cm_rf_normalizada_filas.png")
    save_heatmap(confusion_matrix(y_test, rf_pred, labels=labels, normalize="pred"), labels, "Random Forest - matriz normalizada por columna", BASE_DIR / "cm_rf_normalizada_columnas.png")
    save_heatmap(confusion_matrix(y_test, rf_pred, labels=labels, normalize="all"), labels, "Random Forest - matriz normalizada global", BASE_DIR / "cm_rf_normalizada_global.png")
    save_counts_plus_percent(y_test, rf_pred, labels, "Random Forest - porcentajes y conteos", BASE_DIR / "cm_rf_conteos_porcentajes.png")
    save_error_only(y_test, rf_pred, labels, "Random Forest - errores fuera de la diagonal", BASE_DIR / "cm_rf_solo_errores.png")
    save_marginal_bars(y_test, rf_pred, labels, "Random Forest - precision y recall por clase", BASE_DIR / "cm_rf_barras_precision_recall.png")

    # Comparativas
    save_side_by_side(
        y_test,
        mlp_pred,
        rf_pred,
        labels,
        "MLP - normalizada por fila",
        "Random Forest - normalizada por fila",
        BASE_DIR / "cm_comparada_normalizada_filas.png",
        normalize="true",
    )
    save_side_by_side(
        y_test,
        mlp_pred,
        rf_pred,
        labels,
        "MLP - normalizada por columna",
        "Random Forest - normalizada por columna",
        BASE_DIR / "cm_comparada_normalizada_columnas.png",
        normalize="pred",
    )

    print("Archivos generados:")
    for name in [
        "cm_mlp_normalizada_filas.png",
        "cm_mlp_normalizada_columnas.png",
        "cm_mlp_normalizada_global.png",
        "cm_mlp_conteos_porcentajes.png",
        "cm_mlp_solo_errores.png",
        "cm_mlp_barras_precision_recall.png",
        "cm_rf_normalizada_filas.png",
        "cm_rf_normalizada_columnas.png",
        "cm_rf_normalizada_global.png",
        "cm_rf_conteos_porcentajes.png",
        "cm_rf_solo_errores.png",
        "cm_rf_barras_precision_recall.png",
        "cm_comparada_normalizada_filas.png",
        "cm_comparada_normalizada_columnas.png",
    ]:
        print(BASE_DIR / name)


if __name__ == "__main__":
    main()
