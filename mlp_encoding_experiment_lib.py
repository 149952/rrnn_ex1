"""Utilidades para comparar codificaciones categoricas en el MLP principal."""

from __future__ import annotations

import hashlib
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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
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
ORDERED = ["ParentalEducation", "ParentalSupport"]

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


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    df["GPA"] = df["GPA"].astype(float)
    return df


def split_df(df: pd.DataFrame):
    x_df = df[CONTINUOUS + CATEGORICAL].copy()
    y = df[TARGET].astype(int).to_numpy()
    x_train_df, x_temp_df, y_train, y_temp = train_test_split(
        x_df, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val_df, x_test_df, y_val, y_test = train_test_split(
        x_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return x_train_df.reset_index(drop=True), x_val_df.reset_index(drop=True), x_test_df.reset_index(drop=True), y_train.astype(np.int32), y_val.astype(np.int32), y_test.astype(np.int32)


def stable_bucket(token: str, n_buckets: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def _series_mean_map(values: pd.Series, targets: np.ndarray) -> pd.Series:
    tmp = pd.DataFrame({"value": values, "target": targets})
    return tmp.groupby("value")["target"].mean()


def target_encode_column(
    train_col: pd.Series,
    y_train: np.ndarray,
    val_col: pd.Series,
    test_col: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_mean = float(np.mean(y_train))
    train_enc = np.zeros(len(train_col), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_idx, hold_idx in skf.split(train_col, y_train):
        means = _series_mean_map(train_col.iloc[fit_idx], y_train[fit_idx])
        train_enc[hold_idx] = train_col.iloc[hold_idx].map(means).fillna(global_mean).to_numpy()
    full_means = _series_mean_map(train_col, y_train)
    val_enc = val_col.map(full_means).fillna(global_mean).to_numpy(dtype=float)
    test_enc = test_col.map(full_means).fillna(global_mean).to_numpy(dtype=float)
    return train_enc, val_enc, test_enc


def loo_encode_column(
    train_col: pd.Series,
    y_train: np.ndarray,
    val_col: pd.Series,
    test_col: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_mean = float(np.mean(y_train))
    tmp = pd.DataFrame({"value": train_col, "target": y_train})
    sums = tmp.groupby("value")["target"].sum()
    counts = tmp.groupby("value")["target"].count()

    sum_vals = train_col.map(sums).to_numpy(dtype=float)
    count_vals = train_col.map(counts).to_numpy(dtype=float)
    train_enc = np.where(
        count_vals > 1,
        (sum_vals - y_train) / (count_vals - 1.0),
        global_mean,
    )

    means = sums / counts
    val_enc = val_col.map(means).fillna(global_mean).to_numpy(dtype=float)
    test_enc = test_col.map(means).fillna(global_mean).to_numpy(dtype=float)
    return train_enc, val_enc, test_enc


def frequency_encode_column(
    train_col: pd.Series,
    val_col: pd.Series,
    test_col: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = train_col.value_counts(normalize=True)
    train_enc = train_col.map(freqs).fillna(0.0).to_numpy(dtype=float)
    val_enc = val_col.map(freqs).fillna(0.0).to_numpy(dtype=float)
    test_enc = test_col.map(freqs).fillna(0.0).to_numpy(dtype=float)
    return train_enc, val_enc, test_enc


def binary_encode_column(
    train_col: pd.Series,
    val_col: pd.Series,
    test_col: pd.Series,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_value = int(max(train_col.max(), val_col.max(), test_col.max()))
    n_bits = max(1, int(np.ceil(np.log2(max_value + 1 if max_value > 0 else 2))))

    def to_bits(series: pd.Series) -> pd.DataFrame:
        arr = np.zeros((len(series), n_bits), dtype=float)
        values = series.to_numpy(dtype=int)
        for bit in range(n_bits):
            arr[:, bit] = (values >> bit) & 1
        cols = [f"{prefix}_bit{bit}" for bit in range(n_bits)]
        return pd.DataFrame(arr, columns=cols)

    return to_bits(train_col), to_bits(val_col), to_bits(test_col)


def hash_encode_column(
    train_col: pd.Series,
    val_col: pd.Series,
    test_col: pd.Series,
    prefix: str,
    n_buckets: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [f"{prefix}_hash{i}" for i in range(n_buckets)]

    def to_hash_df(series: pd.Series) -> pd.DataFrame:
        arr = np.zeros((len(series), n_buckets), dtype=float)
        for idx, value in enumerate(series.astype(str)):
            bucket = stable_bucket(f"{prefix}={value}", n_buckets)
            arr[idx, bucket] = 1.0
        return pd.DataFrame(arr, columns=cols)

    return to_hash_df(train_col), to_hash_df(val_col), to_hash_df(test_col)


def encode_frames(
    x_train_df: pd.DataFrame,
    x_val_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
    y_train: np.ndarray,
    encoding: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_parts = [x_train_df[CONTINUOUS].reset_index(drop=True)]
    val_parts = [x_val_df[CONTINUOUS].reset_index(drop=True)]
    test_parts = [x_test_df[CONTINUOUS].reset_index(drop=True)]

    if encoding == "label":
        train_parts.append(x_train_df[CATEGORICAL].reset_index(drop=True))
        val_parts.append(x_val_df[CATEGORICAL].reset_index(drop=True))
        test_parts.append(x_test_df[CATEGORICAL].reset_index(drop=True))

    elif encoding == "ordinal":
        train_ord = x_train_df[CATEGORICAL].copy()
        val_ord = x_val_df[CATEGORICAL].copy()
        test_ord = x_test_df[CATEGORICAL].copy()
        for col in ORDERED:
            train_ord[col] = train_ord[col].astype(float)
            val_ord[col] = val_ord[col].astype(float)
            test_ord[col] = test_ord[col].astype(float)
        train_parts.append(train_ord.reset_index(drop=True))
        val_parts.append(val_ord.reset_index(drop=True))
        test_parts.append(test_ord.reset_index(drop=True))

    elif encoding == "target":
        for col in CATEGORICAL:
            tr, va, te = target_encode_column(x_train_df[col], y_train, x_val_df[col], x_test_df[col])
            train_parts.append(pd.DataFrame({f"{col}_target": tr}))
            val_parts.append(pd.DataFrame({f"{col}_target": va}))
            test_parts.append(pd.DataFrame({f"{col}_target": te}))

    elif encoding == "leave_one_out":
        for col in CATEGORICAL:
            tr, va, te = loo_encode_column(x_train_df[col], y_train, x_val_df[col], x_test_df[col])
            train_parts.append(pd.DataFrame({f"{col}_loo": tr}))
            val_parts.append(pd.DataFrame({f"{col}_loo": va}))
            test_parts.append(pd.DataFrame({f"{col}_loo": te}))

    elif encoding == "frequency":
        for col in CATEGORICAL:
            tr, va, te = frequency_encode_column(x_train_df[col], x_val_df[col], x_test_df[col])
            train_parts.append(pd.DataFrame({f"{col}_freq": tr}))
            val_parts.append(pd.DataFrame({f"{col}_freq": va}))
            test_parts.append(pd.DataFrame({f"{col}_freq": te}))

    elif encoding == "binary":
        for col in CATEGORICAL:
            tr, va, te = binary_encode_column(x_train_df[col], x_val_df[col], x_test_df[col], col)
            train_parts.append(tr.reset_index(drop=True))
            val_parts.append(va.reset_index(drop=True))
            test_parts.append(te.reset_index(drop=True))

    elif encoding == "hashing":
        for col in CATEGORICAL:
            tr, va, te = hash_encode_column(x_train_df[col], x_val_df[col], x_test_df[col], col, n_buckets=4)
            train_parts.append(tr.reset_index(drop=True))
            val_parts.append(va.reset_index(drop=True))
            test_parts.append(te.reset_index(drop=True))

    else:
        raise ValueError(f"Encoding desconocido: {encoding}")

    x_train = pd.concat(train_parts, axis=1)
    x_val = pd.concat(val_parts, axis=1)
    x_test = pd.concat(test_parts, axis=1)

    scaler = StandardScaler()
    x_train_arr = scaler.fit_transform(x_train)
    x_val_arr = scaler.transform(x_val)
    x_test_arr = scaler.transform(x_test)
    return (
        np.asarray(x_train_arr, dtype=np.float32),
        np.asarray(x_val_arr, dtype=np.float32),
        np.asarray(x_test_arr, dtype=np.float32),
    )


def tune_mlp(x_train, x_val, y_train, y_val, num_classes: int):
    rows: List[Dict[str, object]] = []
    best_model = None
    best_cfg = None
    best_score = -np.inf
    best_loss = np.inf

    for cfg in ParameterGrid(MLP_GRID):
        set_seed()
        model = build_exam_mlp(x_train.shape[1], num_classes, cfg["optimizer"], cfg["learning_rate"])
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


def run_experiment(encoding: str, label: str) -> None:
    set_seed()
    df = load_df()
    x_train_df, x_val_df, x_test_df, y_train, y_val, y_test = split_df(df)
    x_train, x_val, x_test = encode_frames(x_train_df, x_val_df, x_test_df, y_train, encoding)
    model, cfg, tuning_df = tune_mlp(x_train, x_val, y_train, y_val, int(df[TARGET].nunique()))

    pred_test = np.argmax(model.predict(x_test, verbose=0), axis=1)
    metrics = collect_metrics(y_test, pred_test)

    print(f"Dataset: {len(df)} registros")
    print(f"Codificacion evaluada: {label}")
    print(f"Dimensionalidad final de entrada: {x_train.shape[1]}")
    print(f"Distribucion de clases:\n{pd.Series(df[TARGET]).value_counts().sort_index()}\n")

    print(f"=== Mejor configuracion MLP con {label} ===")
    print(cfg)
    print(tuning_df.head(12).to_string(index=False))

    print("\n=== Resultados finales en prueba ===")
    print(
        pd.DataFrame(
            [{**{k: round(v, 4) for k, v in metrics.items()}, "model": f"MLP + {label}"}]
        ).to_string(index=False)
    )

    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred_test, zero_division=0))

    stem = encoding.lower()
    tuning_path = BASE_DIR / f"tuning_mlp_{stem}.csv"
    compare_path = BASE_DIR / f"comparacion_mlp_{stem}.csv"
    tuning_df.to_csv(tuning_path, index=False)
    pd.DataFrame(
        [{"model": f"MLP + {label}", **{k: round(v, 4) for k, v in metrics.items()}, "best_config": str(cfg)}]
    ).to_csv(compare_path, index=False)

    print("\nArchivos generados:")
    print(tuning_path)
    print(compare_path)
