"""MLP con embeddings para variables categoricas en el escenario principal con GPA.

Objetivo:
- comparar una representacion con embeddings frente al MLP con one-hot encoding;
- mantener el mismo problema principal de clasificacion multiclase.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    df["GPA"] = df["GPA"].astype(float)
    return df


def split_data(df: pd.DataFrame):
    x_df = df[CONTINUOUS + CATEGORICAL].copy()
    y = df[TARGET].astype(int).to_numpy()

    x_train_df, x_temp_df, y_train, y_temp = train_test_split(
        x_df, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val_df, x_test_df, y_val, y_test = train_test_split(
        x_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler()
    x_train_cont = scaler.fit_transform(x_train_df[CONTINUOUS])
    x_val_cont = scaler.transform(x_val_df[CONTINUOUS])
    x_test_cont = scaler.transform(x_test_df[CONTINUOUS])

    x_train_cat = {col: x_train_df[col].astype("int32").to_numpy() for col in CATEGORICAL}
    x_val_cat = {col: x_val_df[col].astype("int32").to_numpy() for col in CATEGORICAL}
    x_test_cat = {col: x_test_df[col].astype("int32").to_numpy() for col in CATEGORICAL}

    cardinalities = {
        col: int(df[col].max()) + 1 for col in CATEGORICAL
    }

    train_inputs = {"continuous": np.asarray(x_train_cont, dtype=np.float32), **x_train_cat}
    val_inputs = {"continuous": np.asarray(x_val_cont, dtype=np.float32), **x_val_cat}
    test_inputs = {"continuous": np.asarray(x_test_cont, dtype=np.float32), **x_test_cat}

    return (
        train_inputs,
        val_inputs,
        test_inputs,
        y_train.astype(np.int32),
        y_val.astype(np.int32),
        y_test.astype(np.int32),
        cardinalities,
    )


def embedding_dim(cardinality: int) -> int:
    return min(8, max(2, (cardinality + 1) // 2))


def build_embedding_mlp(
    cardinalities: Dict[str, int],
    num_classes: int,
    optimizer_name: str,
    learning_rate: float,
) -> keras.Model:
    inputs = []

    cont_input = keras.Input(shape=(len(CONTINUOUS),), name="continuous")
    inputs.append(cont_input)
    encoded_parts = [cont_input]

    for col in CATEGORICAL:
        inp = keras.Input(shape=(1,), name=col, dtype="int32")
        inputs.append(inp)
        emb = keras.layers.Embedding(
            input_dim=cardinalities[col],
            output_dim=embedding_dim(cardinalities[col]),
            name=f"emb_{col}",
        )(inp)
        emb = keras.layers.Flatten()(emb)
        encoded_parts.append(emb)

    x = keras.layers.Concatenate()(encoded_parts)
    x = keras.layers.Dense(200, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(200, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(150, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(200, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=out, name="mlp_embeddings")
    model.compile(
        optimizer=build_optimizer(optimizer_name, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def tune_mlp(train_inputs, val_inputs, y_train, y_val, num_classes: int, cardinalities: Dict[str, int]):
    rows: List[Dict[str, object]] = []
    best_model = None
    best_cfg = None
    best_score = -np.inf
    best_loss = np.inf

    for cfg in ParameterGrid(MLP_GRID):
        set_seed()
        model = build_embedding_mlp(cardinalities, num_classes, cfg["optimizer"], cfg["learning_rate"])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            train_inputs,
            y_train,
            validation_data=(val_inputs, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )
        pred_val = np.argmax(model.predict(val_inputs, verbose=0), axis=1)
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
    train_inputs, val_inputs, test_inputs, y_train, y_val, y_test, cardinalities = split_data(df)
    num_classes = int(df[TARGET].nunique())

    print(f"Dataset: {len(df)} registros")
    print(f"Continuas: {len(CONTINUOUS)} | Categoricas con embedding: {len(CATEGORICAL)}")
    print(f"Distribucion de clases:\n{pd.Series(df[TARGET]).value_counts().sort_index()}\n")
    print("Cardinalidades categoricas:")
    for col in CATEGORICAL:
        print(f"  {col}: {cardinalities[col]} -> emb_dim={embedding_dim(cardinalities[col])}")

    model, cfg, tuning_df = tune_mlp(train_inputs, val_inputs, y_train, y_val, num_classes, cardinalities)
    pred_test = np.argmax(model.predict(test_inputs, verbose=0), axis=1)
    metrics = collect_metrics(y_test, pred_test)

    print("\n=== Mejor configuracion MLP con embeddings ===")
    print(cfg)
    print(tuning_df.head(12).to_string(index=False))

    print("\n=== Resultados finales en prueba ===")
    print(
        pd.DataFrame(
            [{**{k: round(v, 4) for k, v in metrics.items()}, "model": "MLP + embeddings"}]
        ).to_string(index=False)
    )

    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred_test, zero_division=0))

    tuning_df.to_csv(BASE_DIR / "tuning_mlp_embeddings.csv", index=False)
    pd.DataFrame(
        [{"model": "MLP + embeddings", **{k: round(v, 4) for k, v in metrics.items()}, "best_config": str(cfg)}]
    ).to_csv(BASE_DIR / "comparacion_mlp_embeddings.csv", index=False)

    print("\nArchivos generados:")
    print(BASE_DIR / "tuning_mlp_embeddings.csv")
    print(BASE_DIR / "comparacion_mlp_embeddings.csv")


if __name__ == "__main__":
    main()
