"""Genera una comparacion de matrices de confusion para MLP con one-hot vs embeddings.

Objetivo:
- respaldar visualmente la verificacion adicional sobre representacion categorica;
- producir una figura normalizada por columna, consistente con el resto del reporte.
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

MLP_ONEHOT_CFG = {
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 50,
}

MLP_EMB_CFG = {
    "optimizer": "sgd",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 150,
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


def build_onehot_mlp(input_dim: int, num_classes: int) -> keras.Sequential:
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
        optimizer=build_optimizer(MLP_ONEHOT_CFG["optimizer"], MLP_ONEHOT_CFG["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def embedding_dim(cardinality: int) -> int:
    return min(8, max(2, (cardinality + 1) // 2))


def build_embedding_mlp(cardinalities: dict[str, int], num_classes: int) -> keras.Model:
    inputs = []
    cont_input = keras.Input(shape=(len(CONTINUOUS),), name="continuous")
    inputs.append(cont_input)
    encoded = [cont_input]

    for col in CATEGORICAL:
        inp = keras.Input(shape=(1,), name=col, dtype="int32")
        inputs.append(inp)
        emb = keras.layers.Embedding(
            input_dim=cardinalities[col],
            output_dim=embedding_dim(cardinalities[col]),
            name=f"emb_{col}",
        )(inp)
        encoded.append(keras.layers.Flatten()(emb))

    x = keras.layers.Concatenate()(encoded)
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
        optimizer=build_optimizer(MLP_EMB_CFG["optimizer"], MLP_EMB_CFG["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_split():
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)

    x_df = df[CONTINUOUS + CATEGORICAL].copy()
    y = df[TARGET].to_numpy()

    x_train_df, x_temp_df, y_train, y_temp = train_test_split(
        x_df, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val_df, x_test_df, y_val, y_test = train_test_split(
        x_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return df, x_train_df, x_val_df, x_test_df, y_train.astype(np.int32), y_val.astype(np.int32), y_test.astype(np.int32)


def prepare_onehot(x_train_df, x_val_df, x_test_df):
    preprocessor = ColumnTransformer(
        [
            ("continuous", StandardScaler(), CONTINUOUS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )

    x_train = np.asarray(preprocessor.fit_transform(x_train_df), dtype=np.float32)
    x_val = np.asarray(preprocessor.transform(x_val_df), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(x_test_df), dtype=np.float32)
    return x_train, x_val, x_test


def prepare_embeddings(df, x_train_df, x_val_df, x_test_df):
    scaler = StandardScaler()
    train_cont = scaler.fit_transform(x_train_df[CONTINUOUS])
    val_cont = scaler.transform(x_val_df[CONTINUOUS])
    test_cont = scaler.transform(x_test_df[CONTINUOUS])

    cardinalities = {col: int(df[col].max()) + 1 for col in CATEGORICAL}
    train_inputs = {"continuous": np.asarray(train_cont, dtype=np.float32)}
    val_inputs = {"continuous": np.asarray(val_cont, dtype=np.float32)}
    test_inputs = {"continuous": np.asarray(test_cont, dtype=np.float32)}

    for col in CATEGORICAL:
        train_inputs[col] = x_train_df[col].astype("int32").to_numpy()
        val_inputs[col] = x_val_df[col].astype("int32").to_numpy()
        test_inputs[col] = x_test_df[col].astype("int32").to_numpy()

    return train_inputs, val_inputs, test_inputs, cardinalities


def fit_models():
    df, x_train_df, x_val_df, x_test_df, y_train, y_val, y_test = load_and_split()
    num_classes = int(df[TARGET].nunique())

    x_train_oh, x_val_oh, x_test_oh = prepare_onehot(x_train_df, x_val_df, x_test_df)
    train_emb, val_emb, test_emb, cardinalities = prepare_embeddings(df, x_train_df, x_val_df, x_test_df)

    x_train_full_oh = np.concatenate([x_train_oh, x_val_oh], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    set_seed()
    model_oh = build_onehot_mlp(x_train_full_oh.shape[1], num_classes)
    model_oh.fit(
        x_train_full_oh,
        y_train_full,
        epochs=MLP_ONEHOT_CFG["epochs"],
        batch_size=MLP_ONEHOT_CFG["batch_size"],
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)],
    )
    pred_oh = np.argmax(model_oh.predict(x_test_oh, verbose=0), axis=1)

    train_emb_full = {k: np.concatenate([train_emb[k], val_emb[k]], axis=0) for k in train_emb}
    set_seed()
    model_emb = build_embedding_mlp(cardinalities, num_classes)
    model_emb.fit(
        train_emb_full,
        y_train_full,
        epochs=MLP_EMB_CFG["epochs"],
        batch_size=MLP_EMB_CFG["batch_size"],
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)],
    )
    pred_emb = np.argmax(model_emb.predict(test_emb, verbose=0), axis=1)

    return y_test, pred_oh, pred_emb


def save_side_by_side(y_true, pred_onehot, pred_emb):
    labels = sorted(np.unique(y_true).tolist())
    cm_oh = confusion_matrix(y_true, pred_onehot, labels=labels, normalize="pred")
    cm_emb = confusion_matrix(y_true, pred_emb, labels=labels, normalize="pred")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    sns.heatmap(cm_oh, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cm_emb, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])

    axes[0].set_title("MLP con one-hot encoding")
    axes[1].set_title("MLP con embeddings")
    for ax in axes:
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")

    plt.tight_layout()
    out = BASE_DIR / "cm_mlp_onehot_vs_embeddings_normalizada_columnas.png"
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


def main():
    y_true, pred_onehot, pred_emb = fit_models()
    save_side_by_side(y_true, pred_onehot, pred_emb)


if __name__ == "__main__":
    main()
