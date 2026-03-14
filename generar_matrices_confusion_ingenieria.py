"""Genera matrices de confusion normalizadas por columna para el escenario sin GPA.

Compara:
- conjunto base sin GPA
- ingenieria_v1 sin GPA
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "Student_performance_data_clean.csv"
TARGET = "GradeClass"

RF_BASE_CFG = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": None,
}

RF_ING_CFG = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "class_weight": None,
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_df() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)
    return df


def build_base(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
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
    ].copy()


def build_ingenieria_v1(df: pd.DataFrame) -> pd.DataFrame:
    x = build_base(df)
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
    return x_train, x_val, x_test, y_train, y_val, y_test


def save_side_by_side(y_true: np.ndarray, pred_left: np.ndarray, pred_right: np.ndarray, labels: list[int], path: Path):
    cm_left = confusion_matrix(y_true, pred_left, labels=labels, normalize="pred")
    cm_right = confusion_matrix(y_true, pred_right, labels=labels, normalize="pred")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_left, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    sns.heatmap(cm_right, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[0].set_title("Base sin GPA")
    axes[1].set_title("Ingenieria de variables")
    for ax in axes:
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    set_seed()
    df = load_df()
    y = df[TARGET].to_numpy()
    labels = sorted(np.unique(y).tolist())

    x_base = build_base(df)
    xb_train, xb_val, xb_test, y_train, y_val, y_test = split_data(x_base, y)
    x_ing = build_ingenieria_v1(df)
    xi_train, xi_val, xi_test, _, _, _ = split_data(x_ing, y)

    rf_base = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_BASE_CFG)
    rf_base.fit(pd.concat([xb_train, xb_val]), np.concatenate([y_train, y_val]))
    pred_base = rf_base.predict(xb_test)

    rf_ing = RandomForestClassifier(random_state=SEED, n_jobs=-1, **RF_ING_CFG)
    rf_ing.fit(pd.concat([xi_train, xi_val]), np.concatenate([y_train, y_val]))
    pred_ing = rf_ing.predict(xi_test)

    save_side_by_side(
        y_test,
        pred_base,
        pred_ing,
        labels,
        BASE_DIR / "cm_comparada_ingenieria_normalizada_columnas.png",
    )

    print("Archivo generado:")
    print(BASE_DIR / "cm_comparada_ingenieria_normalizada_columnas.png")


if __name__ == "__main__":
    main()
