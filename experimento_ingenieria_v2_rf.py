"""Random Forest sobre una ingenieria de variables extendida sin GPA.

Objetivo:
- ampliar ingenieria_v1 con atributos derivados adicionales;
- verificar si una representacion mas rica mejora al mejor RF previo.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, train_test_split


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "Student_performance_data_clean.csv"
TARGET = "GradeClass"

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


def build_ingenieria_v2(df: pd.DataFrame) -> pd.DataFrame:
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

    # Atributos adicionales con sentido conductual y de intensidad.
    x["AbsenceRateLog"] = np.log1p(x["Absences"])
    x["StudyTimeSquared"] = np.square(x["StudyTimeWeekly"])
    x["AbsencesSquared"] = np.square(x["Absences"])
    x["TutoringPerSupport"] = x["Tutoring"] / (x["ParentalSupport"] + 1.0)
    x["StudySupportProduct"] = x["StudyTimeWeekly"] * (x["ParentalSupport"] + 1.0)
    x["ActivityWeightedSupport"] = x["ActivityCount"] * (x["ParentalSupport"] + 1.0)
    x["EngagementPerActivity"] = x["AcademicEngagementScore"] / (x["ActivityCount"] + 1.0)
    x["AbsencePressure"] = x["Absences"] / (x["StudyTimeWeekly"] + 1.0)
    x["SupportMinusAbsences"] = x["ParentalSupport"] - 0.3 * x["Absences"]
    x["TutoringAbsenceInteraction"] = x["Tutoring"] * x["Absences"]
    x["StudyActivityBalance"] = x["StudyTimeWeekly"] / (x["ActivityCount"] + 1.0)
    x["CommitmentComposite"] = (
        0.5 * x["AcademicEngagementScore"]
        + 0.3 * x["SupportActivityInteraction"]
        - 0.2 * x["Absences"]
    )
    return x


def split_data(x: pd.DataFrame, y: np.ndarray):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def tune_rf(x_train, y_train, x_val, y_val):
    best_model = None
    best_cfg = None
    best_score = -np.inf
    rows: List[Dict[str, object]] = []

    for cfg in ParameterGrid(RF_GRID):
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **cfg)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        metrics = collect_metrics(y_val, pred_val)
        rows.append({**cfg, **{f"val_{k}": round(v, 4) for k, v in metrics.items()}})
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = model
            best_cfg = dict(cfg)

    return best_model, best_cfg, pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "val_accuracy"], ascending=[False, False]
    )


def main() -> None:
    set_seed()
    df = load_df()
    x = build_ingenieria_v2(df)
    y = df[TARGET].to_numpy()

    print(f"Dataset: {len(df)} registros")
    print(f"Features ingenieria_v2_ext: {x.shape[1]}")
    print(f"Distribucion de clases:\n{pd.Series(y).value_counts().sort_index()}\n")

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)
    model, cfg, tuning_df = tune_rf(x_train, y_train, x_val, y_val)

    pred_test = model.predict(x_test)
    metrics = collect_metrics(y_test, pred_test)

    print("=== Mejor configuracion RF sobre ingenieria_v2_ext ===")
    print(cfg)
    print(tuning_df.head(12).to_string(index=False))

    print("\n=== Resultados finales en prueba ===")
    print(
        pd.DataFrame(
            [{**{k: round(v, 4) for k, v in metrics.items()}, "model": "RF + ingenieria_v2_ext"}]
        ).to_string(index=False)
    )

    print("\nReporte de clasificacion:")
    print(classification_report(y_test, pred_test, zero_division=0))

    importances = pd.DataFrame(
        {"feature": x.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    tuning_df.to_csv(BASE_DIR / "tuning_rf_ingenieria_v2_ext.csv", index=False)
    pd.DataFrame(
        [{"model": "RF + ingenieria_v2_ext", **{k: round(v, 4) for k, v in metrics.items()}, "best_config": str(cfg)}]
    ).to_csv(BASE_DIR / "comparacion_rf_ingenieria_v2_ext.csv", index=False)
    importances.to_csv(BASE_DIR / "importancia_rf_ingenieria_v2_ext.csv", index=False)

    print("\nTop 15 importancias:")
    print(importances.head(15).to_string(index=False))

    print("\nArchivos generados:")
    print(BASE_DIR / "tuning_rf_ingenieria_v2_ext.csv")
    print(BASE_DIR / "comparacion_rf_ingenieria_v2_ext.csv")
    print(BASE_DIR / "importancia_rf_ingenieria_v2_ext.csv")


if __name__ == "__main__":
    main()
