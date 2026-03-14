"""Genera visuales PCA 3D para el escenario sin GPA.

Produce:
- una vista 3D con todas las clases;
- una vista 3D centrada en clase 0 vs clase 1.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "Student_performance_data_clean.csv"
TARGET = "GradeClass"


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_and_build() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATASET)
    df[TARGET] = df[TARGET].astype(int)
    df["StudyTimeWeekly"] = df["StudyTimeWeekly"].astype(float)

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
    y = df[TARGET].to_numpy()
    return x, y


def project_pca_3d(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pca = PCA(n_components=3, random_state=SEED)
    coords = pca.fit_transform(x_scaled)
    return coords, pca.explained_variance_ratio_


def plot_all_classes(coords: np.ndarray, y: np.ndarray, variance: np.ndarray, out_path: Path) -> None:
    colors = {0: "#e63946", 1: "#f4a261", 2: "#2a9d8f", 3: "#457b9d", 4: "#adb5bd"}
    labels = {
        0: "Clase 0 (desempeno mas alto)",
        1: "Clase 1",
        2: "Clase 2",
        3: "Clase 3",
        4: "Clase 4 (desempeno mas bajo)",
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cls in [4, 3, 2, 1, 0]:
        mask = y == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            c=colors[cls],
            label=labels[cls],
            alpha=0.5 if cls != 0 else 0.8,
            s=12 if cls != 0 else 28,
            edgecolors="none",
        )

    ax.set_title("PCA 3D - Todas las clases (sin GPA)")
    ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({variance[2]*100:.1f}%)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_0vs1(coords: np.ndarray, y: np.ndarray, variance: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    specs = [
        (1, "#f4a261", "Clase 1"),
        (0, "#e63946", "Clase 0"),
    ]
    for cls, color, label in specs:
        mask = y == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            c=color,
            label=label,
            alpha=0.75,
            s=28 if cls == 1 else 36,
            edgecolors="black",
            linewidths=0.25,
        )

    ax.set_title("PCA 3D - Clase 0 vs Clase 1 (sin GPA)")
    ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({variance[2]*100:.1f}%)")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    set_seed()
    x, y = load_and_build()
    coords, variance = project_pca_3d(x)

    all_path = BASE_DIR / "separabilidad_pca_3d_todas.png"
    focus_path = BASE_DIR / "separabilidad_pca_3d_0vs1.png"
    plot_all_classes(coords, y, variance, all_path)
    plot_0vs1(coords, y, variance, focus_path)

    print("Archivos generados:")
    print(all_path)
    print(focus_path)


if __name__ == "__main__":
    main()
