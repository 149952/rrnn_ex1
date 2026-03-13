from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "reporte.ipynb"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text,
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }


nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

cells = []

cells.append(
    md(
        "# Examen 1\n\n"
        "## Reporte experimental"
    )
)

cells.append(
    md(
        "## 1. Librerías y configuración"
    )
)

cells.append(
    code(
        "from __future__ import annotations\n\n"
        "import os\n"
        "import random\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import seaborn as sns\n"
        "import tensorflow as tf\n"
        "from IPython.display import display\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score\n"
        "from sklearn.model_selection import ParameterGrid, train_test_split\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
        "from tensorflow import keras\n\n"
        "SEED = 42\n"
        "os.environ['PYTHONHASHSEED'] = str(SEED)\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "tf.random.set_seed(SEED)\n\n"
        "CANDIDATE_DIRS = [\n"
        "    Path.cwd(),\n"
        "    Path('/home/dgx/UABC/RRNN/desercion_escolar'),\n"
        "    Path('/Volumes/dgx/UABC/RRNN/desercion_escolar'),\n"
        "]\n"
        "BASE_DIR = next((p for p in CANDIDATE_DIRS if (p / 'Student_performance_data_clean.csv').exists()), Path.cwd())\n"
        "DATASET = BASE_DIR / 'Student_performance_data_clean.csv'\n"
        "TARGET = 'GradeClass'\n"
        "CONTINUOUS = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']\n"
        "CATEGORICAL = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']\n"
        "DISPLAY_NAMES = {\n"
        "    'StudentID': 'Matrícula',\n"
        "    'Age': 'edad',\n"
        "    'StudyTimeWeekly': 'tiempo de estudio semanal',\n"
        "    'Absences': 'ausencias',\n"
        "    'GPA': 'promedio general (GPA)',\n"
        "    'GradeClass': 'clase de calificación',\n"
        "    'Gender': 'género',\n"
        "    'Ethnicity': 'grupo étnico',\n"
        "    'ParentalEducation': 'educación parental',\n"
        "    'Tutoring': 'tutorías',\n"
        "    'ParentalSupport': 'apoyo parental',\n"
        "    'Extracurricular': 'actividades extracurriculares',\n"
        "    'Sports': 'deportes',\n"
        "    'Music': 'música',\n"
        "    'Volunteering': 'voluntariado',\n"
        "}\n\n"
        "MLP_GRID = {\n"
        "    'optimizer': ['adam', 'rmsprop', 'sgd'],\n"
        "    'learning_rate': [1e-4, 5e-4, 1e-3],\n"
        "    'batch_size': [32, 64, 128],\n"
        "    'epochs': [50, 100, 150],\n"
        "}\n\n"
        "RF_GRID = {\n"
        "    'n_estimators': [200, 400, 700],\n"
        "    'max_depth': [None, 10, 20, 30],\n"
        "    'min_samples_leaf': [1, 2, 4],\n"
        "    'class_weight': [None, 'balanced', 'balanced_subsample'],\n"
        "}\n\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.width', 160)\n"
        "sns.set_theme(style='whitegrid', palette='deep')\n"
        "if not DATASET.exists():\n"
        "    raise FileNotFoundError(f'No se encontró el conjunto de datos en: {DATASET}')"
    )
)

cells.append(md("## 2. Carga y revisión básica del conjunto de datos"))

cells.append(
    code(
        "df = pd.read_csv(DATASET)\n"
        "df[TARGET] = df[TARGET].astype(int)\n"
        "df['StudyTimeWeekly'] = df['StudyTimeWeekly'].astype(float)\n"
        "df['GPA'] = df['GPA'].astype(float)\n\n"
        "columnas_mostrar = {k: (v.capitalize() if k != 'StudentID' else v) for k, v in DISPLAY_NAMES.items()}\n"
        "print('Forma del dataset:', df.shape)\n"
        "display(df.head().rename(columns=columnas_mostrar))\n\n"
        "print('Tipos de datos:')\n"
        "display(df.dtypes.rename(index=columnas_mostrar).to_frame('tipo'))\n\n"
        "print('Valores nulos por columna:')\n"
        "display(df.isna().sum().rename(index=columnas_mostrar).to_frame('nulos'))\n\n"
        "print('Distribución de clases:')\n"
        "dist_clases = df[TARGET].value_counts().sort_index().to_frame('cantidad')\n"
        "dist_clases.index.name = 'Clase'\n"
        "display(dist_clases)"
    )
)

cells.append(md("## 3. Visualización exploratoria"))

cells.append(
    code(
        "fig, ax = plt.subplots(1, 1, figsize=(8, 4))\n"
        "sns.countplot(data=df, x=TARGET, hue=TARGET, dodge=False, palette='viridis', legend=False, ax=ax)\n"
        "ax.set_title('Distribución de clases')\n"
        "ax.set_xlabel('Clase')\n"
        "ax.set_ylabel('Cantidad de estudiantes')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'grafica_distribucion_clases.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
        "for ax, col in zip(axes.flat, CONTINUOUS):\n"
        "    sns.histplot(df[col], bins=20, kde=True, ax=ax, color='#2E86AB')\n"
        "    ax.set_title(f\"Distribución de {DISPLAY_NAMES[col]}\")\n"
        "    ax.set_xlabel(DISPLAY_NAMES[col])\n"
        "    ax.set_ylabel('Frecuencia')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'graficas_distribucion_variables_continuas.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(
    code(
        "numeric_cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA', 'GradeClass']\n"
        "corr = df[numeric_cols].corr(numeric_only=True)\n"
        "corr = corr.rename(index=columnas_mostrar, columns=columnas_mostrar)\n"
        "plt.figure(figsize=(11, 8))\n"
        "sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)\n"
        "plt.title('Matriz de correlación')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'matriz_correlacion.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
        "for ax, col in zip(axes.flat, CONTINUOUS):\n"
        "    sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, dodge=False, legend=False, ax=ax, palette='Set2')\n"
        "    ax.set_title(f\"{DISPLAY_NAMES[col].capitalize()} por clase\")\n"
        "    ax.set_xlabel('Clase')\n"
        "    ax.set_ylabel(DISPLAY_NAMES[col])\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'boxplots_variables_por_clase.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "## 4. Preparación de datos\n\n"
        "- Se excluye `StudentID` por tratarse de un identificador.\n"
        "- Se usa una **partición estratificada 70/15/15**."
    )
)

cells.append(
    code(
        "feature_columns = CONTINUOUS + CATEGORICAL\n"
        "X_df = df[feature_columns].copy()\n"
        "y = df[TARGET].to_numpy(dtype=np.int32)\n\n"
        "X_train_df, X_temp_df, y_train, y_temp = train_test_split(\n"
        "    X_df, y, test_size=0.30, stratify=y, random_state=SEED\n"
        ")\n"
        "X_val_df, X_test_df, y_val, y_test = train_test_split(\n"
        "    X_temp_df, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED\n"
        ")\n\n"
        "preprocessor = ColumnTransformer([\n"
        "    ('continuous', StandardScaler(), CONTINUOUS),\n"
        "    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),\n"
        "])\n\n"
        "X_train_mlp = np.asarray(preprocessor.fit_transform(X_train_df), dtype=np.float32)\n"
        "X_val_mlp = np.asarray(preprocessor.transform(X_val_df), dtype=np.float32)\n"
        "X_test_mlp = np.asarray(preprocessor.transform(X_test_df), dtype=np.float32)\n\n"
        "X_train_rf = X_train_df.astype(float).copy()\n"
        "X_val_rf = X_val_df.astype(float).copy()\n"
        "X_test_rf = X_test_df.astype(float).copy()\n\n"
        "print('Conjunto de entrenamiento (MLP):', X_train_mlp.shape)\n"
        "print('Conjunto de validación (MLP):', X_val_mlp.shape)\n"
        "print('Conjunto de prueba (MLP):', X_test_mlp.shape)\n"
        "print('Conjunto de entrenamiento (Random Forest):', X_train_rf.shape)\n"
        "print('Conjunto de validación (Random Forest):', X_val_rf.shape)\n"
        "print('Conjunto de prueba (Random Forest):', X_test_rf.shape)\n"
        "print('Clases en entrenamiento:', np.unique(y_train))"
    )
)

cells.append(md("## 5. Funciones auxiliares"))

cells.append(
    code(
        "def set_seed(seed: int = SEED) -> None:\n"
        "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
        "    random.seed(seed)\n"
        "    np.random.seed(seed)\n"
        "    tf.random.set_seed(seed)\n\n"
        "def collect_metrics(y_true, y_pred):\n"
        "    return {\n"
        "        'accuracy': accuracy_score(y_true, y_pred),\n"
        "        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),\n"
        "    }\n\n"
        "def build_optimizer(name: str, learning_rate: float):\n"
        "    if name == 'adam':\n"
        "        return keras.optimizers.Adam(learning_rate=learning_rate)\n"
        "    if name == 'rmsprop':\n"
        "        return keras.optimizers.RMSprop(learning_rate=learning_rate)\n"
        "    return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)\n\n"
        "def build_exam_mlp(input_dim: int, num_classes: int, optimizer_name: str, learning_rate: float):\n"
        "    model = keras.Sequential([\n"
        "        keras.layers.Input(shape=(input_dim,)),\n"
        "        keras.layers.Dense(200, activation='relu'),\n"
        "        keras.layers.Dropout(0.2),\n"
        "        keras.layers.Dense(200, activation='relu'),\n"
        "        keras.layers.Dropout(0.2),\n"
        "        keras.layers.Dense(150, activation='relu'),\n"
        "        keras.layers.Dropout(0.2),\n"
        "        keras.layers.Dense(200, activation='relu'),\n"
        "        keras.layers.Dropout(0.2),\n"
        "        keras.layers.Dense(len(np.unique(y_train)), activation='softmax'),\n"
        "    ])\n"
        "    model.compile(\n"
        "        optimizer=build_optimizer(optimizer_name, learning_rate),\n"
        "        loss='sparse_categorical_crossentropy',\n"
        "        metrics=['accuracy'],\n"
        "    )\n"
        "    return model"
    )
)

cells.append(md("## 6. Búsqueda de hiperparámetros para el MLP principal"))

cells.append(
    code(
        "best_mlp_model = None\n"
        "best_mlp_cfg = None\n"
        "best_mlp_history = None\n"
        "best_mlp_score = -np.inf\n"
        "best_mlp_loss = np.inf\n"
        "mlp_rows = []\n\n"
        "for cfg in ParameterGrid(MLP_GRID):\n"
        "    set_seed()\n"
        "    model = build_exam_mlp(X_train_mlp.shape[1], len(np.unique(y_train)), cfg['optimizer'], cfg['learning_rate'])\n"
        "    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]\n"
        "    history = model.fit(\n"
        "        X_train_mlp, y_train,\n"
        "        validation_data=(X_val_mlp, y_val),\n"
        "        epochs=cfg['epochs'],\n"
        "        batch_size=cfg['batch_size'],\n"
        "        verbose=0,\n"
        "        callbacks=callbacks,\n"
        "    )\n"
        "    pred_val = np.argmax(model.predict(X_val_mlp, verbose=0), axis=1)\n"
        "    metrics_val = collect_metrics(y_val, pred_val)\n"
        "    val_loss = float(np.min(history.history['val_loss']))\n"
        "    row = {\n"
        "        **cfg,\n"
        "        'epochs_ran': len(history.history['loss']),\n"
        "        'val_accuracy': round(metrics_val['accuracy'], 4),\n"
        "        'val_f1_macro': round(metrics_val['f1_macro'], 4),\n"
        "        'val_recall_macro': round(metrics_val['recall_macro'], 4),\n"
        "        'val_loss': round(val_loss, 4),\n"
        "    }\n"
        "    mlp_rows.append(row)\n"
        "    if metrics_val['f1_macro'] > best_mlp_score or (np.isclose(metrics_val['f1_macro'], best_mlp_score) and val_loss < best_mlp_loss):\n"
        "        best_mlp_score = metrics_val['f1_macro']\n"
        "        best_mlp_loss = val_loss\n"
        "        best_mlp_model = model\n"
        "        best_mlp_cfg = dict(cfg)\n"
        "        best_mlp_history = history.history\n\n"
        "mlp_search_df = pd.DataFrame(mlp_rows).sort_values(['val_f1_macro', 'val_accuracy'], ascending=[False, False])\n"
        "print('Mejor configuración MLP:', best_mlp_cfg)\n"
        "display(mlp_search_df.head(12))"
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].plot(best_mlp_history['loss'], label='Entrenamiento')\n"
        "axes[0].plot(best_mlp_history['val_loss'], label='Validación')\n"
        "axes[0].set_title('Pérdida del mejor MLP')\n"
        "axes[0].set_xlabel('Época')\n"
        "axes[0].legend()\n\n"
        "axes[1].plot(best_mlp_history['accuracy'], label='Entrenamiento')\n"
        "axes[1].plot(best_mlp_history['val_accuracy'], label='Validación')\n"
        "axes[1].set_title('Exactitud del mejor MLP')\n"
        "axes[1].set_xlabel('Época')\n"
        "axes[1].legend()\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'curvas_entrenamiento_mlp.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 6.1 Diagrama de la arquitectura del MLP principal"))

cells.append(
    code(
        "def draw_network_mlp_examen(input_dim: int, num_classes: int):\n"
        "    fig, ax = plt.subplots(figsize=(15, 7), facecolor='#f7f7f7')\n"
        "    ax.set_facecolor('#f7f7f7')\n"
        "    ax.axis('off')\n\n"
        "    xp = {'input': 0, 'dense1': 2, 'dense2': 4, 'dense3': 6, 'dense4': 8, 'output': 10}\n"
        "    colors = {\n"
        "        'input': '#6C5CE7',\n"
        "        'dense1': '#E84393',\n"
        "        'dense2': '#F39C12',\n"
        "        'dense3': '#2ECC71',\n"
        "        'dense4': '#E74C3C',\n"
        "        'output': '#D4A017',\n"
        "    }\n"
        "    labels = {\n"
        "        'input': f'Entrada\\n({input_dim} variables)',\n"
        "        'dense1': 'Densa + ReLU\\nDropout(0.2)',\n"
        "        'dense2': 'Densa + ReLU\\nDropout(0.2)',\n"
        "        'dense3': 'Densa + ReLU\\nDropout(0.2)',\n"
        "        'dense4': 'Densa + ReLU\\nDropout(0.2)',\n"
        "        'output': f'Salida\\n{num_classes}, Softmax',\n"
        "    }\n"
        "    sizes = {'input': str(input_dim), 'dense1': '200', 'dense2': '200', 'dense3': '150', 'dense4': '200', 'output': str(num_classes)}\n"
        "    y_nodes = [4.5, 3.5, 2.5, 1.5, 0.5]\n\n"
        "    def draw_column(x, color):\n"
        "        for y in y_nodes:\n"
        "            circ = plt.Circle((x, y), 0.18, color=color, ec='white', lw=2, alpha=0.95)\n"
        "            ax.add_patch(circ)\n"
        "        ax.text(x, 2.0, '⋯', ha='center', va='center', fontsize=18, color='#777777')\n\n"
        "    def connect_columns(x1, x2):\n"
        "        for y1 in y_nodes:\n"
        "            for y2 in y_nodes:\n"
        "                ax.plot([x1 + 0.18, x2 - 0.18], [y1, y2], color='#cccccc', lw=0.8, alpha=0.35, zorder=0)\n\n"
        "    order = ['input', 'dense1', 'dense2', 'dense3', 'dense4', 'output']\n"
        "    for left, right in zip(order[:-1], order[1:]):\n"
        "        connect_columns(xp[left], xp[right])\n"
        "    for key in order:\n"
        "        draw_column(xp[key], colors[key])\n"
        "        ax.text(\n"
        "            xp[key], 5.5, labels[key],\n"
        "            ha='center', va='center', fontsize=10, fontweight='bold', color='white',\n"
        "            bbox=dict(boxstyle='round,pad=0.35', facecolor=colors[key], edgecolor='none', alpha=0.95)\n"
        "        )\n"
        "        ax.text(\n"
        "            xp[key], -0.1, f'n = {sizes[key]}',\n"
        "            ha='center', fontsize=8, color='#555555',\n"
        "            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=colors[key], alpha=0.9)\n"
        "        )\n\n"
        "    edge_labels = [('input', 'dense1', '$W_1, b_1$'), ('dense1', 'dense2', '$W_2, b_2$'), ('dense2', 'dense3', '$W_3, b_3$'), ('dense3', 'dense4', '$W_4, b_4$'), ('dense4', 'output', '$W_5, b_5$')]\n"
        "    for left, right, txt in edge_labels:\n"
        "        xm = (xp[left] + xp[right]) / 2\n"
        "        ax.text(xm, 4.95, txt, ha='center', va='bottom', fontsize=9, color='#666666')\n\n"
        "    ax.set_title(\n"
        "        f'Arquitectura del MLP principal del examen ({input_dim} → 200 → 200 → 150 → 200 → {num_classes})',\n"
        "        fontsize=14, fontweight='bold', pad=14, color='#222222'\n"
        "    )\n"
        "    ax.set_xlim(-0.7, 10.7)\n"
        "    ax.set_ylim(-0.5, 6.1)\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(BASE_DIR / 'arquitectura_mlp_principal.png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())\n"
        "    plt.show()\n"
        "draw_network_mlp_examen(X_train_mlp.shape[1], len(np.unique(y_train)))"
    )
)

cells.append(
    code(
        "plt.figure(figsize=(10, 4))\n"
        "top_mlp = mlp_search_df.head(10).copy()\n"
        "top_mlp = top_mlp.rename(columns={\n"
        "    'optimizer': 'optimizador',\n"
        "    'learning_rate': 'tasa_aprendizaje',\n"
        "    'batch_size': 'tamano_lote',\n"
        "    'epochs': 'epocas_solicitadas',\n"
        "    'epochs_ran': 'epocas_ejecutadas',\n"
        "    'val_accuracy': 'exactitud_validacion',\n"
        "    'val_f1_macro': 'puntaje_f1_macro_validacion',\n"
        "    'val_recall_macro': 'sensibilidad_macro_validacion',\n"
        "    'val_loss': 'perdida_validacion',\n"
        "})\n"
        "top_mlp['configuracion'] = top_mlp['optimizador'] + ' | tasa=' + top_mlp['tasa_aprendizaje'].astype(str) + ' | lote=' + top_mlp['tamano_lote'].astype(str)\n"
        "sns.barplot(data=top_mlp, x='puntaje_f1_macro_validacion', y='configuracion', color='#4C78A8')\n"
        "plt.title('Mejores configuraciones del MLP según puntaje F1 macro en validación')\n"
        "plt.xlabel('Puntaje F1 macro')\n"
        "plt.ylabel('Configuración')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'comparacion_hiperparametros_mlp.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 7. Búsqueda de hiperparámetros para Random Forest"))

cells.append(
    code(
        "best_rf_model = None\n"
        "best_rf_cfg = None\n"
        "best_rf_score = -np.inf\n"
        "rf_rows = []\n\n"
        "for cfg in ParameterGrid(RF_GRID):\n"
        "    model = RandomForestClassifier(random_state=SEED, n_jobs=-1, **cfg)\n"
        "    model.fit(X_train_rf, y_train)\n"
        "    pred_val = model.predict(X_val_rf)\n"
        "    metrics_val = collect_metrics(y_val, pred_val)\n"
        "    row = {\n"
        "        **cfg,\n"
        "        'val_accuracy': round(metrics_val['accuracy'], 4),\n"
        "        'val_f1_macro': round(metrics_val['f1_macro'], 4),\n"
        "        'val_recall_macro': round(metrics_val['recall_macro'], 4),\n"
        "    }\n"
        "    rf_rows.append(row)\n"
        "    if metrics_val['f1_macro'] > best_rf_score:\n"
        "        best_rf_score = metrics_val['f1_macro']\n"
        "        best_rf_model = model\n"
        "        best_rf_cfg = dict(cfg)\n\n"
        "rf_search_df = pd.DataFrame(rf_rows).sort_values(['val_f1_macro', 'val_accuracy'], ascending=[False, False])\n"
        "print('Mejor configuración Random Forest:', best_rf_cfg)\n"
        "display(rf_search_df.head(12))"
    )
)

cells.append(
    code(
        "plt.figure(figsize=(10, 4))\n"
        "top_rf = rf_search_df.head(10).copy()\n"
        "top_rf = top_rf.rename(columns={\n"
        "    'n_estimators': 'numero_arboles',\n"
        "    'max_depth': 'profundidad_maxima',\n"
        "    'min_samples_leaf': 'min_muestras_hoja',\n"
        "    'class_weight': 'peso_clase',\n"
        "    'val_accuracy': 'exactitud_validacion',\n"
        "    'val_f1_macro': 'puntaje_f1_macro_validacion',\n"
        "    'val_recall_macro': 'sensibilidad_macro_validacion',\n"
        "})\n"
        "top_rf['configuracion'] = (\n"
        "    'árboles=' + top_rf['numero_arboles'].astype(str)\n"
        "    + ' | prof=' + top_rf['profundidad_maxima'].astype(str)\n"
        "    + ' | hoja=' + top_rf['min_muestras_hoja'].astype(str)\n"
        ")\n"
        "sns.barplot(data=top_rf, x='puntaje_f1_macro_validacion', y='configuracion', color='#59A14F')\n"
        "plt.title('Mejores configuraciones de Random Forest según puntaje F1 macro en validación')\n"
        "plt.xlabel('Puntaje F1 macro')\n"
        "plt.ylabel('Configuración')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'comparacion_hiperparametros_random_forest.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 8. Evaluación final en el conjunto de prueba"))

cells.append(
    code(
        "labels = sorted(np.unique(y_test).tolist())\n"
        "mlp_pred = np.argmax(best_mlp_model.predict(X_test_mlp, verbose=0), axis=1)\n"
        "rf_pred = best_rf_model.predict(X_test_rf)\n\n"
        "mlp_metrics = collect_metrics(y_test, mlp_pred)\n"
        "rf_metrics = collect_metrics(y_test, rf_pred)\n\n"
        "comparison_df = pd.DataFrame([\n"
        "    {'modelo': 'MLP principal', 'mejor_configuracion': str(best_mlp_cfg), **{k: round(v, 4) for k, v in mlp_metrics.items()}},\n"
        "    {'modelo': 'Random Forest', 'mejor_configuracion': str(best_rf_cfg), **{k: round(v, 4) for k, v in rf_metrics.items()}},\n"
        "])\n"
        "comparison_df = comparison_df.rename(columns={\n"
        "    'accuracy': 'exactitud',\n"
        "    'precision_macro': 'precisión_macro',\n"
        "    'recall_macro': 'sensibilidad_macro',\n"
        "    'f1_macro': 'puntaje_f1_macro',\n"
        "})\n"
        "display(comparison_df)"
    )
)

cells.append(
    code(
        "report_mlp_df = pd.DataFrame(classification_report(y_test, mlp_pred, output_dict=True, zero_division=0)).transpose().reset_index().rename(columns={'index': 'clase'})\n"
        "report_rf_df = pd.DataFrame(classification_report(y_test, rf_pred, output_dict=True, zero_division=0)).transpose().reset_index().rename(columns={'index': 'clase'})\n\n"
        "print('Reporte de clasificación del MLP principal')\n"
        "display(report_mlp_df)\n\n"
        "print('Reporte de clasificación de Random Forest')\n"
        "display(report_rf_df)"
    )
)

cells.append(md("## 9. Matrices de confusión normalizadas"))

cells.append(
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "cm_mlp = confusion_matrix(y_test, mlp_pred, labels=labels, normalize='true')\n"
        "cm_rf = confusion_matrix(y_test, rf_pred, labels=labels, normalize='true')\n"
        "sns.heatmap(cm_mlp, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])\n"
        "axes[0].set_title('MLP principal')\n"
        "axes[0].set_xlabel('Clase predicha')\n"
        "axes[0].set_ylabel('Clase real')\n"
        "sns.heatmap(cm_rf, annot=True, fmt='.2f', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=axes[1])\n"
        "axes[1].set_title('Random Forest')\n"
        "axes[1].set_xlabel('Clase predicha')\n"
        "axes[1].set_ylabel('Clase real')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'matrices_confusion_comparadas.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 10. Comparación visual de métricas"))

cells.append(
    code(
        "plot_df = comparison_df.melt(id_vars=['modelo', 'mejor_configuracion'], value_vars=['exactitud', 'precisión_macro', 'sensibilidad_macro', 'puntaje_f1_macro'], var_name='metrica', value_name='valor')\n"
        "plot_df['metrica'] = plot_df['metrica'].replace({\n"
        "    'exactitud': 'Exactitud',\n"
        "    'precisión_macro': 'Precisión macro',\n"
        "    'sensibilidad_macro': 'Sensibilidad macro',\n"
        "    'puntaje_f1_macro': 'Puntaje F1 macro',\n"
        "})\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.barplot(data=plot_df, x='metrica', y='valor', hue='modelo')\n"
        "plt.ylim(0, 1.0)\n"
        "plt.title('Comparación de métricas en prueba')\n"
        "plt.xlabel('Métrica')\n"
        "plt.ylabel('Valor')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'comparacion_metricas_modelos.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 11. Importancia de variables con Random Forest"))

cells.append(
    code(
        "importance_df = pd.DataFrame({'variable': X_train_rf.columns, 'importancia': best_rf_model.feature_importances_}).sort_values('importancia', ascending=False).reset_index(drop=True)\n"
        "display(importance_df.head(15))\n\n"
        "plt.figure(figsize=(10, 5))\n"
        "top_importance = importance_df.head(12)\n"
        "sns.barplot(data=top_importance, x='importancia', y='variable', hue='variable', dodge=False, palette='viridis', legend=False)\n"
        "plt.title('Variables más relevantes según Random Forest')\n"
        "plt.xlabel('Importancia')\n"
        "plt.ylabel('Variable')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'importancia_variables_random_forest.png', dpi=200, bbox_inches='tight')\n"
        "plt.show()"
    )
)

cells.append(md("## 12. Exportación de artefactos para el reporte"))

cells.append(
    code(
        "comparison_df.to_csv(BASE_DIR / 'comparacion_modelos_final.csv', index=False)\n"
        "mlp_search_df.to_csv(BASE_DIR / 'hiperparametros_mlp_final.csv', index=False)\n"
        "rf_search_df.to_csv(BASE_DIR / 'hiperparametros_random_forest_final.csv', index=False)\n"
        "report_mlp_df.to_csv(BASE_DIR / 'classification_report_mlp_final.csv', index=False)\n"
        "report_rf_df.to_csv(BASE_DIR / 'classification_report_random_forest_final.csv', index=False)\n"
        "importance_df.to_csv(BASE_DIR / 'importancia_variables_random_forest_final.csv', index=False)\n\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.heatmap(cm_mlp, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)\n"
        "ax.set_title('Matriz de confusión normalizada - MLP principal')\n"
        "ax.set_xlabel('Clase predicha')\n"
        "ax.set_ylabel('Clase real')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'confusion_matrix_mlp_final.png', dpi=200)\n"
        "plt.close(fig)\n\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sns.heatmap(cm_rf, annot=True, fmt='.2f', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax)\n"
        "ax.set_title('Matriz de confusión normalizada - Random Forest final')\n"
        "ax.set_xlabel('Clase predicha')\n"
        "ax.set_ylabel('Clase real')\n"
        "plt.tight_layout()\n"
        "plt.savefig(BASE_DIR / 'confusion_matrix_random_forest_final.png', dpi=200)\n"
        "plt.close(fig)\n\n"
        "for file_name in [\n"
        "    'grafica_distribucion_clases.png',\n"
        "    'graficas_distribucion_variables_continuas.png',\n"
        "    'matriz_correlacion.png',\n"
        "    'boxplots_variables_por_clase.png',\n"
        "    'arquitectura_mlp_principal.png',\n"
        "    'curvas_entrenamiento_mlp.png',\n"
        "    'comparacion_hiperparametros_mlp.png',\n"
        "    'comparacion_hiperparametros_random_forest.png',\n"
        "    'matrices_confusion_comparadas.png',\n"
        "    'comparacion_metricas_modelos.png',\n"
        "    'importancia_variables_random_forest.png',\n"
        "    'confusion_matrix_mlp_final.png',\n"
        "    'confusion_matrix_random_forest_final.png',\n"
        "]:\n"
        "    print(BASE_DIR / file_name)"
    )
)

nb["cells"] = cells
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook generado en: {OUT}")
