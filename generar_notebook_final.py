from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path("/Volumes/dgx/UABC/RRNN/desercion_escolar")
OUT = BASE_DIR / "reporte_final_desercion_escolar.ipynb"


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
        "# Examen 1 - Reporte Experimental Final\n\n"
        "Notebook orientado al reporte IEEE para:\n"
        "- inspeccionar y visualizar el conjunto de datos;\n"
        "- entrenar el **MLP principal** con la arquitectura base del examen;\n"
        "- comparar contra **Random Forest** como modelo adicional sustentado en literatura;\n"
        "- generar tablas, métricas, matrices de confusión e importancia de variables.\n\n"
        "**Escenario central:** clasificación de `GradeClass` usando el dataset completo, incluyendo `GPA`."
    )
)

cells.append(
    md(
        "## 1. Librerías y configuración\n\n"
        "La semilla se fija para hacer reproducibles, en la medida de lo posible, las corridas."
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
        "BASE_DIR = Path.cwd()\n"
        "DATASET = BASE_DIR / 'Student_performance_data_clean.csv'\n"
        "TARGET = 'GradeClass'\n"
        "CONTINUOUS = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']\n"
        "CATEGORICAL = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']\n\n"
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
        "sns.set_theme(style='whitegrid', palette='deep')"
    )
)

cells.append(md("## 2. Carga y revisión básica del conjunto de datos"))

cells.append(
    code(
        "df = pd.read_csv(DATASET)\n"
        "df[TARGET] = df[TARGET].astype(int)\n"
        "df['StudyTimeWeekly'] = df['StudyTimeWeekly'].astype(float)\n"
        "df['GPA'] = df['GPA'].astype(float)\n\n"
        "print('Forma del dataset:', df.shape)\n"
        "display(df.head())\n\n"
        "print('Tipos de datos:')\n"
        "display(df.dtypes.to_frame('tipo'))\n\n"
        "print('Valores nulos por columna:')\n"
        "display(df.isna().sum().to_frame('nulos'))\n\n"
        "print('Distribución de clases:')\n"
        "display(df[TARGET].value_counts().sort_index().to_frame('cantidad'))"
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
        "plt.show()"
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
        "for ax, col in zip(axes.flat, CONTINUOUS):\n"
        "    sns.histplot(df[col], bins=20, kde=True, ax=ax, color='#2E86AB')\n"
        "    ax.set_title(f'Distribución de {col}')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    code(
        "numeric_cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA', 'GradeClass']\n"
        "corr = df[numeric_cols].corr(numeric_only=True)\n"
        "plt.figure(figsize=(11, 8))\n"
        "sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)\n"
        "plt.title('Matriz de correlación')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
        "for ax, col in zip(axes.flat, CONTINUOUS):\n"
        "    sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, dodge=False, legend=False, ax=ax, palette='Set2')\n"
        "    ax.set_title(f'{col} por clase')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    md(
        "## 4. Preparación de datos\n\n"
        "- Se elimina implícitamente `StudentID` por ser un identificador.\n"
        "- Se conserva `GPA` porque el enunciado no prohíbe su uso y la explicación oral del profesor sugiere que el dataset debe permitir desempeños superiores al 80%.\n"
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
        "print('Train MLP:', X_train_mlp.shape, 'Val MLP:', X_val_mlp.shape, 'Test MLP:', X_test_mlp.shape)\n"
        "print('Train RF:', X_train_rf.shape, 'Val RF:', X_val_rf.shape, 'Test RF:', X_test_rf.shape)\n"
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
        "plt.show()"
    )
)

cells.append(
    code(
        "plt.figure(figsize=(10, 4))\n"
        "top_mlp = mlp_search_df.head(10).copy()\n"
        "top_mlp['config'] = top_mlp['optimizer'] + ' | lr=' + top_mlp['learning_rate'].astype(str) + ' | lote=' + top_mlp['batch_size'].astype(str)\n"
        "sns.barplot(data=top_mlp, x='val_f1_macro', y='config', color='#4C78A8')\n"
        "plt.title('Mejores configuraciones del MLP según puntaje F1 macro en validación')\n"
        "plt.xlabel('Puntaje F1 macro')\n"
        "plt.ylabel('Configuración')\n"
        "plt.tight_layout()\n"
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
        "top_rf['config'] = (\n"
        "    'árboles=' + top_rf['n_estimators'].astype(str)\n"
        "    + ' | prof=' + top_rf['max_depth'].astype(str)\n"
        "    + ' | hoja=' + top_rf['min_samples_leaf'].astype(str)\n"
        ")\n"
        "sns.barplot(data=top_rf, x='val_f1_macro', y='config', color='#59A14F')\n"
        "plt.title('Mejores configuraciones de Random Forest según puntaje F1 macro en validación')\n"
        "plt.xlabel('Puntaje F1 macro')\n"
        "plt.ylabel('Configuración')\n"
        "plt.tight_layout()\n"
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
        "    'precision_macro': 'precision_macro',\n"
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
        "print('Reporte de clasificación - MLP principal')\n"
        "display(report_mlp_df)\n\n"
        "print('Reporte de clasificación - Random Forest')\n"
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
        "plt.show()"
    )
)

cells.append(md("## 10. Comparación visual de métricas"))

cells.append(
    code(
        "plot_df = comparison_df.melt(id_vars=['modelo', 'mejor_configuracion'], value_vars=['exactitud', 'precision_macro', 'sensibilidad_macro', 'puntaje_f1_macro'], var_name='metrica', value_name='valor')\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.barplot(data=plot_df, x='metrica', y='valor', hue='modelo')\n"
        "plt.ylim(0, 1.0)\n"
        "plt.title('Comparación de métricas en prueba')\n"
        "plt.xlabel('Métrica')\n"
        "plt.ylabel('Valor')\n"
        "plt.tight_layout()\n"
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
        "print('Archivos actualizados en:', BASE_DIR)"
    )
)

cells.append(
    md(
        "## 13. Observaciones para Overleaf\n\n"
        "- Incluir una figura con la **distribución de clases**.\n"
        "- Incluir una figura con la **matriz de correlación**.\n"
        "- Incluir una tabla con la **búsqueda de hiperparámetros del MLP**.\n"
        "- Incluir una tabla comparativa entre **MLP principal** y **Random Forest**.\n"
        "- Incluir las **matrices de confusión normalizadas** de ambos modelos.\n"
        "- Incluir una figura con la **importancia de variables** del Random Forest.\n"
        "- En la discusión, destacar que el `Random Forest` superó al MLP en este problema tabular.\n"
        "- Justificar el modelo adicional con literatura relacionada de predicción de rendimiento académico y deserción."
    )
)

nb["cells"] = cells
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook generado en: {OUT}")
