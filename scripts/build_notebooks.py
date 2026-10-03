"""
Script para generar los 4 Jupyter Notebooks del proyecto NeuroScan AI.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


# ==============================================================================
# NOTEBOOK 1: Exploración, Manifiesto y Detección de Fuga de Datos
# ==============================================================================
nb1_cells = [
    md_cell(
        """# NeuroScan AI: Exploración de Datos, Manifiesto y Auditoría de Fuga
## Módulo 3: Databricks e IA Aplicada - Proyecto Final

Este notebook implementa la **Etapa 1** del pipeline:
1. Verificación de integridad y descarte de imágenes corruptas.
2. Generación del manifiesto estructurado con `pandas` y el paso equivalente en `PySpark`.
3. Auditoría empírica de **fuga de datos (data leakage)** entre particiones.
4. Creación del split estratificado (70% train, 15% val, 15% test) guardado en `data/manifest_split.csv`."""
    ),
    code_cell(
        """import sys
from pathlib import Path

# Configurar path raíz del proyecto
PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CLASSES, DATA_DIR, MANIFEST_PATH, SPLIT_MANIFEST_PATH, SPLIT_RATIOS, SEED
from src.dataset import build_manifest, split_manifest, build_manifest_spark

print(f"Ruta de datos: {DATA_DIR}")
print(f"Clases a clasificar: {CLASSES}")"""
    ),
    md_cell("### 1. Construcción del Manifiesto con Auditoría de Integridad"),
    code_cell(
        """# Construir el manifiesto auditando imágenes corruptas
manifest_df = build_manifest(data_dir=DATA_DIR, use_cache=False)
print(f"Total imágenes válidas: {len(manifest_df)}")
manifest_df.head()"""
    ),
    md_cell("### 2. Distribución de Clases"),
    code_cell(
        """class_counts = manifest_df["label_name"].value_counts()
print("Distribución de imágenes por patología:")
print(class_counts)

plt.figure(figsize=(8, 4.5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="Blues_r")
plt.title("Distribución de Resonancias por Clase (Brain Tumor MRI)")
plt.xlabel("Diagnóstico")
plt.ylabel("Número de Imágenes")
plt.grid(axis='y', alpha=0.3)
plt.show()"""
    ),
    md_cell("### 3. Equivalente Simbólico en PySpark (Fase 2 Databricks)"),
    code_cell(
        """try:
    spark_df = build_manifest_spark(data_dir=DATA_DIR)
    print("Esquema generado en PySpark (Simulación de Unity Catalog):")
    spark_df.printSchema()
    spark_df.groupBy("raw_folder").count().show()
except Exception as e:
    print(f"Aviso PySpark (requiere Java/Spark en entorno local): {e}")"""
    ),
    md_cell("### 4. Partición Estratificada y Verificación de Fuga de Datos"),
    code_cell(
        """# Generar split 70/15/15 estratificado
split_df = split_manifest(manifest_df, split_ratios=SPLIT_RATIOS, seed=SEED)

train_paths = set(split_df[split_df["split"] == "train"]["path"])
val_paths = set(split_df[split_df["split"] == "val"]["path"])
test_paths = set(split_df[split_df["split"] == "test"]["path"])

# Auditoría estricta de intersección
leakage_train_val = len(train_paths.intersection(val_paths))
leakage_train_test = len(train_paths.intersection(test_paths))
leakage_val_test = len(val_paths.intersection(test_paths))

print(f"Fuga Train - Val: {leakage_train_val} imágenes")
print(f"Fuga Train - Test: {leakage_train_test} imágenes")
print(f"Fuga Val - Test: {leakage_val_test} imágenes")

assert leakage_train_val == 0 and leakage_train_test == 0 and leakage_val_test == 0, "¡Fuga de datos detectada!"
print("\\n✅ COMPROBACIÓN EXITOSA: Partición sin fuga de datos confirmada y guardada en data/manifest_split.csv")"""
    ),
]

# ==============================================================================
# NOTEBOOK 2: Preprocesamiento y Pipeline tf.data
# ==============================================================================
nb2_cells = [
    md_cell(
        """# NeuroScan AI: Preprocesamiento y Pipeline tf.data
## Módulo 3: Databricks e IA Aplicada - Proyecto Final

Este notebook implementa la **Etapa 2**:
1. Construcción de canales estandarizados `(224, 224, 3)`.
2. Normalización de pesos ImageNet (BGR mean centering).
3. Data Augmentation clínicamente seguro para neuroimagen.
4. Optimización de pipeline con `prefetch(AUTOTUNE)` y verificación de lotes."""
    ),
    code_cell(
        """import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import BATCH_SIZE, CLASSES, IMG_CHANNELS, IMG_SIZE, NUM_CLASSES, SPLIT_MANIFEST_PATH
from src.preprocessing import create_tf_dataset

print(f"TensorFlow Version: {tf.__version__}")
print(f"Dispositivos disponibles: {tf.config.list_physical_devices()}")"""
    ),
    md_cell("### 1. Cargar el Split de Datos"),
    code_cell(
        """df_split = pd.read_csv(SPLIT_MANIFEST_PATH)
print("Resumen de splits:")
print(df_split.groupby(["split", "label_name"]).size().unstack())"""
    ),
    md_cell("### 2. Construir Pipelines tf.data (Train, Val, Test)"),
    code_cell(
        """train_ds = create_tf_dataset(df_split, split_name="train", batch_size=BATCH_SIZE, model_type="resnet50")
val_ds = create_tf_dataset(df_split, split_name="val", batch_size=BATCH_SIZE, model_type="resnet50")
test_ds = create_tf_dataset(df_split, split_name="test", batch_size=BATCH_SIZE, model_type="resnet50")

print(f"Pipeline construido: Lotes de tamaño {BATCH_SIZE} listos para GPU/CPU.")"""
    ),
    md_cell("### 3. Inspección Visual de un Lote"),
    code_cell(
        """for images, labels in train_ds.take(1):
    print(f"Tensor de imágenes: {images.shape}, tipo: {images.dtype}")
    print(f"Tensor de etiquetas: {labels.shape}, tipo: {labels.dtype}")
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, ax in enumerate(axes.flat):
        img_np = images[i].numpy()
        # Deshacer centrado de media para visualización legible
        img_disp = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        label_idx = np.argmax(labels[i].numpy())
        ax.imshow(img_disp)
        ax.set_title(f"Clase: {CLASSES[label_idx]}")
        ax.axis("off")
    plt.suptitle("Muestras de Resonancias Preprocesadas con Data Augmentation")
    plt.tight_layout()
    plt.show()"""
    ),
]

# ==============================================================================
# NOTEBOOK 3: Entrenamiento en Colab Pro con GPU y MLflow Tracking
# ==============================================================================
nb3_cells = [
    md_cell(
        """# NeuroScan AI: Entrenamiento Acelerado en Google Colab Pro con MLflow
## Módulo 3: Databricks e IA Aplicada - Proyecto Final

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/)

Este notebook implementa la **Etapa 3**:
1. Detección y activación de aceleración por GPU (NVIDIA A100 / V100 / L4) y **Mixed Precision (FP16)**.
2. Sincronización automática con Google Drive para respaldar checkpoints y base de datos SQLite de MLflow.
3. Ejecución de **5 Experimentos de Transfer Learning**:
   - **Run 1:** ResNet50 Base Congelada (LR 1e-3, Dropout 0.3)
   - **Run 2:** ResNet50 Base Congelada (LR 1e-4, Dropout 0.5)
   - **Run 3:** ResNet50 Fine-Tuning capa 143 (LR 1e-5, Dropout 0.3)
   - **Run 4 (Variante 1):** ResNet50 Fine-Tuning profundo capa 120 + Cosine Decay
   - **Run 5 (Variante 2):** EfficientNetB2 Transfer Learning (Comparación Multiarquitectura)"""
    ),
    code_cell(
        """# 1. Configuración de Entorno y Detección de GPU
import os
import sys
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Aceleradora Detectada: {gpus}")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("🚀 Mixed Precision FP16 habilitada para Tensor Cores.")
else:
    print("⚠️ No se detectó GPU. Ejecutando en CPU.")"""
    ),
    code_cell(
        """# 2. Montar Google Drive para Respaldo Resiliente
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = '/content/drive/MyDrive/m3_project'
    os.makedirs(f"{DRIVE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{DRIVE_DIR}/backups", exist_ok=True)
    print(f"✅ Google Drive conectado en: {DRIVE_DIR}")
except Exception as e:
    print(f"Ejecutando en entorno local sin montaje de Colab Drive: {e}")"""
    ),
    code_cell(
        """# 3. Importar Módulos del Sistema
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import BATCH_SIZE, CLASSES, SPLIT_MANIFEST_PATH
from src.preprocessing import create_tf_dataset
from src.train import train_and_log_run

df_split = pd.read_csv(SPLIT_MANIFEST_PATH)
train_ds = create_tf_dataset(df_split, split_name="train", batch_size=BATCH_SIZE, model_type="resnet50")
val_ds = create_tf_dataset(df_split, split_name="val", batch_size=BATCH_SIZE, model_type="resnet50")
test_ds = create_tf_dataset(df_split, split_name="test", batch_size=BATCH_SIZE, model_type="resnet50")
print(f"Datasets listos con {len(df_split)} imágenes en total.")"""
    ),
    md_cell("### 4. Ejecución de los 5 Experimentos en MLflow"),
    code_cell(
        """# Run 1: ResNet50 Frozen Base (LR 1e-3, Dropout 0.3)
m1, h1, met1 = train_and_log_run(
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    run_name="Run_1_ResNet50_Frozen_1e3",
    backbone_name="resnet50",
    epochs=12, learning_rate=1e-3, dropout=0.3,
    base_trainable=False
)"""
    ),
    code_cell(
        """# Run 2: ResNet50 Frozen Base (LR 1e-4, Dropout 0.5)
m2, h2, met2 = train_and_log_run(
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    run_name="Run_2_ResNet50_Frozen_1e4",
    backbone_name="resnet50",
    epochs=12, learning_rate=1e-4, dropout=0.5,
    base_trainable=False
)"""
    ),
    code_cell(
        """# Run 3: ResNet50 Fine-Tuning capa 143 (LR 1e-5, Dropout 0.3)
m3, h3, met3 = train_and_log_run(
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    run_name="Run_3_ResNet50_FineTune_143",
    backbone_name="resnet50",
    epochs=15, learning_rate=1e-5, dropout=0.3,
    base_trainable=True, fine_tune_at=143
)"""
    ),
    code_cell(
        """# Run 4 (Variante 1): ResNet50 Fine-Tuning profundo capa 120 + Cosine Decay
m4, h4, met4 = train_and_log_run(
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    run_name="Run_4_ResNet50_DeepFineTune_120_Cosine",
    backbone_name="resnet50",
    epochs=15, learning_rate=3e-5, dropout=0.3,
    base_trainable=True, fine_tune_at=120,
    use_cosine_decay=True
)"""
    ),
    code_cell(
        """# Run 5 (Variante 2): EfficientNetB2 Transfer Learning (Comparación Multiarquitectura)
train_eff = create_tf_dataset(df_split, split_name="train", batch_size=BATCH_SIZE, model_type="efficientnetb2")
val_eff = create_tf_dataset(df_split, split_name="val", batch_size=BATCH_SIZE, model_type="efficientnetb2")
test_eff = create_tf_dataset(df_split, split_name="test", batch_size=BATCH_SIZE, model_type="efficientnetb2")

m5, h5, met5 = train_and_log_run(
    train_dataset=train_eff, val_dataset=val_eff, test_dataset=test_eff,
    run_name="Run_5_EfficientNetB2_Transfer",
    backbone_name="efficientnetb2",
    epochs=12, learning_rate=1e-3, dropout=0.3,
    base_trainable=False
)"""
    ),
    md_cell("### 5. Sincronización Final de Resultados a Google Drive"),
    code_cell(
        """from scripts.sync_drive import backup_to_gdrive
backup_to_gdrive()
print("✅ Todos los checkpoints, modelos y la base de datos de MLflow fueron respaldados en Google Drive.")"""
    ),
]

# ==============================================================================
# NOTEBOOK 4: Evaluación, Model Registry y Fase 2 Databricks
# ==============================================================================
nb4_cells = [
    md_cell(
        """# NeuroScan AI: Evaluación Comparativa, Model Registry y Fase 2 Databricks
## Módulo 3: Databricks e IA Aplicada - Proyecto Final

Este notebook implementa la **Etapa 5**:
1. Consulta y tabla comparativa de los 5 experimentos registrados en MLflow.
2. Identificación del modelo ganador mediante Macro-F1 y Multiclass AUC.
3. Registro formal en el **Model Registry** de MLflow (`neuro_mri_classifier`) y promoción a `Staging`.
4. Exportación atómica del modelo desacoplado a `models/neuro_mri_model.keras`.
5. Especificación técnica de la **Fase 2 (Migración a Databricks Workspace, Unity Catalog y Delta Lake)**."""
    ),
    code_cell(
        """import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import (
    FINAL_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CHECKPOINTS_DIR,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
print(f"Experimento MLflow: {experiment.name} (ID: {experiment.experiment_id})")"""
    ),
    md_cell("### 1. Tabla Comparativa de Experimentos"),
    code_cell(
        """runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_macro_f1 DESC"])

records = []
for r in runs:
    records.append({
        "run_id": r.info.run_id[:8],
        "run_name": r.data.tags.get("mlflow.runName", "N/A"),
        "backbone": r.data.params.get("backbone"),
        "learning_rate": r.data.params.get("learning_rate"),
        "macro_f1": round(r.data.metrics.get("test_macro_f1", 0.0), 4),
        "weighted_f1": round(r.data.metrics.get("test_weighted_f1", 0.0), 4),
        "precision": round(r.data.metrics.get("test_macro_precision", 0.0), 4),
        "recall": round(r.data.metrics.get("test_macro_recall", 0.0), 4),
        "multiclass_auc": round(r.data.metrics.get("test_multiclass_auc", 0.0), 4),
    })

comparison_df = pd.DataFrame(records)
print("=== TABLA COMPARATIVA DE EXPERIMENTOS MLFLOW ===")
comparison_df"""
    ),
    md_cell("### 2. Registro en Model Registry y Promoción a Staging"),
    code_cell(
        """best_run = runs[0]
best_run_name = best_run.data.tags.get("mlflow.runName")
best_run_id = best_run.info.run_id
print(f"Modelo Campeón: {best_run_name} (Run ID: {best_run_id})")

model_uri = f"runs:/{best_run_id}/model"

try:
    mv = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
    print(f"✅ Modelo registrado: '{MLFLOW_MODEL_NAME}' Versión: {mv.version}")
    
    # Transición a etapa Staging
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=True,
    )
    print(f"✅ Versión {mv.version} promovida formalmente a etapa 'Staging'.")
except Exception as e:
    print(f"Aviso en registro de modelo MLflow: {e}")"""
    ),
    md_cell("### 3. Exportación Atómica a `models/neuro_mri_model.keras`"),
    code_cell(
        """# Copiar el mejor checkpoint a la ruta de producción desacoplada
best_ckpt_file = MODEL_CHECKPOINTS_DIR / f"{best_run_name}_best.keras"
if best_ckpt_file.exists():
    shutil.copy2(best_ckpt_file, FINAL_MODEL_PATH)
    print(f"✅ Artefacto de producción desacoplado exportado a: {FINAL_MODEL_PATH}")
else:
    print(f"Checkpoint {best_ckpt_file} no encontrado; verifique checkpoints locales.")"""
    ),
    md_cell(
        """### 4. Fase 2: Hoja de Ruta de Migración a Databricks

| Componente Actual (Fase 1 Híbrida) | Migración en Databricks (Fase 2 Producción) |
|---|---|
| Manifiesto local `manifest_split.csv` | **Tabla Delta en Unity Catalog** (`catalog.neuro_mri.manifest`) con versionado y time-travel. |
| Pipeline `tf.data` | **Petastorm / Spark-TensorFlow-Distributor** sobre volúmenes montados en DBFS / ADLS Gen2. |
| SQLite `mlflow.db` | **Managed MLflow Workspace** con tracking colaborativo integrado nativamente. |
| API FastAPI local en Uvicorn | **Databricks Model Serving** con endpoints de escalado a cero y monitoreo de deriva (*data drift*). |"""
    ),
]


def main():
    notebooks = {
        "01_exploracion_eda.ipynb": nb1_cells,
        "02_preprocessing.ipynb": nb2_cells,
        "03_training_colab_mlflow.ipynb": nb3_cells,
        "04_evaluation_registry.ipynb": nb4_cells,
    }

    for name, cells in notebooks.items():
        nb_path = NOTEBOOKS_DIR / name
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(make_notebook(cells), f, indent=2)
        print(f"Creado notebook: {nb_path.name}")


if __name__ == "__main__":
    main()
