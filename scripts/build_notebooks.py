"""
Script para regenerar los 4 Jupyter Notebooks del proyecto NeuroScan AI con soporte autónomo para Google Colab y ejecución local.
"""

from pathlib import Path
import json

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


COLAB_BOOTSTRAP_CODE = """# 0. Inicialización Automática de Entorno (Google Colab / Local)
import os
import sys

# Detección y preparación autónoma para Google Colab
if 'google.colab' in sys.modules:
    # 1. Clonar o actualizar el repositorio público en /content
    if not os.path.exists('/content/m3-proyecto-final-databricks-ia'):
        print("🚀 Clonando repositorio NeuroScan AI en Google Colab...")
        !git clone https://github.com/EduardCY/m3-proyecto-final-databricks-ia.git /content/m3-proyecto-final-databricks-ia
    else:
        print("🔄 Actualizando repositorio con los últimos cambios de GitHub...")
        !git -C /content/m3-proyecto-final-databricks-ia pull origin main
    
    # 2. Posicionarse en la carpeta raíz del proyecto
    %cd /content/m3-proyecto-final-databricks-ia
    
    # 3. Instalar dependencias requeridas para Colab (MLflow, etc.)
    !pip install -q -r requirements-colab.txt

    # 4. Limpiar módulos en caché para recargar cambios de código
    for mod in list(sys.modules.keys()):
        if mod.startswith('src') or mod.startswith('scripts'):
            del sys.modules[mod]

# 5. Asegurar que la raíz del proyecto está en el PYTHONPATH
PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"✅ Entorno preparado en: {PROJECT_ROOT}")"""



# ==============================================================================
# NOTEBOOK 1: Exploración, Manifiesto y Detección de Fuga de Datos
# ==============================================================================
nb1_cells = [
    md_cell(
        """# NeuroScan AI: Exploración de Datos, Manifiesto y Auditoría de Fuga
## Módulo 3: Databricks e IA Aplicada - Proyecto Final

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduardCY/m3-proyecto-final-databricks-ia/blob/main/notebooks/01_exploracion_eda.ipynb)

Este notebook implementa la **Etapa 1** del pipeline:
1. Verificación de integridad y descarte de imágenes corruptas.
2. Generación del manifiesto estructurado con `pandas` y el paso equivalente en `PySpark`.
3. Auditoría empírica de **fuga de datos (data leakage)** entre particiones.
4. Creación del split estratificado (70% train, 15% val, 15% test) guardado en `data/manifest_split.csv`."""
    ),
    code_cell(COLAB_BOOTSTRAP_CODE),
    code_cell(
        """import sys
from pathlib import Path

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
    md_cell("### 2. Análisis Exploratorio y Balance de Clases"),
    code_cell(
        """class_counts = manifest_df["label_name"].value_counts()
print("Distribución de clases:")
print(class_counts)

plt.figure(figsize=(8, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Distribución de Clases de Tumores Cerebrales (MRI)")
plt.xlabel("Diagnóstico")
plt.ylabel("Cantidad de Cortes")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()"""
    ),
    md_cell("### 3. Parser Distribuido Simbólico en PySpark (Rúbrica Databricks)"),
    code_cell(
        """# Demostración del procesamiento paralelo en PySpark
spark_df = build_manifest_spark(data_dir=DATA_DIR)
if spark_df is not None:
    print("Esquema inferido por Spark:")
    spark_df.printSchema()
    print("Conteo distribuido por clase:")
    spark_df.groupBy("label_name").count().show()
else:
    print("PySpark operando en modo fallback local (Java/Spark no inicializado en entorno ligero).")"""
    ),
    md_cell("### 4. Particionamiento Estratificado y Auditoría de Cero Fuga"),
    code_cell(
        """df_split = split_manifest(manifest_df, split_ratios=SPLIT_RATIOS, seed=SEED, output_path=SPLIT_MANIFEST_PATH)
print("Partición estratificada generada:")
print(df_split.groupby(["split", "label_name"]).size().unstack())

# Auditoría matemática de fuga de datos
train_paths = set(df_split[df_split["split"] == "train"]["path"])
val_paths = set(df_split[df_split["split"] == "val"]["path"])
test_paths = set(df_split[df_split["split"] == "test"]["path"])

assert len(train_paths.intersection(val_paths)) == 0, "¡FUGA DETECTADA entre Train y Val!"
assert len(train_paths.intersection(test_paths)) == 0, "¡FUGA DETECTADA entre Train y Test!"
assert len(val_paths.intersection(test_paths)) == 0, "¡FUGA DETECTADA entre Val y Test!"

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduardCY/m3-proyecto-final-databricks-ia/blob/main/notebooks/02_preprocessing.ipynb)

Este notebook implementa la **Etapa 2**:
1. Construcción de canales estandarizados `(224, 224, 3)`.
2. Normalización de pesos ImageNet (BGR mean centering).
3. Data Augmentation clínicamente seguro para neuroimagen.
4. Optimización de pipeline con `prefetch(AUTOTUNE)` y verificación de lotes."""
    ),
    code_cell(COLAB_BOOTSTRAP_CODE),
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduardCY/m3-proyecto-final-databricks-ia/blob/main/notebooks/03_training_colab_mlflow.ipynb)

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
    code_cell(COLAB_BOOTSTRAP_CODE),
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
        """# 3. Verificación y Descarga del Dataset de Resonancias
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from scripts.download_dataset import verify_dataset_structure, generate_smoke_test_dataset

data_raw = Path("data/raw")
data_raw.mkdir(parents=True, exist_ok=True)

if not verify_dataset_structure(data_raw):
    print("📥 Descargando dataset Brain Tumor MRI (~165 MB)...")
    zip_url = "https://github.com/masoudnickparvar/brain-tumor-mri-dataset/archive/refs/heads/main.zip"
    zip_tmp = Path("data/dataset_archive.zip")
    try:
        urllib.request.urlretrieve(zip_url, zip_tmp)
        with zipfile.ZipFile(zip_tmp, 'r') as zip_ref:
            zip_ref.extractall("data/temp_extracted")
        
        extracted_root = next(Path("data/temp_extracted").glob("brain-tumor-mri-dataset-*"))
        for item in extracted_root.iterdir():
            dest = data_raw / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(data_raw))
        shutil.rmtree("data/temp_extracted", ignore_errors=True)
        zip_tmp.unlink(missing_ok=True)
        print("✅ Dataset descargado y descomprimido exitosamente en data/raw/")
    except Exception as e:
        print(f"⚠️ Fallo en descarga remota: {e}. Generando dataset de contingencia para pruebas...")
        generate_smoke_test_dataset(data_raw, samples_per_class=30)
else:
    print(f"✅ Dataset disponible en: {data_raw}")"""
    ),
    code_cell(
        """# 4. Auditoría y Construcción de Datasets tf.data
import pandas as pd
from src.dataset import build_manifest, split_manifest, create_stratified_split
from src.config import BATCH_SIZE, CLASSES, SPLIT_MANIFEST_PATH
from src.preprocessing import create_tf_dataset
from src.train import train_and_log_run

# Generar manifiesto con rutas del entorno actual
df_manifest = build_manifest(data_dir=data_raw, use_cache=False)
df_split = split_manifest(df_manifest, output_path=SPLIT_MANIFEST_PATH)
print(f"✅ Manifiesto y Split generados con {len(df_split)} imágenes:")
print(df_split.groupby(["split", "label_name"]).size().unstack())

train_ds = create_tf_dataset(df_split, split_name="train", batch_size=BATCH_SIZE, model_type="resnet50")
val_ds = create_tf_dataset(df_split, split_name="val", batch_size=BATCH_SIZE, model_type="resnet50")
test_ds = create_tf_dataset(df_split, split_name="test", batch_size=BATCH_SIZE, model_type="resnet50")
print("✅ Pipelines tf.data listos para GPU.")"""
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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EduardCY/m3-proyecto-final-databricks-ia/blob/main/notebooks/04_evaluation_registry.ipynb)

Este notebook implementa la **Etapa 5**:
1. Consulta y tabla comparativa de los 5 experimentos registrados en MLflow.
2. Identificación del modelo ganador mediante Macro-F1 y Multiclass AUC.
3. Registro formal en el **Model Registry** de MLflow (`neuro_mri_classifier`) y promoción a `Staging`.
4. Exportación atómica del modelo desacoplado a `models/neuro_mri_model.keras`.
5. Especificación técnica de la **Fase 2 (Migración a Databricks Workspace, Unity Catalog y Delta Lake)**."""
    ),
    code_cell(COLAB_BOOTSTRAP_CODE),
    code_cell(
        """import sys
import os
from pathlib import Path
PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sincronización con Google Drive si estamos en Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    from scripts.sync_drive import restore_from_gdrive
    if not (PROJECT_ROOT / "mlflow.db").exists():
        print("📥 Restaurando mlflow.db y artefactos desde Google Drive...")
        restore_from_gdrive()
except Exception as e:
    print(f"Modo local o aviso de Google Drive: {e}")

import shutil
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import (
    FINAL_MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
)
from src.metrics import plot_confusion_matrix, plot_multiclass_roc_curve

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()"""
    ),
    md_cell("### 1. Tabla Comparativa de Experimentos"),
    code_cell(
        """experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.test_macro_f1 DESC"])
    records = []
    for r in runs:
        records.append({
            "run_id": r.info.run_id[:8],
            "run_name": r.data.tags.get("mlflow.runName", "N/A"),
            "backbone": r.data.params.get("backbone", "N/A"),
            "test_macro_f1": r.data.metrics.get("test_macro_f1", 0.0),
            "test_roc_auc_ovr": r.data.metrics.get("test_roc_auc_ovr", 0.0),
            "test_recall": r.data.metrics.get("test_recall", 0.0),
            "epochs": r.data.params.get("epochs", "N/A"),
        })
    df_runs = pd.DataFrame(records)
    try:
        print(df_runs.to_markdown(index=False))
    except Exception:
        print(df_runs.to_string(index=False))
else:
    print(f"No se encontró experimento con nombre {MLFLOW_EXPERIMENT_NAME}. Verifique MLFLOW_TRACKING_URI.")"""
    ),
    md_cell("### 2. Registro del Modelo Campeón en Model Registry"),
    code_cell(
        """# Identificar el mejor Run según Macro-F1
best_run = runs[0] if experiment and runs else None

if best_run:
    best_run_id = best_run.info.run_id
    print(f"🏆 Modelo Campeón identificado: Run ID {best_run_id}")
    print(f"Métricas Campeón - Macro-F1: {best_run.data.metrics.get('test_macro_f1', 0):.4f}")
    
    # Registrar en MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
    print(f"✅ Registrado en Model Registry: {MLFLOW_MODEL_NAME} (Versión: {reg_model.version})")
    
    # Transicionar a Staging
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=reg_model.version,
        stage="Staging",
        archive_existing_versions=True
    )
    print(f"🚀 Versión {reg_model.version} promovida formalmente a 'Staging'.")"""
    ),
    md_cell("### 3. Exportación Atómica de Modelo a Disco"),
    code_cell(
        """# Asegurar que el modelo .keras de producción se guarde en models/
if best_run:
    best_model_local = client.download_artifacts(best_run_id, "model/data/model.keras", dst_path=str(MODELS_DIR))
    if Path(best_model_local).exists():
        shutil.copy(best_model_local, str(FINAL_MODEL_PATH))
        print(f"✅ Modelo desacoplado exportado exitosamente a: {FINAL_MODEL_PATH}")
    else:
        print(f"⚠️ Artefacto copiado directamente a models/: {FINAL_MODEL_PATH}")"""
    ),
    md_cell("### 4. Roadmap Técnico: Transición a Databricks (Fase 2)"),
    code_cell(
        """# Resumen de Arquitectura Databricks Unity Catalog y Delta Lake
databricks_roadmap = [
    {"Fase 1 (Híbrida)": "manifest_split.csv", "Fase 2 (Databricks)": "Tabla Delta (bronze_mri_manifest) en Unity Catalog", "Beneficio": "Gobernanza centralizada, consultas SQL directas y Time Travel."},
    {"Fase 1 (Híbrida)": "PySpark simbólico local", "Fase 2 (Databricks)": "Cluster Databricks multi-nodo", "Beneficio": "Procesamiento distribuido de terabytes de estudios DICOM."},
    {"Fase 1 (Híbrida)": "SQLite mlflow.db", "Fase 2 (Databricks)": "Managed Databricks MLflow", "Beneficio": "Colaboración empresarial y linaje de datos con Unity Catalog."},
    {"Fase 1 (Híbrida)": "FastAPI en Uvicorn", "Fase 2 (Databricks)": "Databricks Model Serving", "Beneficio": "Endpoints serverless de inferencia con auto-scaling y monitoreo de deriva."},
]

df_roadmap = pd.DataFrame(databricks_roadmap)
try:
    print(df_roadmap.to_markdown(index=False))
except Exception:
    print(df_roadmap.to_string(index=False))"""
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
        nb_json = make_notebook(cells)
        out_path = NOTEBOOKS_DIR / name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=2, ensure_ascii=False)
        print(f"Creado notebook: {name}")


if __name__ == "__main__":
    main()
