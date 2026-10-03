"""
Pruebas automatizadas de la Etapa 1: Ingesta, Auditoría de Integridad y Partición de Datos.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.config import CLASSES, SPLIT_RATIOS
from src.dataset import build_manifest, split_manifest, verify_image_integrity


@pytest.fixture
def mock_dataset_dir(tmp_path: Path) -> Path:
    """Crea una estructura temporal con imágenes mock válidas y una corrupta."""
    data_dir = tmp_path / "mock_mri"
    for split in ["Training", "Testing"]:
        for cls_name in CLASSES:
            folder = data_dir / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img_path = folder / f"scan_{cls_name}_{i:02d}.jpg"
                arr = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr).save(img_path)

    # Introducir deliberadamente una imagen corrupta (0 bytes)
    corrupted_path = data_dir / "Training" / CLASSES[0] / "corrupt_scan.jpg"
    with open(corrupted_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_CONTENT")

    return data_dir


def test_verify_image_integrity(mock_dataset_dir: Path):
    """Verifica que el detector de integridad identifique correctamente archivos corruptos."""
    valid_img = next(mock_dataset_dir.rglob("scan_*.jpg"))
    corrupt_img = next(mock_dataset_dir.rglob("corrupt_*.jpg"))

    assert verify_image_integrity(valid_img) is True
    assert verify_image_integrity(corrupt_img) is False


def test_build_manifest(mock_dataset_dir: Path):
    """Verifica que build_manifest cargue todas las clases y descarte corruptos."""
    df = build_manifest(mock_dataset_dir, use_cache=False)

    assert not df.empty
    assert set(df["label_name"].unique()) == set(CLASSES)
    # Debe haber descartado el archivo corrupto
    assert "corrupt_scan.jpg" not in df["filename"].values
    assert all(col in df.columns for col in ["path", "filename", "label", "label_name", "split" if "split" in df.columns else "file_size_bytes"])


def test_split_manifest_no_leakage(mock_dataset_dir: Path):
    """Verifica que el split no tenga fuga de datos (intersección nula entre conjuntos)."""
    df = build_manifest(mock_dataset_dir, use_cache=False)
    split_df = split_manifest(df, split_ratios=SPLIT_RATIOS, seed=42)

    train_paths = set(split_df[split_df["split"] == "train"]["path"])
    val_paths = set(split_df[split_df["split"] == "val"]["path"])
    test_paths = set(split_df[split_df["split"] == "test"]["path"])

    # Failsafe: Comprobar intersección vacía (cero fuga de datos)
    assert len(train_paths.intersection(val_paths)) == 0, "Fuga de datos detectada entre Train y Val"
    assert len(train_paths.intersection(test_paths)) == 0, "Fuga de datos detectada entre Train y Test"
    assert len(val_paths.intersection(test_paths)) == 0, "Fuga de datos detectada entre Val y Test"

    # Comprobar que todas las clases existan en todos los splits
    for s in ["train", "val", "test"]:
        classes_in_split = set(split_df[split_df["split"] == s]["label_name"])
        assert classes_in_split == set(CLASSES), f"El split {s} no contiene todas las clases requeridas"
