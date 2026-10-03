"""
Pruebas automatizadas del módulo de métricas clínicas y artefactos de evaluación.
"""

from pathlib import Path
import numpy as np
import pytest

from src.config import CLASSES
from src.metrics import (
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_multiclass_roc_curve,
    plot_training_history,
)


@pytest.fixture
def dummy_predictions():
    """Genera etiquetas reales y probabilidades sintéticas para 4 clases."""
    num_samples = 40
    num_classes = len(CLASSES)
    rng = np.random.default_rng(42)

    # 10 muestras por cada una de las 4 clases
    y_true = np.array([i for i in range(num_classes) for _ in range(num_samples // num_classes)])

    # Generar probabilidades realistas con ruido
    logits = rng.normal(size=(num_samples, num_classes))
    for i in range(num_samples):
        logits[i, y_true[i]] += 2.0  # Sesgo positivo hacia la clase correcta

    # Softmax
    exp_logits = np.exp(logits)
    y_prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return y_true, y_prob


def test_compute_comprehensive_metrics(dummy_predictions):
    """Verifica el cálculo de Macro-F1, ROC-AUC, Recall y generación del classification report."""
    y_true, y_prob = dummy_predictions
    metrics, report_str = compute_comprehensive_metrics(y_true, y_prob, classes=CLASSES)

    assert "test_macro_f1" in metrics
    assert "test_weighted_f1" in metrics
    assert "test_macro_precision" in metrics
    assert "test_macro_recall" in metrics
    assert "test_multiclass_auc" in metrics

    assert 0.0 <= metrics["test_macro_f1"] <= 1.0
    assert 0.0 <= metrics["test_multiclass_auc"] <= 1.0
    assert all(cls_name in report_str for cls_name in CLASSES)


def test_plot_artifacts(tmp_path: Path, dummy_predictions):
    """Verifica que la generación y guardado de matriz de confusión y curvas ROC no fallen."""
    y_true, y_prob = dummy_predictions
    y_pred = np.argmax(y_prob, axis=1)
    y_true_onehot = np.eye(len(CLASSES))[y_true]

    cm_path = tmp_path / "test_cm.png"
    roc_path = tmp_path / "test_roc.png"

    plot_confusion_matrix(y_true, y_pred, output_path=cm_path, classes=CLASSES)
    plot_multiclass_roc_curve(y_true_onehot, y_prob, output_path=roc_path, classes=CLASSES)

    assert cm_path.exists()
    assert cm_path.stat().st_size > 1000
    assert roc_path.exists()
    assert roc_path.stat().st_size > 1000
