"""
Métricas de evaluación clínica y generación de artefactos visuales para MLflow.

Calcula Macro-F1, ROC-AUC Multiclase One-vs-Rest, matriz de confusión y curvas
de convergencia de entrenamiento y validación.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASSES


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    classes: list[str] = CLASSES,
) -> None:
    """Genera y guarda la matriz de confusión normalizada en formato gráfico."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.title("Matriz de Confusión Normalizada (NeuroScan AI)")
    plt.xlabel("Clase Predicha por el Modelo")
    plt.ylabel("Clase Real (Diagnóstico Ground Truth)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_multiclass_roc_curve(
    y_true_onehot: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    classes: list[str] = CLASSES,
) -> None:
    """Genera curvas ROC One-vs-Rest para cada una de las 4 patologías."""
    plt.figure(figsize=(8, 6))
    for i, cls_name in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_true_onehot[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{cls_name} (AUC = {auc_val:.3f})")
        except Exception:
            plt.plot([0, 1], [0, 1], ":", label=f"{cls_name} (AUC N/A)")

    plt.plot([0, 1], [0, 1], "k--", label="Clasificador Azar")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)")
    plt.title("Curvas ROC Multiclase (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_history(history: Any, output_path: Path) -> None:
    """Grafica y guarda las curvas de pérdida y exactitud por época."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Curvas de Loss
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_title("Evolución de la Función de Pérdida")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Curvas de Accuracy
    ax2.plot(history.history["accuracy"], label="Train Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax2.set_title("Evolución de la Exactitud")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str] = CLASSES,
) -> tuple[dict[str, float], str]:
    """Calcula todas las métricas numéricas y genera el classification report.

    Returns:
        Tupla con (diccionario de métricas para MLflow, reporte en texto).
    """
    y_pred = np.argmax(y_prob, axis=1)
    y_true_onehot = np.eye(len(classes))[y_true]

    try:
        multiclass_auc = float(roc_auc_score(y_true_onehot, y_prob, multi_class="ovr"))
    except Exception:
        multiclass_auc = 0.0

    metrics = {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_multiclass_auc": multiclass_auc,
    }

    report_str = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    return metrics, report_str

