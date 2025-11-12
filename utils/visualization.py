"""
可視化工具模塊
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def plot_digits(images, labels=None, n_rows=5, n_cols=5, figsize=(10, 10), title=""):
    """
    繪製手寫數字
    
    Args:
        images: 圖像數組，形狀(n_samples, 784)或(n_samples, 28, 28)
        labels: 標籤
        n_rows, n_cols: 網格大小
        figsize: 圖形大小
        title: 標題
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # 重塑圖像
    if len(images.shape) == 2:
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(-1, side, side)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            if labels is not None:
                ax.set_title(f"Label: {labels[i]}", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, labels=None, figsize=(10, 8)):
    """
    繪製混淆矩陣
    
    Args:
        cm: 混淆矩陣
        labels: 類別標籤
        figsize: 圖形大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_classification_metrics(metrics, figsize=(12, 5)):
    """
    繪製分類指標
    
    Args:
        metrics: 指標字典 {'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ...}
        figsize: 圖形大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 分類器對比
    if isinstance(metrics, dict) and all(isinstance(v, dict) for v in metrics.values()):
        classifiers = list(metrics.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        
        x = np.arange(len(classifiers))
        width = 0.2
        
        for i, metric in enumerate(metric_names):
            values = [metrics[clf].get(metric, 0) for clf in classifiers]
            ax1.bar(x + i * width, values, width, label=metric)
        
        ax1.set_xlabel('Classifier', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Classification Metrics by Classifier', fontsize=12)
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(classifiers)
        ax1.legend()
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)
    
    # 單個分類器的詳細指標
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        metric_names = list(metrics.keys())
        values = list(metrics.values())
        
        ax2.barh(metric_names, values, color='skyblue')
        ax2.set_xlabel('Score', fontsize=12)
        ax2.set_title('Classification Metrics', fontsize=12)
        ax2.set_xlim([0, 1.1])
        
        for i, v in enumerate(values):
            ax2.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_dimensionality_reduction(X_original, X_reduced, labels=None, figsize=(14, 5)):
    """
    繪製降維前後的數據分佈（使用PCA投影）
    
    Args:
        X_original: 原始數據
        X_reduced: 降維後的數據
        labels: 類別標籤
        figsize: 圖形大小
    """
    from sklearn.decomposition import PCA
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 原始數據的PCA投影
    pca_orig = PCA(n_components=2)
    X_orig_2d = pca_orig.fit_transform(X_original)
    
    scatter1 = ax1.scatter(X_orig_2d[:, 0], X_orig_2d[:, 1], c=labels, cmap='tab10', s=30, alpha=0.6)
    ax1.set_xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.2%})', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.2%})', fontsize=11)
    ax1.set_title('Original Data (PCA projection)', fontsize=12)
    plt.colorbar(scatter1, ax=ax1, label='Class')
    ax1.grid(True, alpha=0.3)
    
    # 降維後數據的PCA投影
    pca_reduced = PCA(n_components=2)
    X_reduced_2d = pca_reduced.fit_transform(X_reduced)
    
    scatter2 = ax2.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], c=labels, cmap='tab10', s=30, alpha=0.6)
    ax2.set_xlabel(f'PC1 ({pca_reduced.explained_variance_ratio_[0]:.2%})', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca_reduced.explained_variance_ratio_[1]:.2%})', fontsize=11)
    ax2.set_title('Reduced Data (PCA projection)', fontsize=12)
    plt.colorbar(scatter2, ax=ax2, label='Class')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_explained_variance(explained_variance_ratio, figsize=(10, 6)):
    """
    繪製解釋方差比
    
    Args:
        explained_variance_ratio: 解釋方差比數組
        figsize: 圖形大小
    """
    cumsum = np.cumsum(explained_variance_ratio)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, 'bo-', label='Individual')
    ax.plot(range(1, len(cumsum) + 1), 
            cumsum, 'rs-', label='Cumulative')
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('Explained Variance by Principal Component', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_history(history, figsize=(14, 5)):
    """
    繪製訓練歷史
    
    Args:
        history: 歷史字典 {'loss': [...], 'val_loss': [...], ...}
        figsize: 圖形大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 損失曲線
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 精度曲線
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_proba, figsize=(10, 8)):
    """
    繪製ROC曲線
    
    Args:
        y_true: 真實標籤
        y_proba: 預測概率
        figsize: 圖形大小
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 二分類ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic', fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 測試
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X = digits.data[:100]
    y = digits.target[:100]
    
    fig = plot_digits(X, y, title="Sample Handwritten Digits")
    plt.show()
