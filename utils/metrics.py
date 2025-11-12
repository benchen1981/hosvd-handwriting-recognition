"""
評估指標模塊
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """評估指標類"""
    
    @staticmethod
    def compute_metrics(y_true, y_pred, y_proba=None, average='weighted'):
        """
        計算多項分類指標
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_proba: 預測概率（用於計算AUC）
            average: 'weighted', 'macro', 'micro'
        
        Returns:
            指標字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        }
        
        # 計算AUC（對於二分類）
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['auc'] = None
        
        return metrics
    
    @staticmethod
    def per_class_metrics(y_true, y_pred):
        """
        計算每個類別的指標
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
        
        Returns:
            報告字符串
        """
        return classification_report(y_true, y_pred)
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """
        獲取混淆矩陣
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
        
        Returns:
            混淆矩陣
        """
        return confusion_matrix(y_true, y_pred)


class ModelEvaluator:
    """模型評估器"""
    
    def __init__(self, y_true, y_pred, y_proba=None):
        """
        初始化
        
        Args:
            y_true: 真實標籤
            y_pred: 預測標籤
            y_proba: 預測概率
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.metrics = Metrics.compute_metrics(y_true, y_pred, y_proba)
        self.cm = Metrics.get_confusion_matrix(y_true, y_pred)
    
    def get_metrics(self):
        """獲取指標"""
        return self.metrics
    
    def get_confusion_matrix(self):
        """獲取混淆矩陣"""
        return self.cm
    
    def print_report(self):
        """打印報告"""
        logger.info("Classification Report:")
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        logger.info(f"F1-Score: {self.metrics['f1']:.4f}")
        
        if 'auc' in self.metrics and self.metrics['auc'] is not None:
            logger.info(f"AUC: {self.metrics['auc']:.4f}")
        
        logger.info("\nPer-class metrics:")
        logger.info(Metrics.per_class_metrics(self.y_true, self.y_pred))
    
    def summary(self):
        """返回摘要字典"""
        return {
            'metrics': self.metrics,
            'confusion_matrix': self.cm,
            'report': Metrics.per_class_metrics(self.y_true, self.y_pred)
        }


def compare_classifiers(results_dict):
    """
    比較多個分類器
    
    Args:
        results_dict: {'classifier_name': {'y_true': ..., 'y_pred': ...}, ...}
    
    Returns:
        比較結果字典
    """
    comparison = {}
    
    for name, result in results_dict.items():
        y_true = result['y_true']
        y_pred = result['y_pred']
        y_proba = result.get('y_proba', None)
        
        evaluator = ModelEvaluator(y_true, y_pred, y_proba)
        comparison[name] = evaluator.get_metrics()
        
        logger.info(f"\n{name}:")
        evaluator.print_report()
    
    return comparison


def compute_per_class_accuracy(cm):
    """
    從混淆矩陣計算每類精度
    
    Args:
        cm: 混淆矩陣
    
    Returns:
        每類精度數組
    """
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc


def compute_per_class_metrics(y_true, y_pred):
    """
    計算每類的詳細指標
    
    Args:
        y_true: 真實標籤
        y_pred: 預測標籤
    
    Returns:
        詳細指標字典
    """
    classes = np.unique(y_true)
    results = {}
    
    for cls in classes:
        mask = y_true == cls
        precision = precision_score(y_true, y_pred, labels=[cls], zero_division=0)[0]
        recall = recall_score(y_true, y_pred, labels=[cls], zero_division=0)[0]
        f1 = f1_score(y_true, y_pred, labels=[cls], zero_division=0)[0]
        
        results[f'Class {cls}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': mask.sum()
        }
    
    return results


if __name__ == "__main__":
    # 測試
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2])
    
    metrics = Metrics.compute_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    cm = Metrics.get_confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    
    per_class = compute_per_class_metrics(y_true, y_pred)
    print("Per-class metrics:", per_class)
