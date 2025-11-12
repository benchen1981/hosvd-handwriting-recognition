"""
高級示例 - 展示HOSVD系統的各種功能
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加項目路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load_data, DataPreprocessor, augment_data
from models import HOSVDModel, ClassifierPipeline, EnsembleClassifier
from utils import (
    Metrics, ModelEvaluator, FileManager, Logger,
    plot_digits, plot_confusion_matrix, plot_classification_metrics,
    compare_classifiers
)


def example_1_basic_workflow():
    """示例1: 基本工作流程"""
    print("\n" + "="*80)
    print("示例1: 基本工作流程")
    print("="*80)
    
    # 加載數據
    X_train, y_train, X_test, y_test = load_data('mnist', normalize=True)
    print(f"✓ 數據已加載: Train {X_train.shape}, Test {X_test.shape}")
    
    # 應用HOSVD
    hosvd = HOSVDModel(n_components=50)
    X_train_reduced = hosvd.fit_transform(X_train)
    X_test_reduced = hosvd.transform(X_test)
    print(f"✓ HOSVD降維完成: {X_train.shape[1]} -> {X_train_reduced.shape[1]}")
    
    # 訓練分類器
    classifier = ClassifierPipeline('knn', n_neighbors=5)
    classifier.fit(X_train_reduced, y_train)
    
    # 評估
    accuracy = classifier.score(X_test_reduced, y_test)
    print(f"✓ 測試精度: {accuracy:.4f}")


def example_2_classifier_comparison():
    """示例2: 分類器比較"""
    print("\n" + "="*80)
    print("示例2: 分類器比較")
    print("="*80)
    
    X_train, y_train, X_test, y_test = load_data('mnist', normalize=True)
    
    # HOSVD降維
    hosvd = HOSVDModel(n_components=50)
    X_train_reduced = hosvd.fit_transform(X_train)
    X_test_reduced = hosvd.transform(X_test)
    
    # 比較分類器
    classifiers = ['knn', 'svm', 'rf']
    results = {}
    
    for clf_type in classifiers:
        print(f"\n訓練 {clf_type.upper()}...")
        clf = ClassifierPipeline(clf_type)
        clf.fit(X_train_reduced, y_train)
        y_pred = clf.predict(X_test_reduced)
        
        results[clf_type] = {
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        acc = (y_pred == y_test).mean()
        print(f"  精度: {acc:.4f}")
    
    # 比較結果
    comparison = compare_classifiers(results)
    print("\n✓ 分類器比較完成")


def example_3_parameter_tuning():
    """示例3: 參數調優"""
    print("\n" + "="*80)
    print("示例3: 參數調優 - 主成分數的影響")
    print("="*80)
    
    X_train, y_train, X_test, y_test = load_data('digits', normalize=True)
    
    components = [5, 10, 20, 30, 40, 50]
    results = []
    
    for n_comp in components:
        print(f"\n測試 n_components={n_comp}...", end=' ')
        
        hosvd = HOSVDModel(n_components=n_comp)
        X_train_r = hosvd.fit_transform(X_train)
        X_test_r = hosvd.transform(X_test)
        
        clf = ClassifierPipeline('knn', n_neighbors=5)
        clf.fit(X_train_r, y_train)
        acc = clf.score(X_test_r, y_test)
        
        results.append((n_comp, acc))
        print(f"精度: {acc:.4f}")
    
    print("\n✓ 參數調優完成")
    
    # 找最佳值
    best_n, best_acc = max(results, key=lambda x: x[1])
    print(f"\n最佳配置: n_components={best_n}, 精度={best_acc:.4f}")


def example_4_ensemble_learning():
    """示例4: 集成學習"""
    print("\n" + "="*80)
    print("示例4: 集成學習")
    print("="*80)
    
    X_train, y_train, X_test, y_test = load_data('digits', normalize=True)
    
    # HOSVD降維
    hosvd = HOSVDModel(n_components=50)
    X_train_reduced = hosvd.fit_transform(X_train)
    X_test_reduced = hosvd.transform(X_test)
    
    # 創建集成分類器
    ensemble = EnsembleClassifier()
    
    print("\n訓練集成分類器...")
    classifiers = [
        (ClassifierPipeline('knn', n_neighbors=5), 1.0),
        (ClassifierPipeline('svm', kernel='rbf'), 0.8),
        (ClassifierPipeline('rf', n_estimators=50), 0.9),
    ]
    
    for clf, weight in classifiers:
        print(f"  添加分類器 {clf.classifier_type} (權重={weight})")
        ensemble.add_classifier(clf, weight)
    
    ensemble.fit(X_train_reduced, y_train)
    
    # 評估
    acc_ensemble = ensemble.score(X_test_reduced, y_test)
    print(f"\n✓ 集成精度: {acc_ensemble:.4f}")


def example_5_advanced_analysis():
    """示例5: 高級分析"""
    print("\n" + "="*80)
    print("示例5: 高級分析")
    print("="*80)
    
    X_train, y_train, X_test, y_test = load_data('digits', normalize=True)
    
    # HOSVD分析
    print("\n1. HOSVD分析")
    hosvd = HOSVDModel(n_components=30)
    X_train_reduced = hosvd.fit_transform(X_train)
    X_test_reduced = hosvd.transform(X_test)
    
    print(f"  原始維度: {X_train.shape[1]}")
    print(f"  降維後維度: {X_train_reduced.shape[1]}")
    print(f"  壓縮比: {hosvd.get_compression_ratio():.4f}")
    print(f"  核心張量形狀: {hosvd.get_core_tensor_shape()}")
    
    # 分類與評估
    print("\n2. 分類與評估")
    clf = ClassifierPipeline('svm', kernel='rbf')
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)
    
    evaluator = ModelEvaluator(y_test, y_pred)
    metrics = evaluator.get_metrics()
    
    print(f"  精度: {metrics['accuracy']:.4f}")
    print(f"  精度: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分數: {metrics['f1']:.4f}")
    
    # 錯誤分析
    print("\n3. 錯誤分析")
    errors = y_pred != y_test
    error_rate = errors.sum() / len(y_test) * 100
    print(f"  錯誤數: {errors.sum()}")
    print(f"  錯誤率: {error_rate:.2f}%")
    
    print("\n✓ 高級分析完成")


def example_6_custom_workflow():
    """示例6: 自定義工作流程"""
    print("\n" + "="*80)
    print("示例6: 自定義工作流程")
    print("="*80)
    
    print("\n1. 加載並預處理數據")
    X_train, y_train, X_test, y_test = load_data('mnist', normalize=True)
    
    preprocessor = DataPreprocessor(normalize=True, standardize=True)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print("  ✓ 數據已預處理")
    
    print("\n2. 應用多階段HOSVD")
    # 第一階段: 粗降維
    hosvd1 = HOSVDModel(n_components=100)
    X_train = hosvd1.fit_transform(X_train)
    X_test = hosvd1.transform(X_test)
    print(f"  ✓ 階段1完成: {X_train.shape}")
    
    # 第二階段: 細降維
    hosvd2 = HOSVDModel(n_components=50)
    X_train = hosvd2.fit_transform(X_train)
    X_test = hosvd2.transform(X_test)
    print(f"  ✓ 階段2完成: {X_train.shape}")
    
    print("\n3. 訓練並評估")
    clf = ClassifierPipeline('rf', n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"  ✓ 最終精度: {accuracy:.4f}")


def main():
    """運行所有示例"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "HOSVD 手寫辨識系統 - 高級示例" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # 運行示例
        example_1_basic_workflow()
        example_2_classifier_comparison()
        example_3_parameter_tuning()
        example_4_ensemble_learning()
        example_5_advanced_analysis()
        example_6_custom_workflow()
        
        print("\n" + "="*80)
        print("✓ 所有示例運行完成！")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
