"""
🎓 主程式 - HOSVD 手寫數字識別系統
============================================
想像這個程式是一個完整的實驗室工作流:
1. 拿出所有工具和材料 (準備數據)
2. 進行實驗 (訓練模型)
3. 測試結果 (評估模型)
4. 拍照記錄 (生成圖表)
5. 寫下來報告 (保存結果)

作者: 陳宥興 (5114050015)
"""

# ==================== 第1步: 準備所有工具 ====================
# 就像實驗開始前要準備各種儀器和試劑

import os              # 文件和路徑管理
import sys              # 系統功能
import argparse        # 命令行參數解析
import numpy as np     # 數學計算
import matplotlib.pyplot as plt  # 畫圖工具

from datetime import datetime  # 時間功能

# 導入我們自己寫的模組
from config import DATA_CONFIG, HOSVD_CONFIG, CLASSIFIER_CONFIG, PATH_CONFIG, VIZ_CONFIG  # 配置
from data import load_data, DataPreprocessor  # 數據載入和準備
from models import HOSVDModel, ClassifierPipeline  # 機器學習模型
from utils import (
    Metrics,                    # 評估指標計算
    ModelEvaluator,             # 模型評估器
    FileManager,                # 文件管理
    Logger,                     # 日誌記錄
    plot_digits,                # 畫數字函數
    plot_confusion_matrix,      # 畫混淆矩陣
    plot_classification_metrics, # 畫分類指標
    plot_dimensionality_reduction # 畫降維圖
)


# ==================== 第2步: 設置日誌 ====================
# 日誌就像記者記錄整個實驗過程

def setup_logging():
    """
    設置日誌系統。
    
    想像過程:
    就像開始寫日記,記錄所有發生的事情
    """
    logger = Logger.setup_logger("HOSVD_System")  # 建立日誌記錄器
    return logger


# ==================== 第3步: 創建輸出目錄 ====================
# 就像做實驗前要準備好放結果的文件夾

def create_directories():
    """
    創建所有必要的輸出目錄。
    
    想像過程:
    1. 看看需要哪些文件夾
    2. 如果文件夾不存在,就建立它
    3. 準備好放結果
    """
    for path in PATH_CONFIG.values():  # 對每個路徑
        os.makedirs(path, exist_ok=True)  # 建立文件夾 (如果不存在)


# ==================== 第4步: 加載和準備數據 ====================
# 就像做菜先要準備食材

def load_and_preprocess_data(dataset='mnist', normalize=True):
    """
    加載和預處理數據。
    
    想像過程:
    1. 打開包含數據的文件 (像打開食譜)
    2. 檢查數據的大小 (像稱量食材)
    3. 清理和整理數據 (像洗菜和切菜)
    4. 測試數據質量 (像嚐味道)
    
    參數:
        dataset: 數據集的名字 ('mnist' 是手寫數字圖片)
        normalize: 是否要把數據標準化到 0~1 之間
    
    回傳:
        4 個東西: 訓練圖片、訓練標籤、測試圖片、測試標籤
    """
    logger = Logger.setup_logger(__name__)  # 建立日誌記錄器
    
    # 打開數據 (就像打開食材包)
    logger.info(f"Loading {dataset} dataset...")
    X_train, y_train, X_test, y_test = load_data(dataset, normalize=normalize)
    
    # 顯示數據的大小
    logger.info(f"Original shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    # 例如: Original shapes - Train: (60000, 784), Test: (10000, 784)
    # 意思是: 60000 張訓練圖片,10000 張測試圖片,每張 784 個像素
    
    # 初始化預處理工具 (就像準備烹飪工具)
    preprocessor = DataPreprocessor(normalize=normalize, standardize=False)
    
    # 對訓練數據進行預處理 (就像洗菜)
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # 對測試數據進行預處理 (就像再洗一遍菜)
    X_test_processed = preprocessor.transform(X_test)
    
    # 顯示預處理後的大小
    logger.info(f"Preprocessed shapes - Train: {X_train_processed.shape}, Test: {X_test_processed.shape}")
    
    # 回傳所有東西
    return X_train_processed, y_train, X_test_processed, y_test


# ==================== 第5步: 應用 HOSVD 降維 ====================
# 就像把菜的營養壓縮到更小的空間

def apply_hosvd(X_train, X_test, n_components=50):
    """
    應用 HOSVD 演算法進行降維。
    
    想像過程:
    1. 建立 HOSVD 工具 (就像買一個壓縮機)
    2. 訓練這個工具 (就像學會如何操作)
    3. 對訓練數據壓縮 (就像壓縮訓練菜)
    4. 對測試數據壓縮 (就像壓縮測試菜)
    5. 檢查壓縮率 (就像看能省多少空間)
    
    參數:
        X_train: 訓練圖片數據
        X_test: 測試圖片數據
        n_components: 要保留多少個主要特徵 (50 = 保留 50 個最重要的特徵)
    
    回傳:
        壓縮後的訓練數據、壓縮後的測試數據、HOSVD 模型
    """
    logger = Logger.setup_logger(__name__)
    
    # 告訴用戶正在進行降維
    logger.info(f"Applying HOSVD with {n_components} components...")
    
    # 建立 HOSVD 模型 (就像建立壓縮機)
    hosvd = HOSVDModel(n_components=n_components)
    
    # 訓練和壓縮訓練數據 (就像壓縮訓練菜)
    X_train_reduced = hosvd.fit_transform(X_train)
    
    # 用已學會的方法壓縮測試數據 (就像用同樣的方法壓縮測試菜)
    X_test_reduced = hosvd.transform(X_test)
    
    # 顯示壓縮後的大小
    logger.info(f"Reduced shapes - Train: {X_train_reduced.shape}, Test: {X_test_reduced.shape}")
    # 例如: Reduced shapes - Train: (60000, 50), Test: (10000, 50)
    # 意思是: 從 784 個特徵降到 50 個!
    
    # 顯示核心張量的大小
    logger.info(f"Core tensor shape: {hosvd.get_core_tensor_shape()}")
    
    # 顯示壓縮率 (能節省多少空間)
    logger.info(f"Compression ratio: {hosvd.get_compression_ratio():.4f}")
    # 例如: 0.0638 = 壓縮到原來大小的 6.38%
    
    # 把模型保存到文件 (就像備份壓縮機的配置)
    model_path = os.path.join(PATH_CONFIG['model_dir'], 'hosvd_model.pkl')
    FileManager.save_model(hosvd, model_path)
    
    return X_train_reduced, X_test_reduced, hosvd


# ==================== 第6步: 訓練分類器 ====================
# 就像訓練一個廚師認出菜的名字

def train_classifier(X_train, y_train, classifier_type='knn', **kwargs):
    """
    訓練分類器。
    
    想像過程:
    1. 選擇一個分類器 (就像選擇一個學生)
    2. 給他看訓練圖片和標籤 (就像教他認出菜)
    3. 學生重複學習直到能正確預測 (就像反複練習)
    4. 測試學生在訓練集上的準確率
    5. 保存這個訓練好的學生 (就像記錄他的知識)
    
    參數:
        X_train: 訓練圖片 (壓縮後)
        y_train: 訓練標籤 (正確答案)
        classifier_type: 分類器類型 ('knn', 'svm', 'rf', 或 'mlp')
        **kwargs: 分類器的參數設定
    
    回傳:
        訓練好的分類器
    """
    logger = Logger.setup_logger(__name__)
    
    # 告訴用戶正在訓練
    logger.info(f"Training {classifier_type} classifier...")
    
    # 建立分類器 (就像招聘一個新學生)
    classifier = ClassifierPipeline(classifier_type, **kwargs)
    
    # 訓練分類器 (就像教他)
    classifier.fit(X_train, y_train)
    
    # 測試分類器在訓練集上的準確率
    train_accuracy = classifier.score(X_train, y_train)
    logger.info(f"Train accuracy: {train_accuracy:.4f}")
    # 例如: Train accuracy: 0.9752 = 97.52% 的準確率
    
    # 保存訓練好的分類器 (就像備份他的知識)
    model_path = os.path.join(PATH_CONFIG['model_dir'], f'{classifier_type}_classifier.pkl')
    FileManager.save_model(classifier, model_path)
    
    return classifier


# ==================== 第7步: 評估模型 ====================
# 就像用測試題考學生

def evaluate_model(classifier, X_test, y_test, dataset_name=""):
    """
    評估模型的性能。
    
    想像過程:
    1. 用測試數據考學生 (沒看過的題目)
    2. 計算他的準確率、精確度、召回率等
    3. 製作混淆矩陣 (看他容易把哪些數字搞混)
    4. 記錄所有結果
    
    參數:
        classifier: 訓練好的分類器
        X_test: 測試圖片 (壓縮後)
        y_test: 測試標籤 (正確答案)
        dataset_name: 數據集的名字 (用於顯示)
    
    回傳:
        評估結果 (包含所有指標和混淆矩陣)
    """
    logger = Logger.setup_logger(__name__)
    
    # 告訴用戶正在評估
    logger.info(f"\nEvaluating model on {dataset_name}...")
    
    # 預測 (學生回答所有測試題)
    y_pred = classifier.predict(X_test)
    
    # 建立評估器 (就像評分老師)
    evaluator = ModelEvaluator(y_test, y_pred)
    
    # 計算所有評估指標
    metrics = evaluator.get_metrics()
    
    # 顯示結果
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    
    return evaluator


# ==================== 第8步: 生成圖表 ====================
# 就像把實驗結果畫成圖表

def generate_visualizations(X_train, y_train, X_test, y_test, X_test_reduced, 
                          evaluator, classifier_type, output_dir):
    """
    生成各種圖表和圖片。
    
    想像過程:
    1. 畫出一些樣本數字 (看看數據長什麼樣)
    2. 畫混淆矩陣 (看模型容易出錯的地方)
    3. 畫降維效果 (看壓縮後的效果)
    4. 畫分類指標 (看性能指標)
    5. 保存所有圖片
    
    參數:
        X_train, y_train: 訓練圖片和標籤
        X_test, y_test: 測試圖片和標籤
        X_test_reduced: 壓縮後的測試圖片
        evaluator: 評估器 (含評估結果)
        classifier_type: 分類器類型 (用於命名文件)
        output_dir: 輸出圖片的文件夾
    """
    logger = Logger.setup_logger(__name__)
    
    logger.info("Generating visualizations...")  # 告訴用戶正在生成圖表
    
    # 用時間戳作為文件名 (每次運行都不同,不會覆蓋)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 例如: 20250103_143022
    
    # ====== 圖表 1: 樣本數字 ======
    # 就像展示你收集的數據樣本
    fig = plot_digits(X_test[:25], y_test[:25], n_rows=5, n_cols=5, 
                     title="Sample Test Digits")
    fig.savefig(os.path.join(output_dir, f'sample_digits_{timestamp}.png'), dpi=100)
    plt.close(fig)  # 關閉圖表,節省記憶體
    
    # ====== 圖表 2: 混淆矩陣 ======
    # 10x10 的表格,顯示模型把哪些數字搞混
    cm = evaluator.get_confusion_matrix()
    fig = plot_confusion_matrix(cm)
    fig.savefig(os.path.join(output_dir, f'confusion_matrix_{classifier_type}_{timestamp}.png'), dpi=100)
    plt.close(fig)
    
    # ====== 圖表 3: 降維效果 ======
    # 用 PCA 把降維後的數據畫成 2D 圖,看看分佈情況
    try:
        fig = plot_dimensionality_reduction(X_test, X_test_reduced, y_test)
        fig.savefig(os.path.join(output_dir, f'dimensionality_reduction_{timestamp}.png'), dpi=100)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate dimensionality reduction plot: {e}")
    
    # ====== 圖表 4: 分類指標 ======
    # 用柱狀圖顯示準確率、精確度等指標
    metrics = evaluator.get_metrics()
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = list(metrics.keys())       # 指標名字
    metric_values = list(metrics.values())    # 指標值
    
    # 畫柱狀圖
    ax.barh(metric_names, metric_values, color='skyblue')
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(f'Classification Metrics ({classifier_type})', fontsize=14)
    ax.set_xlim([0, 1.1])  # 限制 x 軸範圍 (0 到 1.1)
    
    # 在每個柱子上寫上數值
    for i, v in enumerate(metric_values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    ax.grid(axis='x', alpha=0.3)  # 加上網格線
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'metrics_{classifier_type}_{timestamp}.png'), dpi=100)
    plt.close(fig)
    
    logger.info(f"Visualizations saved to {output_dir}")


# ==================== 第9步: 主函數 ====================
# 這是整個程序的控制中心

def main(args):
    """
    主函數 - 協調整個實驗流程。
    
    想像過程:
    這就像一個實驗室主任,指揮所有步驟:
    1. 準備工作 (建立目錄,設置日誌)
    2. 第一步: 加載數據
    3. 第二步: 降維
    4. 第三步: 訓練模型
    5. 第四步: 評估模型
    6. 第五步: 生成圖表
    7. 第六步: 保存結果
    
    參數:
        args: 命令行參數 (例如使用哪個數據集、多少個組件等)
    """
    # ====== 準備工作 ======
    logger = setup_logging()  # 開始記錄
    create_directories()      # 建立輸出文件夾
    
    # 顯示標題和配置信息
    logger.info("=" * 80)
    logger.info("HOSVD Handwriting Recognition System")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  HOSVD components: {args.n_components}")
    logger.info(f"  Classifier: {args.classifier}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info("=" * 80)
    
    # ====== 第 1 步: 加載和準備數據 ======
    X_train, y_train, X_test, y_test = load_and_preprocess_data(
        dataset=args.dataset,
        normalize=True
    )
    
    # ====== 第 2 步: 應用 HOSVD 降維 ======
    X_train_reduced, X_test_reduced, hosvd_model = apply_hosvd(
        X_train, X_test,
        n_components=args.n_components
    )
    
    # ====== 第 3 步: 訓練分類器 ======
    classifier_kwargs = CLASSIFIER_CONFIG.get(args.classifier, {})
    classifier = train_classifier(
        X_train_reduced, y_train,
        classifier_type=args.classifier,
        **classifier_kwargs
    )
    
    # ====== 第 4 步: 評估模型 ======
    evaluator = evaluate_model(
        classifier, X_test_reduced, y_test,
        dataset_name=args.dataset
    )
    
    # ====== 第 5 步: 生成圖表 ======
    if args.visualize:  # 如果用戶要求生成圖表
        generate_visualizations(
            X_train, y_train, X_test, y_test, X_test_reduced,
            evaluator, args.classifier, PATH_CONFIG['figure_dir']
        )
    
    # ====== 第 6 步: 保存結果 ======
    results = {
        'timestamp': datetime.now().isoformat(),  # 時間戳
        'configuration': {  # 配置信息
            'dataset': args.dataset,
            'n_components': args.n_components,
            'classifier': args.classifier,
            'test_size': args.test_size,
        },
        'hosvd_info': {  # HOSVD 信息
            'core_tensor_shape': hosvd_model.get_core_tensor_shape(),
            'compression_ratio': float(hosvd_model.get_compression_ratio()),
        },
        'metrics': evaluator.get_metrics(),  # 評估指標
    }
    
    # 保存結果到 JSON 文件
    result_path = os.path.join(PATH_CONFIG['model_dir'], 'results.json')
    FileManager.save_json(results, result_path)
    
    # 顯示完成消息
    logger.info("=" * 80)
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to {result_path}")
    logger.info("=" * 80)
    
    return results  # 回傳結果


# ==================== 第 10 步: 命令行參數解析 ====================
# 讓用戶可以自訂程序的行為

if __name__ == "__main__":  # 只有直接運行這個文件時才執行
    # 建立命令行參數解析器 (就像寫使用說明書)
    parser = argparse.ArgumentParser(
        description="HOSVD Handwriting Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例子:
  # 基本用法
  python main.py
  
  # 使用 Fashion-MNIST 數據集
  python main.py --dataset fashion_mnist
  
  # 使用 SVM 分類器
  python main.py --classifier svm --n_components 100
  
  # 使用隨機森林,更多組件
  python main.py --classifier rf --n_components 150
        """
    )
    
    # 定義各個參數
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'digits'],
                       help='使用哪個數據集 (default: mnist)')
    
    parser.add_argument('--n_components', type=int, default=50,
                       help='HOSVD 的組件數量 (default: 50)')
    
    parser.add_argument('--classifier', type=str, default='knn',
                       choices=['knn', 'svm', 'rf', 'mlp'],
                       help='分類器類型 (default: knn)')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='測試集的比例 (default: 0.2 = 20%)')
    
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                       help='不生成圖表')
    
    # 設定默認值
    parser.set_defaults(visualize=True)
    
    # 解析命令行參數
    args = parser.parse_args()
    
    # ====== 運行主程序 ======
    results = main(args)
    
    # ====== 打印結果摘要 ======
    # 這就像實驗報告的最後一頁,總結所有重要結果
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (結果摘要)")
    print("=" * 80)
    print(f"Dataset: {results['configuration']['dataset']}")
    print(f"Classifier: {results['configuration']['classifier']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")  # 正確率
    print(f"Precision: {results['metrics']['precision']:.4f}")  # 精確度
    print(f"Recall: {results['metrics']['recall']:.4f}")  # 召回率
    print(f"F1-Score: {results['metrics']['f1']:.4f}")  # F1 分數
    print(f"Compression Ratio: {results['hosvd_info']['compression_ratio']:.4f}")  # 壓縮率
    print("=" * 80)
