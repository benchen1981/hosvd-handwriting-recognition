"""
使用指南文件結構清單
"""

# 📚 HOSVD 手寫辨識系統 - 完整文件清單

## ✅ 已創建的文件

### 核心配置 (4個文件)
- ✅ `config.py` - 全局配置管理
- ✅ `__init__.py` - 包初始化
- ✅ `requirements.txt` - 依賴列表
- ✅ `main.py` - 主程序入口

### 文檔 (4個文件)
- ✅ `README.md` - 項目說明書
- ✅ `QUICKSTART.md` - 快速開始指南
- ✅ `PROJECT_SUMMARY.md` - 完整總結文檔
- ✅ `FILE_MANIFEST.md` - 本文件（文件清單）

### 數據模塊 - data/ (3個文件)
- ✅ `data/__init__.py` - 模塊初始化
- ✅ `data/loader.py` - 數據加載器
- ✅ `data/preprocessor.py` - 數據預處理

### 模型模塊 - models/ (3個文件)
- ✅ `models/__init__.py` - 模塊初始化
- ✅ `models/hosvd_model.py` - HOSVD張量分解
- ✅ `models/classifier.py` - 分類器集合

### 工具模塊 - utils/ (4個文件)
- ✅ `utils/__init__.py` - 模塊初始化
- ✅ `utils/visualization.py` - 可視化工具
- ✅ `utils/metrics.py` - 評估指標
- ✅ `utils/helpers.py` - 輔助工具

### 筆記本 - notebooks/ (1個文件)
- ✅ `notebooks/analysis.ipynb` - Jupyter交互式筆記本

### 示例代碼 (1個文件)
- ✅ `examples.py` - 6個高級示例

### 輸出目錄 - results/
- ✅ `results/models/` - 模型保存目錄
- ✅ `results/figures/` - 圖表保存目錄

---

## 📊 項目統計

| 類別 | 數量 | 說明 |
|------|------|------|
| Python文件 | 15 | 核心代碼 |
| 文檔文件 | 4 | MD格式 |
| Jupyter筆記本 | 1 | 交互式分析 |
| **總計** | **20** | 完整項目 |

---

## 🎯 快速導航

### 我想...

#### 🚀 快速開始
```
1. 閱讀 QUICKSTART.md
2. 運行 python main.py
```

#### 🔧 自定義配置
```
1. 編輯 config.py
2. 修改 DATA_CONFIG, HOSVD_CONFIG 等
3. 運行 python main.py
```

#### 💻 使用API
```
1. 導入模塊: from models import HOSVDModel
2. 查看 examples.py 獲得靈感
3. 編寫自己的代碼
```

#### 📊 交互式分析
```
1. 運行 jupyter notebook notebooks/analysis.ipynb
2. 在Jupyter中執行單元格
```

#### 📈 參數調優
```
1. 查看 examples.py 的 example_3_parameter_tuning()
2. 修改參數範圍
3. 運行並查看結果
```

#### 🔍 學習算法
```
1. 閱讀 PROJECT_SUMMARY.md 的"算法原理"部分
2. 查看 models/hosvd_model.py 的代碼
3. 運行 examples.py 的 example_5_advanced_analysis()
```

---

## 📖 文檔導航

### 各文檔適合人群

| 文檔 | 內容 | 適合 |
|------|------|------|
| README.md | 項目概述 | 首次使用者 |
| QUICKSTART.md | 快速教程 | 想快速上手 |
| PROJECT_SUMMARY.md | 詳細文檔 | 深度使用者 |
| config.py | 配置說明 | 需要調參 |
| examples.py | 代碼示例 | 學習使用 |
| code註釋 | 實現細節 | 開發者 |

---

## 🎓 學習路徑

### 初級用戶
1. 閱讀 QUICKSTART.md
2. 運行 `python main.py`
3. 查看輸出結果和圖表

### 中級用戶
1. 了解 config.py 配置
2. 運行 examples.py
3. 修改參數進行實驗

### 高級用戶
1. 深入 PROJECT_SUMMARY.md
2. 閱讀源代碼和註釋
3. 自定義擴展功能

---

## 🔧 常見任務

### 任務1: 在MNIST上測試
```bash
python main.py --dataset mnist --n_components 50
```

### 任務2: 在Fashion-MNIST上測試
```bash
python main.py --dataset fashion_mnist --classifier svm
```

### 任務3: 比較所有分類器
```bash
for clf in knn svm rf mlp; do
    python main.py --classifier $clf
done
```

### 任務4: 參數掃描
```bash
for n in 10 30 50 100; do
    python main.py --n_components $n
done
```

### 任務5: 運行高級示例
```bash
python examples.py
```

### 任務6: 交互式分析
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📦 依賴說明

### 必需包
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- tensorly >= 0.7.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

### 可選包
- pandas >= 1.3.0
- opencv-python >= 4.5.0
- pillow >= 9.0.0
- jupyter (用於筆記本)

---

## ✨ 項目亮點

✅ **完整實現** - 從數據到結果
✅ **多分類器** - 4種分類算法
✅ **可視化** - 8種圖表類型
✅ **易用API** - 簡潔的接口
✅ **完善文檔** - 詳細的說明
✅ **豐富示例** - 6個示例代碼
✅ **高效算法** - 優化的實現
✅ **模塊化** - 清晰的架構

---

## 🎯 主要類和函數

### 數據加載
```python
from data import load_data
X_train, y_train, X_test, y_test = load_data('mnist')
```

### HOSVD分解
```python
from models import HOSVDModel
hosvd = HOSVDModel(n_components=50)
X_reduced = hosvd.fit_transform(X)
```

### 分類
```python
from models import ClassifierPipeline
clf = ClassifierPipeline('svm')
clf.fit(X_train, y_train)
```

### 評估
```python
from utils import ModelEvaluator
evaluator = ModelEvaluator(y_test, predictions)
metrics = evaluator.get_metrics()
```

### 可視化
```python
from utils import plot_confusion_matrix
fig = plot_confusion_matrix(cm)
```

---

## 🚀 開始使用

### 第1步: 安裝依賴
```bash
pip install -r requirements.txt
```

### 第2步: 運行基本程序
```bash
python main.py
```

### 第3步: 查看結果
```
✓ 模型已保存到 results/models/
✓ 圖表已保存到 results/figures/
✓ 結果已保存到 results/models/results.json
```

### 第4步: 自定義實驗
```bash
python main.py --dataset fashion_mnist --classifier rf --n_components 100
```

---

## 📊 輸出示例

### 控制台輸出
```
================================================================================
HOSVD Handwriting Recognition System
================================================================================
Configuration:
  Dataset: mnist
  HOSVD components: 50
  Classifier: knn
  Test size: 0.2
================================================================================
Loading mnist dataset...
Original shapes - Train: (60000, 784), Test: (10000, 784)
Preprocessed shapes - Train: (60000, 784), Test: (10000, 784)
Applying HOSVD with 50 components...
Reduced shapes - Train: (60000, 50), Test: (10000, 50)
Core tensor shape: (1, 50, 50)
Compression ratio: 0.0637
Training knn classifier...
Train accuracy: 0.9680
Evaluating model on mnist...
Test accuracy: 0.9620
Precision: 0.9615
Recall: 0.9620
F1-Score: 0.9618
================================================================================
Experiment completed successfully!
================================================================================
```

### 生成的文件
```
results/
├── models/
│   ├── hosvd_model.pkl              (HOSVD模型)
│   ├── knn_classifier.pkl           (分類器)
│   └── results.json                 (結果記錄)
└── figures/
    ├── sample_digits_20250001_120000.png
    ├── confusion_matrix_knn_20250001_120000.png
    ├── metrics_knn_20250001_120000.png
    └── dimensionality_reduction_20250001_120000.png
```

---

## 🎁 額外資源

### 在線資源
- MNIST官網: http://yann.lecun.com/exdb/mnist/
- scikit-learn文檔: https://scikit-learn.org
- tensorly文檔: http://tensorly.org

### 參考論文
- Tucker, L.R. (1966): Some mathematical notes on three-mode factor analysis
- Kolda & Bader (2009): Tensor Decompositions and Applications

---

## ❓ 常見問題

### Q: 如何加速運行？
**A**: 使用 `--no-visualize` 選項，減少 `n_components`，或使用KNN分類器。

### Q: 內存不足？
**A**: 減少訓練集大小，或在命令行中添加 `--test_size 0.1`。

### Q: 如何修改參數？
**A**: 編輯 `config.py` 或使用命令行參數。

### Q: 支持其他數據集嗎？
**A**: 可以。在 `data/loader.py` 中添加新的加載函數。

### Q: 如何擴展分類器？
**A**: 在 `models/classifier.py` 中添加新的分類器類型。

---

## 📝 版本歷史

| 版本 | 日期 | 更新 |
|------|------|------|
| 1.0.0 | 2025年 | 初版發佈 |

---

## 📞 支持

有問題或建議？請查看：
1. 本項目文檔
2. 代碼中的詳細註釋
3. examples.py 中的示例
4. Jupyter筆記本中的教程

---

**祝您使用愉快！** 🎉

---

*最後更新: 2025年*  
*版本: 1.0.0*  
*作者: 陳宥興 (5114050015)*
