"""
RESOURCES.md - 項目資源和參考指南
"""

# 📚 HOSVD 手寫辨識系統 - 資源和參考指南

## 📖 文檔資源

### 📄 使用文檔（按推薦閱讀順序）
1. **README.md** ⭐ 首先閱讀
   - 項目概述
   - 功能特性
   - 安裝說明
   - 基本用法示例
   - 典型結果
   - 參考文獻

2. **QUICKSTART.md** ⭐ 快速開始
   - 5分鐘快速上手
   - 命令行用法
   - Python API示例
   - Jupyter使用
   - 常見參數

3. **INDEX.md** 🔍 快速導航
   - 快速定位功能
   - 文件樹結構
   - 常見任務速查
   - 函數速查表
   - 學習路線

4. **PROJECT_SUMMARY.md** 📊 詳細文檔
   - 項目背景
   - 完整結構
   - 模塊說明
   - 算法原理
   - 性能指標

5. **FILE_MANIFEST.md** 📋 文件清單
   - 所有文件列表
   - 功能說明
   - 常見任務
   - 文檔導航

6. **COMPLETION_REPORT.md** ✅ 完成報告
   - 項目完成情況
   - 統計信息
   - 技術特性
   - 交付物清單

---

## 💻 代碼資源

### 🔧 主要模塊

#### data/ - 數據模塊
```python
from data import load_data, DataPreprocessor, augment_data

# 加載數據
X_train, y_train, X_test, y_test = load_data('mnist')

# 預處理
prep = DataPreprocessor(normalize=True)
X_processed = prep.fit_transform(X_train)

# 增強
X_aug, y_aug = augment_data(X_train, y_train)
```

#### models/ - 模型模塊
```python
from models import HOSVDModel, ClassifierPipeline

# HOSVD
hosvd = HOSVDModel(n_components=50)
X_reduced = hosvd.fit_transform(X_train)

# 分類
clf = ClassifierPipeline('svm')
clf.fit(X_reduced, y_train)
pred = clf.predict(hosvd.transform(X_test))
```

#### utils/ - 工具模塊
```python
from utils import (
    Metrics, ModelEvaluator,
    plot_confusion_matrix, plot_digits,
    FileManager, Logger
)

# 評估
evaluator = ModelEvaluator(y_true, y_pred)
metrics = evaluator.get_metrics()

# 可視化
fig = plot_confusion_matrix(cm)

# 保存
FileManager.save_model(model, 'path/model.pkl')
```

---

## 🎓 學習資源

### 📖 代碼示例文件

**examples.py** - 6個進階示例
- `example_1_basic_workflow()` - 基本工作流程
- `example_2_classifier_comparison()` - 分類器比較
- `example_3_parameter_tuning()` - 參數調優
- `example_4_ensemble_learning()` - 集成學習
- `example_5_advanced_analysis()` - 高級分析
- `example_6_custom_workflow()` - 自定義工作流程

**運行方式**:
```bash
python examples.py
# 或導入特定示例
from examples import example_1_basic_workflow
example_1_basic_workflow()
```

### 📚 Jupyter筆記本

**notebooks/analysis.ipynb** - 14個交互式單元
1. 環境設置
2. 數據加載
3. 數據預覽
4. HOSVD分解
5. 分類器訓練
6. 模型評估
7. 混淆矩陣
8. 降維可視化
9. 分類器比較
10. 參數敏感性分析
11. 準確度關係
12. 錯誤分析
13. 結果展示
14. 總結

**運行方式**:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## ⚙️ 配置資源

### 📝 config.py 配置項

#### DATA_CONFIG
```python
{
    'dataset': 'mnist',        # mnist, fashion_mnist, digits
    'test_size': 0.2,          # 測試集比例
    'random_state': 42,        # 隨機種子
    'normalize': True,         # 是否歸一化
}
```

#### HOSVD_CONFIG
```python
{
    'n_components': 50,        # 主成分數
    'random_state': 42,        # 隨機種子
}
```

#### CLASSIFIER_CONFIG
```python
{
    'type': 'knn',
    'knn': {'n_neighbors': 5, 'weights': 'uniform'},
    'svm': {'kernel': 'rbf', 'C': 1.0},
    'rf': {'n_estimators': 100, 'max_depth': None},
    'mlp': {'hidden_layer_sizes': (256, 128, 64)},
}
```

#### PATH_CONFIG
```python
{
    'data_dir': './data/raw',
    'model_dir': './results/models',
    'figure_dir': './results/figures',
}
```

---

## 🎯 命令行資源

### 基本命令

```bash
# 默認配置（MNIST + KNN）
python main.py

# 使用Fashion-MNIST
python main.py --dataset fashion_mnist

# 使用SVM分類器
python main.py --classifier svm

# 增加主成分數
python main.py --n_components 100

# 組合使用
python main.py --dataset fashion_mnist --classifier rf --n_components 150

# 禁用可視化（加快速度）
python main.py --no-visualize

# 查看所有選項
python main.py --help

# 運行示例
python examples.py

# 啟動Jupyter
jupyter notebook notebooks/analysis.ipynb
```

### 參數詳解

| 參數 | 類型 | 默認值 | 說明 |
|------|------|-------|------|
| --dataset | str | mnist | 數據集選擇 |
| --n_components | int | 50 | HOSVD主成分數 |
| --classifier | str | knn | 分類器類型 |
| --test_size | float | 0.2 | 測試集比例 |
| --no-visualize | flag | - | 禁用可視化 |
| --help | flag | - | 顯示幫助 |

---

## 📊 輸出資源

### 文件生成位置

```
results/
├── models/
│   ├── hosvd_model.pkl         # HOSVD模型
│   ├── knn_classifier.pkl      # 分類器模型
│   └── results.json            # 結果記錄
└── figures/
    ├── sample_digits_*.png      # 樣本數字
    ├── confusion_matrix_*.png   # 混淆矩陣
    ├── metrics_*.png            # 分類指標
    └── dimensionality_reduction_*.png
```

### 結果JSON格式

```json
{
    "timestamp": "2025-01-01T12:00:00",
    "configuration": {
        "dataset": "mnist",
        "n_components": 50,
        "classifier": "knn",
        "test_size": 0.2
    },
    "hosvd_info": {
        "core_tensor_shape": "(1, 50, 50)",
        "compression_ratio": 0.0637
    },
    "metrics": {
        "accuracy": 0.962,
        "precision": 0.9615,
        "recall": 0.962,
        "f1": 0.9618
    }
}
```

---

## 🔗 外部資源

### 官方文檔
- NumPy: https://numpy.org/doc/
- scikit-learn: https://scikit-learn.org/stable/documentation.html
- tensorly: http://tensorly.org/stable/index.html
- Matplotlib: https://matplotlib.org/stable/contents.html

### 數據集
- MNIST: http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- sklearn digits: https://scikit-learn.org/stable/datasets/toy_dataset.html

### 論文和書籍
- Tucker, L.R. (1966): "Some mathematical notes on three-mode factor analysis"
- Kolda, T.G., & Bader, B.W. (2009): "Tensor Decompositions and Applications"
- LeCun, Y., et al. (1998): "The MNIST Database of Handwritten Digits"

---

## 🛠️ 開發資源

### 代碼結構
- `data/` - 數據處理（300+行）
- `models/` - 算法實現（700+行）
- `utils/` - 工具函數（900+行）
- `config.py` - 配置管理（100+行）
- `main.py` - 主程序（300+行）

### 關鍵類
| 類 | 文件 | 功能 |
|----|------|------|
| HOSVDModel | models/hosvd_model.py | 張量分解 |
| ClassifierPipeline | models/classifier.py | 分類器 |
| DataPreprocessor | data/preprocessor.py | 數據預處理 |
| ModelEvaluator | utils/metrics.py | 性能評估 |
| FileManager | utils/helpers.py | 文件管理 |

---

## 💡 最佳實踐

### 使用建議

1. **數據處理**
   - 始終歸一化輸入數據
   - 使用訓練集統計進行測試集預處理
   - 考慮數據增強以提高泛化性能

2. **模型選擇**
   - 從KNN開始快速基準測試
   - 使用SVM獲得更好的準確度
   - 嘗試RF以獲得特徵重要性

3. **參數調優**
   - 使用 example_3 進行參數搜索
   - 監控訓練和測試精度
   - 避免過擬合

4. **性能評估**
   - 使用多個指標（不僅是精度）
   - 檢查混淆矩陣
   - 分析每類的性能

---

## 📈 性能優化

### 加速技巧

1. **減少主成分數**
   ```bash
   python main.py --n_components 30
   ```

2. **禁用可視化**
   ```bash
   python main.py --no-visualize
   ```

3. **使用KNN分類器**
   ```bash
   python main.py --classifier knn
   ```

4. **減少測試集**
   ```python
   # 在代碼中修改
   test_size = 0.1  # 減少測試集
   ```

### 內存優化

1. 限制訓練集大小
2. 使用較小的主成分數
3. 禁用數據增強

---

## 🎓 學習路線

### 初級（30分鐘）
1. 閱讀 README.md
2. 運行 `python main.py`
3. 查看輸出結果

### 中級（2小時）
1. 閱讀 QUICKSTART.md
2. 修改 config.py 進行實驗
3. 運行 examples.py

### 高級（全天）
1. 深入 PROJECT_SUMMARY.md
2. 研究源代碼實現
3. 編寫自己的擴展

---

## 📞 支持資源

### 常見問題

**Q: 如何修改默認參數？**
A: 編輯 config.py 或使用命令行參數

**Q: 支持哪些數據集？**
A: MNIST, Fashion-MNIST, sklearn digits

**Q: 如何增加新的分類器？**
A: 在 models/classifier.py 中添加

**Q: 如何自定義可視化？**
A: 修改 utils/visualization.py

---

## ✅ 檢查清單

- [ ] 安裝 requirements.txt
- [ ] 閱讀 README.md
- [ ] 運行 `python main.py`
- [ ] 查看生成的結果
- [ ] 修改配置進行實驗
- [ ] 運行 examples.py
- [ ] 探索 Jupyter 筆記本

---

## 📚 推薦閱讀順序

1. **快速開始**（5分鐘）
   - README.md 概述部分
   - QUICKSTART.md

2. **基本使用**（30分鐘）
   - config.py 說明
   - main.py 幫助信息
   - 運行第一個實驗

3. **深入學習**（2小時）
   - examples.py 所有示例
   - PROJECT_SUMMARY.md 完整文檔
   - notebooks/analysis.ipynb

4. **源代碼研究**（數小時）
   - models/hosvd_model.py 實現
   - utils/visualization.py 實現
   - 完整項目架構

---

## 🎁 額外資源

### 預訓練模型
- 可在 results/models/ 中保存使用過的模型
- 支持模型重用以加快迭代

### 數據預處理
- 支持多種數據增強技術
- 可配置的歸一化策略

### 可視化
- 8種不同的圖表類型
- 可自定義的視覺效果

---

**所有資源就在這裡，祝您使用愉快！** 🚀

*最後更新: 2025年*  
*版本: 1.0.0*
