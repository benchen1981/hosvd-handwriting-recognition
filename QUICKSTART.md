"""
快速開始指南
"""

# HOSVD 手寫辨識系統 - 快速開始

## 💾 安裝依賴

```bash
pip install -r requirements.txt
```

如果使用conda環境：

```bash
conda create -n hosvd python=3.8
conda activate hosvd
pip install -r requirements.txt
```

## 🚀 基本使用

### 方式1：命令行使用

```bash
# 默認配置（MNIST + KNN）
python main.py

# 使用自定義參數
python main.py --dataset mnist --n_components 50 --classifier svm

# 使用Fashion-MNIST數據集
python main.py --dataset fashion_mnist --classifier rf

# 不生成可視化（加速運行）
python main.py --n_components 100 --no-visualize
```

### 方式2：Python API使用

```python
from data import load_data
from models import HOSVDModel, ClassifierPipeline
from utils import ModelEvaluator

# 1. 加載數據
X_train, y_train, X_test, y_test = load_data('mnist')

# 2. HOSVD分解
hosvd = HOSVDModel(n_components=50)
X_train_reduced = hosvd.fit_transform(X_train)
X_test_reduced = hosvd.transform(X_test)

# 3. 訓練分類器
classifier = ClassifierPipeline('knn', n_neighbors=5)
classifier.fit(X_train_reduced, y_train)

# 4. 評估
from sklearn.metrics import accuracy_score
predictions = classifier.predict(X_test_reduced)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

### 方式3：Jupyter Notebook

在 `notebooks/analysis.ipynb` 中有完整的交互式演示。

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 📊 命令行參數說明

| 參數 | 默認值 | 說明 |
|------|-------|------|
| `--dataset` | `mnist` | 數據集 (mnist/fashion_mnist/digits) |
| `--n_components` | `50` | HOSVD主成分數 |
| `--classifier` | `knn` | 分類器 (knn/svm/rf/mlp) |
| `--test_size` | `0.2` | 測試集比例 |
| `--no-visualize` | - | 禁用可視化 |

## 📁 項目結構

```
hosvd_handwriting_recognition/
├── config.py                 # 配置文件
├── main.py                   # 主程序
├── data/
│   ├── loader.py            # 數據加載
│   ├── preprocessor.py      # 數據預處理
│   └── __init__.py
├── models/
│   ├── hosvd_model.py       # HOSVD實現
│   ├── classifier.py        # 分類器集合
│   └── __init__.py
├── utils/
│   ├── visualization.py     # 可視化工具
│   ├── metrics.py           # 評估指標
│   ├── helpers.py           # 輔助函數
│   └── __init__.py
├── notebooks/
│   └── analysis.ipynb       # Jupyter筆記本
└── results/
    ├── models/              # 保存的模型
    └── figures/             # 生成的圖表
```

## 🔧 配置說明

編輯 `config.py` 修改默認配置：

### 數據配置
```python
DATA_CONFIG = {
    'dataset': 'mnist',
    'test_size': 0.2,
    'random_state': 42,
    'normalize': True,
}
```

### HOSVD配置
```python
HOSVD_CONFIG = {
    'n_components': 50,
    'random_state': 42,
}
```

### 分類器配置
```python
CLASSIFIER_CONFIG = {
    'type': 'knn',
    'knn': {
        'n_neighbors': 5,
        'weights': 'uniform',
    },
    # ... 其他分類器配置
}
```

## 📈 實驗示例

### 實驗1：比較不同主成分數

```bash
# 測試n_components = 10, 30, 50, 100
for n in 10 30 50 100; do
    python main.py --n_components $n
done
```

### 實驗2：比較分類器

```bash
# 測試所有分類器
for clf in knn svm rf mlp; do
    python main.py --classifier $clf
done
```

### 實驗3：多數據集

```bash
# 在不同數據集上測試
for ds in mnist fashion_mnist; do
    python main.py --dataset $ds
done
```

## 📊 輸出文件

運行後會生成以下文件：

- `results/models/hosvd_model.pkl` - HOSVD模型
- `results/models/{classifier_type}_classifier.pkl` - 分類器模型
- `results/models/results.json` - 實驗結果（JSON格式）
- `results/figures/sample_digits_*.png` - 樣本數字
- `results/figures/confusion_matrix_*.png` - 混淆矩陣
- `results/figures/metrics_*.png` - 分類指標

## 🎓 算法原理

### HOSVD (Higher-Order SVD)

HOSVD是SVD在多維張量上的推廣：

1. **張量重塑**: 將二維影像數據重塑為三階張量
2. **分解**: $\mathcal{T} = \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$
3. **降維**: 通過因子矩陣投影實現維度縮減
4. **特徵提取**: 使用降維後的特徵進行分類

### 分類器說明

- **KNN**: K-最近鄰，適合小到中等規模數據
- **SVM**: 支持向量機，泛化能力強
- **RF**: 隨機森林，適合並行處理
- **MLP**: 多層感知機，深度學習分類

## ⚡ 性能優化

1. **數據預處理**: 歸一化加快計算
2. **主成分選擇**: 較少的成分可加速訓練
3. **分類器選擇**: KNN最快，SVM次之，RF較慢
4. **並行處理**: 某些分類器支持多核

## 🐛 故障排除

### 問題1: 內存不足
**解決方案**:
- 減少訓練集大小
- 降低主成分數
- 使用更少的數據

### 問題2: 運行速度慢
**解決方案**:
- 使用 `--no-visualize` 禁用可視化
- 減少主成分數
- 使用KNN替代SVM

### 問題3: 導入錯誤
**解決方案**:
- 確保在項目目錄運行
- 檢查所有依賴已安裝
- 驗證Python版本 >= 3.7

## 📚 相關文獻

1. Tucker, L.R., "Some mathematical notes on three-mode factor analysis", Psychometrika, 1966.
2. Kolda, T.G., & Bader, B.W., "Tensor Decompositions and Applications", SIAM, 2009.
3. LeCun, Y., et al., "The MNIST Database of Handwritten Digits", 1998.

## 📝 作者信息

- 學生ID: 5114050015
- 課程: 數據分析數學
- 機構: 中興大學

## 📄 許可證

MIT License

## 💡 建議與反饋

如有問題或建議，歡迎提出！

---

**最後更新**: 2025年
