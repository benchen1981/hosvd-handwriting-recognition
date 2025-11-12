"""
INDEX.md - 快速索引和導航
"""

# 🗂️ HOSVD 手寫辨識系統 - 快速索引

## 📍 快速定位

### 🔍 我要找...

#### 使用文檔
- ❓ 不知道怎麼開始？ → **QUICKSTART.md**
- 📖 詳細說明？ → **README.md**
- 📊 完整信息？ → **PROJECT_SUMMARY.md**
- 📋 文件清單？ → **FILE_MANIFEST.md**
- 🔗 快速導航？ → **INDEX.md** (本文件)

#### 配置
- ⚙️ 全局配置？ → **config.py**
- 🎯 修改參數？ → config.py 的 `DATA_CONFIG`, `HOSVD_CONFIG` 等
- 🔧 自定义分类器？ → config.py 的 `CLASSIFIER_CONFIG`

#### 代码
- 📥 數據加载？ → **data/loader.py**
- 🔄 數據预处理？ → **data/preprocessor.py**
- 🧠 HOSVD算法？ → **models/hosvd_model.py**
- 🤖 分类器？ → **models/classifier.py**
- 📊 可视化？ → **utils/visualization.py**
- 📈 评估指標？ → **utils/metrics.py**
- 🛠️ 辅助工具？ → **utils/helpers.py**

#### 示例和演示
- 🎓 學习用法？ → **examples.py** (6個例子)
- 📚 交互式分析？ → **notebooks/analysis.ipynb**

#### 运行
- 🚀 快速运行？ → `python main.py`
- 📝 所有选项？ → `python main.py --help`
- 🧪 运行示例？ → `python examples.py`

---

## 📂 文件树

```
hosvd_handwriting_recognition/
│
├── 📖 DOCUMENTATION
│   ├── README.md                ← 開始這里
│   ├── QUICKSTART.md           ← 快速上手
│   ├── PROJECT_SUMMARY.md      ← 详细文档
│   ├── FILE_MANIFEST.md        ← 文件清单
│   └── INDEX.md                ← 本文件
│
├── ⚙️ CONFIGURATION
│   ├── config.py               ← 所有配置
│   ├── requirements.txt        ← 依赖
│   └── __init__.py             ← 包初始化
│
├── 🚀 MAIN PROGRAM
│   ├── main.py                 ← 主程序
│   └── examples.py             ← 示例代码
│
├── 📥 DATA MODULE (data/)
│   ├── loader.py               ← 數據加载
│   ├── preprocessor.py         ← 數據预处理
│   └── __init__.py
│
├── 🤖 MODELS MODULE (models/)
│   ├── hosvd_model.py          ← HOSVD张量分解
│   ├── classifier.py           ← 分类器集合
│   └── __init__.py
│
├── 🛠️ UTILS MODULE (utils/)
│   ├── visualization.py        ← 8种可视化
│   ├── metrics.py              ← 评估指標
│   ├── helpers.py              ← 辅助工具
│   └── __init__.py
│
├── 📚 NOTEBOOKS (notebooks/)
│   └── analysis.ipynb          ← Jupyter笔记本
│
└── 📊 RESULTS (results/)
    ├── models/                 ← 保存的模型
    └── figures/                ← 生成的圖表
```

---

## 🎯 常见任务速查

### 任务 1️⃣ : 第一次使用
```
1. 阅读 → QUICKSTART.md (5分钟)
2. 安裝 → pip install -r requirements.txt
3. 运行 → python main.py
4. 查看 → results/ 目录下的結果
```

### 任务 2️⃣ : 修改參數
```
1. 编辑 → config.py
2. 修改 → n_components, classifier 等參數
3. 运行 → python main.py
```

### 任务 3️⃣ : 自己的代码
```python
from data import load_data
from models import HOSVDModel, ClassifierPipeline

# 1. 加载數據
X_train, y_train, X_test, y_test = load_data('mnist')

# 2. HOSVD
hosvd = HOSVDModel(n_components=50)
X_tr = hosvd.fit_transform(X_train)
X_te = hosvd.transform(X_test)

# 3. 分类
clf = ClassifierPipeline('svm')
clf.fit(X_tr, y_train)

# 4. 评估
acc = clf.score(X_te, y_test)
print(f"精度: {acc:.4f}")
```

### 任务 4️⃣ : 學习算法
```
1. 理论 → PROJECT_SUMMARY.md 的"算法原理"
2. 代码 → models/hosvd_model.py
3. 实驗 → examples.py 的 example_5
4. 演示 → notebooks/analysis.ipynb
```

### 任务 5️⃣ : 試驗不同參數
```bash
# 試驗主成分數
for n in 10 30 50 100; do
    python main.py --n_components $n --no-visualize
done

# 試驗分类器
for clf in knn svm rf mlp; do
    python main.py --classifier $clf --no-visualize
done

# 試驗數據集
for ds in mnist fashion_mnist digits; do
    python main.py --dataset $ds --no-visualize
done
```

### 任务 6️⃣ : 交互式分析
```bash
jupyter notebook notebooks/analysis.ipynb
# 然後在浏览器中打開 localhost:8888
```

---

## 📚 函數速查表

### 數據
```python
from data import load_data, DataPreprocessor

# 加载
X_train, y_train, X_test, y_test = load_data('mnist')

# 预处理
prep = DataPreprocessor(normalize=True)
X_train = prep.fit_transform(X_train)
```

### 模型
```python
from models import HOSVDModel, ClassifierPipeline

# HOSVD
hosvd = HOSVDModel(n_components=50)
X_red = hosvd.fit_transform(X_train)

# 分类
clf = ClassifierPipeline('svm')
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
```

### 评估
```python
from utils import ModelEvaluator, Metrics

# 评估
evaluator = ModelEvaluator(y_true, y_pred)
metrics = evaluator.get_metrics()

# 混淆矩阵
cm = Metrics.get_confusion_matrix(y_true, y_pred)
```

### 可视化
```python
from utils import (
    plot_digits,
    plot_confusion_matrix,
    plot_classification_metrics,
    plot_dimensionality_reduction
)

# 绘圖
fig = plot_digits(images, labels)
fig = plot_confusion_matrix(cm)
```

---

## 🎓 學习路线

### 🌱 初级 (15分钟)
1. 读 QUICKSTART.md
2. 跑 `python main.py`
3. 看結果圖表

### 🌿 中级 (1小时)
1. 研究 examples.py
2. 修改參數試驗
3. 理解 config.py

### 🌳 高级 (多小时)
1. 深入 models/hosvd_model.py
2. 阅读 PROJECT_SUMMARY.md
3. 编写自己的擴展

---

## 🔗 關键链接

### 在项目中
| 文件 | 用途 | 行數 |
|------|------|------|
| main.py | 主程序 | 300+ |
| models/hosvd_model.py | HOSVD实现 | 300+ |
| utils/visualization.py | 可视化 | 400+ |
| examples.py | 示例代码 | 300+ |

### 外部资源
- MNIST官网: http://yann.lecun.com/exdb/mnist/
- scikit-learn: https://scikit-learn.org
- tensorly: http://tensorly.org

---

## ⚡ 快速參考

### 命令行
```bash
# 基本
python main.py

# 自定义
python main.py --dataset mnist --classifier svm --n_components 100

# 帮助
python main.py --help

# 示例
python examples.py

# Jupyter
jupyter notebook notebooks/analysis.ipynb
```

### 所有參數
| 參數 | 值 | 默认 |
|------|-----|------|
| --dataset | mnist/fashion_mnist/digits | mnist |
| --classifier | knn/svm/rf/mlp | knn |
| --n_components | 整數 | 50 |
| --test_size | 0-1浮數 | 0.2 |
| --no-visualize | - | 禁用 |

---

## ✅ 检查清单

首次設置:
- [ ] Python 3.7+
- [ ] pip install -r requirements.txt
- [ ] python main.py (測試)

開始使用:
- [ ] 阅读 QUICKSTART.md
- [ ] 修改 config.py (可选)
- [ ] 运行你的第一個实驗

深入學习:
- [ ] 研究 examples.py
- [ ] 理解 PROJECT_SUMMARY.md
- [ ] 修改代码進行实驗

---

## 📞 支持和问题

### 常见问题 ❓
- 内存不足？→ 减少 n_components 或 test_size
- 速度慢？→ 使用 --no-visualize
- 导入错误？→ 检查 requirements.txt 安裝

### 需要帮助？
1. 查看 PROJECT_SUMMARY.md
2. 阅读代码注释
3. 运行 examples.py
4. 检查 notebooks/analysis.ipynb

---

## 🎯 30秒快速開始

```bash
# 1. 安裝
pip install -r requirements.txt

# 2. 运行
python main.py

# 3. 查看結果
# results/ 目录中有所有输出
```

---

## 📊 一览表

| 功能 | 文件 | 主要类/函數 |
|------|------|-----------|
| 數據加载 | data/loader.py | load_data() |
| 數據预处理 | data/preprocessor.py | DataPreprocessor |
| HOSVD分解 | models/hosvd_model.py | HOSVDModel |
| 分类 | models/classifier.py | ClassifierPipeline |
| 可视化 | utils/visualization.py | plot_* 系列 |
| 评估 | utils/metrics.py | Metrics, ModelEvaluator |
| 工具 | utils/helpers.py | FileManager, Logger |

---

## 🚀 開始吧！

1. 📖 **阅读**: README.md 或 QUICKSTART.md
2. 🔧 **安裝**: `pip install -r requirements.txt`
3. ▶️ **运行**: `python main.py`
4. 🎉 **成功**: 查看 results/ 中的输出

---

**祝您使用愉快！** ✨

*版本: 1.0.0 | 更新: 2025年 | 作者: 陳宥興 (5114050015)*
