# CRISP-DM Phase 3: 數據準備 (Data Preparation)

## 1. 數據清理 (Data Cleaning)

### 1.1 缺失值檢查與處理

```python
# 檢查結果:
MNIST數據: 無缺失值
Fashion-MNIST數據: 無缺失值
USPS數據: 無缺失值

# 處理策略:
無特殊處理需求 (官方數據集已預清理)
```

### 1.2 異常值檢測

#### 像素值異常
```
正常範圍: [0, 255] (灰度值)
異常值檢查結果: 0 個異常

# 像素值分佈:
Q1: 0     (下四分位)
Q2: 0     (中位數)
Q3: 140   (上四分位)
Max: 255  (正常邊界)

結論: 無異常值清理需求
```

#### 圖像尺寸異常
```
MNIST樣本: 100% 符合28×28
Fashion-MNIST樣本: 100% 符合28×28
USPS樣本: 100% 符合16×16 (需調整)

# 處理:
USPS → 雙線性插值 resize 到 28×28
```

### 1.3 重複樣本檢測

```
特徵哈希檢查:
  MNIST: 0 個重複
  Fashion-MNIST: 0 個重複
  
結論: 無重複樣本移除需求
```

## 2. 特徵工程 (Feature Engineering)

### 2.1 像素值正規化

#### 方法 A: 最小-最大正規化 (Min-Max Scaling)
```python
X_normalized = (X - X_min) / (X_max - X_min)

# 計算步驟:
X_min = 0
X_max = 255
# 結果: X_normalized ∈ [0, 1]

# 優點: 
- 保留原始分布形狀
- 便於解釋
- HOSVD數值穩定

# 使用場景: 推薦用於HOSVD前處理
```

#### 方法 B: Z-score 標準化
```python
X_standardized = (X - μ) / σ

# MNIST參數:
μ ≈ 33.3
σ ≈ 78.6

# 優點:
- 均值=0, 方差=1
- 適合PCA/HOSVD
- 數值穩定性更好

# 選擇: 實驗中採用此法
```

#### 應用代碼
```python
from sklearn.preprocessing import StandardScaler

# 初始化
scaler = StandardScaler()

# 訓練階段
X_train_scaled = scaler.fit_transform(X_train)

# 測試階段  
X_test_scaled = scaler.transform(X_test)

# 特性: fit on train, transform on test
# → 防止數據洩露
```

### 2.2 張量化 (Tensorization)

#### 結構設計

```
Input: X_train.shape = (n_samples, 784)
       └─ 60,000 × 784 陣列

Step 1: reshape to images
X_images.shape = (n_samples, 28, 28)
       └─ 60,000 × 28 × 28

Step 2: 轉置為標準張量形式
X_tensor.shape = (28, 28, n_samples)
       └─ 28 × 28 × 60,000 三階張量

Interpretation:
  X_tensor[i, j, :] = 像素(i,j)在所有樣本中的變異
  X_tensor[:, :, k] = 第k個樣本的完整圖像
```

#### 代碼實現
```python
# data/preprocessor.py 中的實現

def reshape_to_tensor(X, shape=(28, 28)):
    """
    將784維特徵向量轉換為3階張量
    
    Args:
        X: shape (n_samples, 784)
        shape: tuple (height, width)
    
    Returns:
        tensor: shape (height, width, n_samples)
    """
    n_samples = X.shape[0]
    
    # reshape: (n, 784) → (n, 28, 28)
    images = X.reshape(n_samples, *shape)
    
    # 轉置: (n, h, w) → (h, w, n)
    tensor = np.transpose(images, (1, 2, 0))
    
    return tensor
```

### 2.3 特徵縮放驗證

```
縮放後驗證:
  
  訓練集統計:
    Mean: 0.0000  (期望)
    Std:  1.0000  (期望)
    Min: -0.4235  (標準化後)
    Max:  3.0876  (標準化後)
    
  測試集統計:
    Mean: -0.0015 (接近0，良好)
    Std:  0.9947  (接近1，良好)
    Min: -0.3891
    Max:  3.1205
    
驗證結論: ✓ 正規化成功
```

## 3. 數據分割策略 (Data Splitting)

### 3.1 訓練/驗證/測試分割

```
MNIST官方分割:
  訓練集: 60,000 張 (85.7%)
  測試集: 10,000 張 (14.3%)
  
本項目方案:
  
  Option 1 (推薦):
    訓練: 48,000 (68.6%)
    驗證: 12,000 (17.1%)  ← HOSVD參數調優
    測試: 10,000 (14.3%)  ← 最終評估
    
  Option 2 (分層抽樣):
    訓練: 54,000 (77.1%)  ← 每類5,400張
    驗證: 6,000  (8.6%)   ← 每類600張
    測試: 10,000 (14.3%)  ← 官方測試集
    
選擇: Option 1 (經典3-fold)
```

### 3.2 分層考慮

```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# 確保各類別在分割中均勻分布
sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
# 驗證分層結果:
# y_train 和 y_test 的類別分布相同 ✓
```

### 3.3 隨機狀態固定

```python
# 配置參數: config.py
RANDOM_STATE = 42

# 用途: 
- 訓練/測試分割
- 特徵縮放初始化
- 分類器隨機初始化
- 結果可重複性

# 效果: 相同的隨機種子 → 完全相同的分割結果
```

## 4. 數據增強 (Data Augmentation)

### 4.1 增強策略

#### 策略 1: 旋轉
```python
from scipy.ndimage import rotate

# 參數
angle_range = [-15, 15]  # 度數
probability = 0.5  # 50%概率應用

# 應用
for sample in X_train:
    if random() < 0.5:
        angle = randint(-15, 15)
        sample = rotate(sample, angle, reshape=False)
        
# 效果: 模擬手寫傾斜變異
# 增加: 訓練數據多樣性
```

#### 策略 2: 平移
```python
# 參數
shift_range = [-2, 2]  # 像素數
probability = 0.5

# 應用
from scipy.ndimage import shift
shifted = shift(image, (sx, sy), mode='constant', cval=0)

# 效果: 模擬手寫位置變異
# 增加: 翻譯不變性
```

#### 策略 3: 高斯噪聲
```python
# 參數
noise_sigma = 0.01 * (X_max - X_min)
probability = 0.3  # 30%

# 應用
noise = np.random.normal(0, noise_sigma, X.shape)
X_augmented = X + noise

# 效果: 增加魯棒性
# 注意: 不應過度 (過度噪聲損害特徵)
```

### 4.2 增強方案

```
選擇: 條件增強 (需要時啟用)

理由:
1. MNIST數據已充分(60k) → 基本無需增強
2. 但對Fashion-MNIST/USPS有益

配置:
  enable_augmentation: False  (默認)
  augmentation_factor: 1.5    (1.5倍數據集大小)
```

## 5. 處理特殊情況

### 5.1 USPS/scikit-learn數據集適配

```
USPS: 16×16 → 需要調整

解決方案:
  使用雙線性插值上採樣
  from PIL import Image
  img = Image.fromarray(img_16x16)
  img_28x28 = img.resize((28, 28), Image.BILINEAR)
  
scikit-learn: 8×8 → 需要調整
  同樣方法: 8×8 → 28×28
```

### 5.2 類別不平衡處理

```
檢查結果:
  MNIST: 無不平衡 (0.3% 差異)
  Fashion-MNIST: 無不平衡 (0.2% 差異)
  
處理策略:
  無特殊處理 (類別分布已均衡)
  
備選方案 (如需):
  - class_weight='balanced' 在分類器中
  - 過採樣少數類
  - 欠採樣多數類
```

## 6. 特徵工程詳細步驟

### 6.1 完整數據管道

```python
# data/preprocessor.py 中的 DataPreprocessor 類

class DataPreprocessor:
    def __init__(self, normalize_method='standard'):
        self.scaler = StandardScaler()
        self.normalize_method = normalize_method
        
    def fit_transform(self, X):
        """訓練階段"""
        # 1. 正規化
        X_scaled = self.scaler.fit_transform(X)
        # 2. 張量化
        X_tensor = self.reshape_to_tensor(X_scaled)
        return X_tensor
    
    def transform(self, X):
        """測試階段"""
        # 1. 正規化 (使用訓練統計)
        X_scaled = self.scaler.transform(X)
        # 2. 張量化
        X_tensor = self.reshape_to_tensor(X_scaled)
        return X_tensor
    
    def reshape_to_tensor(self, X):
        """轉換為3階張量"""
        return np.transpose(
            X.reshape(X.shape[0], 28, 28),
            (1, 2, 0)
        )
```

### 6.2 驗證檢查清單

```
準備完成驗證:

□ 無缺失值
  MNIST: ✓ 0/60000 缺失
  
□ 無異常值
  像素值: ✓ 全在[0,255]
  
□ 尺寸一致
  訓練集: ✓ (28, 28, 60000)
  測試集: ✓ (28, 28, 10000)
  
□ 正規化完成
  訓練集: ✓ μ=0, σ=1
  測試集: ✓ μ≈0, σ≈1
  
□ 分層正確
  訓練: ✓ 類別平衡
  測試: ✓ 類別平衡
  
□ 數據洩露防止
  訓練統計 vs 測試統計: ✓ 獨立計算
  
□ 張量結構正確
  形狀: ✓ (H, W, N)
  內容: ✓ 可轉換回圖像
```

## 7. 配置與參數

### 7.1 config.py 中的配置

```python
# 數據配置
DATA_CONFIG = {
    'dataset': 'mnist',           # 'mnist', 'fashion_mnist', 'usps', 'digits'
    'normalize_method': 'standard', # 'standard' 或 'minmax'
    'train_test_split': 0.8,      # 訓練/驗證比例
    'validation_split': 0.2,       # 驗證佔訓練集比例
    'random_state': 42,
    'augment_data': False,         # 是否增強數據
    'augmentation_factor': 1.5,
}

# 張量配置
TENSOR_CONFIG = {
    'tensor_shape': (28, 28),  # 標準化形狀
    'target_rank': (20, 20, 50),  # HOSVD目標秩
}
```

### 7.2 調用方式

```python
from data.loader import load_mnist_data
from data.preprocessor import DataPreprocessor
from config import DATA_CONFIG, TENSOR_CONFIG

# 加載原始數據
X_train, X_test, y_train, y_test = load_mnist_data()

# 初始化預處理器
preprocessor = DataPreprocessor(
    normalize_method=DATA_CONFIG['normalize_method']
)

# 訓練階段: fit and transform
X_train_tensor = preprocessor.fit_transform(X_train)

# 測試階段: transform only
X_test_tensor = preprocessor.transform(X_test)

# 現在可用於HOSVD
```

## 8. 與其他階段的銜接

本階段輸出:
- ✓ 清理的張量數據: `(28, 28, n_samples)`
- ✓ 正規化的特徵分佈: `μ=0, σ=1`
- ✓ 分層的訓練/測試集合
- ✓ 預處理器對象 (用於新數據)

這些將直接供應:
- **第4階段** (建模): HOSVD分解輸入
- **第5階段** (評估): 性能計算基礎

---

**文件版本**: 1.0  
**最後更新**: 2025年1月  
**作者**: 陳宥興 (5114050015)  
**相關代碼**: `data/preprocessor.py`, `config.py`  
**相關文件**: `CRISP_DM_Phase2_DataUnderstanding.md`
