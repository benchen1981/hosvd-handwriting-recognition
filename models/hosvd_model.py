"""
HOSVD張量分解模型
"""

import numpy as np
from scipy import linalg
import tensorly as tl

# 處理 Tensorly 版本相容性
# Tensorly 0.9.0+ 使用 tucker 取代 higher_order_svd
try:
    from tensorly.decomposition import higher_order_svd
except (ImportError, ModuleNotFoundError):
    from tensorly.decomposition import tucker as higher_order_svd

import logging

logger = logging.getLogger(__name__)


class HOSVDModel:
    """
    高階奇異值分解（Higher-Order SVD）模型
    用於多維張量的分解和降維
    """
    
    def __init__(self, n_components=50, n_modes=3, random_state=42):
        """
        初始化HOSVD模型
        
        Args:
            n_components: 核心張量最大維度
            n_modes: 張量的階數（維度數）
            random_state: 隨機種子
        """
        self.n_components = n_components
        self.n_modes = n_modes
        self.random_state = random_state
        self.core_tensor = None
        self.factors = None
        self.original_shape = None
        self.fitted = False
    
    def _reshape_to_tensor(self, X):
        """
        將2D數組重塑為三階張量
        
        Args:
            X: 輸入數據，形狀(n_samples, n_features)
        
        Returns:
            三階張量，形狀(n_samples, h, w)
        """
        n_samples = X.shape[0]
        side = int(np.sqrt(X.shape[1]))
        
        if side * side != X.shape[1]:
            # 如果不是完全平方，使用最接近的值
            side = int(np.sqrt(X.shape[1]))
            X = X[:, :side*side]
        
        tensor = X.reshape(n_samples, side, side)
        return tensor
    
    def _reshape_to_matrix(self, tensor):
        """
        將張量重塑為矩陣
        
        Args:
            tensor: 三階張量
        
        Returns:
            二維矩陣
        """
        return tensor.reshape(tensor.shape[0], -1)
    
    def fit(self, X):
        """
        擬合HOSVD模型
        
        Args:
            X: 輸入數據，形狀(n_samples, n_features)
        
        Returns:
            self
        """
        logger.info(f"Fitting HOSVD model with {X.shape[0]} samples...")
        
        self.original_shape = X.shape
        
        # 將數據重塑為張量
        tensor = self._reshape_to_tensor(X)
        logger.info(f"Tensor shape: {tensor.shape}")
        
        # 設置核心張量的形狀
        ranks = [min(self.n_components, s) for s in tensor.shape]
        logger.info(f"Core tensor shape: {tuple(ranks)}")
        
        # 執行HOSVD分解
        try:
            self.core_tensor, self.factors = higher_order_svd(
                tensor, rank=ranks, full_matrices=False
            )
            logger.info("HOSVD decomposition completed")
        except Exception as e:
            logger.error(f"HOSVD decomposition failed: {e}")
            raise
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """
        轉換數據
        
        Args:
            X: 輸入數據，形狀(n_samples, n_features)
        
        Returns:
            降維後的數據
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        
        # 重塑為張量
        tensor = self._reshape_to_tensor(X)
        
        # 應用因子矩陣進行投影
        # tensor_reduced = G ×_1 U_1^T ×_2 U_2^T ×_3 U_3^T
        tensor_reduced = tensor.copy()
        for n, factor in enumerate(self.factors):
            # 沿第n+1個模式進行模態乘積
            tensor_reduced = self._mode_product(tensor_reduced, factor.T, n + 1)
        
        # 轉換為矩陣形式
        X_reduced = self._reshape_to_matrix(tensor_reduced)
        
        return X_reduced
    
    def fit_transform(self, X):
        """
        擬合並轉換數據
        
        Args:
            X: 輸入數據
        
        Returns:
            降維後的數據
        """
        self.fit(X)
        return self.transform(X)
    
    @staticmethod
    def _mode_product(tensor, matrix, mode):
        """
        計算張量與矩陣的模態乘積
        
        Args:
            tensor: 輸入張量
            matrix: 矩陣
            mode: 模態（從1開始）
        
        Returns:
            乘積結果
        """
        # 移動mode軸到第0位
        tensor = np.moveaxis(tensor, mode - 1, 0)
        
        # 進行矩陣乘積
        shape = tensor.shape
        tensor_reshaped = tensor.reshape(shape[0], -1)
        result = matrix @ tensor_reshaped
        
        # 重塑並移動軸回原位置
        result = result.reshape([matrix.shape[0]] + list(shape[1:]))
        result = np.moveaxis(result, 0, mode - 1)
        
        return result
    
    def get_core_tensor_shape(self):
        """獲取核心張量形狀"""
        if self.core_tensor is None:
            return None
        return self.core_tensor.shape
    
    def get_factors_info(self):
        """獲取因子矩陣信息"""
        if self.factors is None:
            return None
        
        info = []
        for i, factor in enumerate(self.factors):
            info.append(f"Factor {i}: {factor.shape}")
        return info
    
    def get_compression_ratio(self):
        """計算壓縮比"""
        if not self.fitted:
            return None
        
        original_size = np.prod(self.original_shape)
        core_size = np.prod(self.core_tensor.shape)
        factors_size = sum(np.prod(f.shape) for f in self.factors)
        compressed_size = core_size + factors_size
        
        ratio = compressed_size / original_size
        return ratio
    
    def get_reconstruction_error(self, X):
        """
        計算重建誤差
        
        Args:
            X: 原始數據
        
        Returns:
            相對誤差（%）
        """
        X_reduced = self.transform(X)
        
        # 反向轉換（簡化版，不包含完整重建）
        # 這裡只是計算投影後的誤差
        original_norm = np.linalg.norm(X)
        reduced_norm = np.linalg.norm(X_reduced)
        
        error = (1 - reduced_norm / (original_norm + 1e-8)) * 100
        return error


class HOSVDClassifier:
    """
    使用HOSVD進行特徵提取的分類器包裝
    """
    
    def __init__(self, n_components=50, classifier=None):
        """
        初始化
        
        Args:
            n_components: HOSVD的主成分數
            classifier: 下游分類器（如KNN、SVM等）
        """
        self.hosvd = HOSVDModel(n_components=n_components)
        self.classifier = classifier
        self.fitted = False
    
    def fit(self, X, y):
        """
        擬合模型
        
        Args:
            X: 訓練數據
            y: 標籤
        """
        logger.info("Fitting HOSVD classifier...")
        self.hosvd.fit(X)
        X_reduced = self.hosvd.transform(X)
        
        if self.classifier is not None:
            self.classifier.fit(X_reduced, y)
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """預測"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_reduced = self.hosvd.transform(X)
        return self.classifier.predict(X_reduced)
    
    def predict_proba(self, X):
        """預測概率"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        
        X_reduced = self.hosvd.transform(X)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_reduced)
        else:
            raise NotImplementedError("Classifier does not support predict_proba")
    
    def score(self, X, y):
        """評分"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


if __name__ == "__main__":
    # 測試
    np.random.seed(42)
    X = np.random.rand(100, 784)
    
    hosvd = HOSVDModel(n_components=50)
    X_reduced = hosvd.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Core tensor shape: {hosvd.get_core_tensor_shape()}")
    print(f"Compression ratio: {hosvd.get_compression_ratio():.4f}")
    print(f"Reconstruction error: {hosvd.get_reconstruction_error(X):.2f}%")
