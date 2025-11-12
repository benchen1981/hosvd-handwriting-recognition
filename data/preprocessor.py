"""
數據預處理模塊
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """數據預處理類"""
    
    def __init__(self, normalize=True, standardize=False):
        """
        初始化預處理器
        
        Args:
            normalize: 是否歸一化到[0,1]
            standardize: 是否標準化（mean=0, std=1）
        """
        self.normalize = normalize
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None
        self.fitted = False
    
    def fit(self, X):
        """
        擬合預處理器
        
        Args:
            X: 輸入數據，形狀(n_samples, n_features)
        """
        if self.standardize:
            self.scaler.fit(X)
        self.fitted = True
        logger.info("Preprocessor fitted")
        return self
    
    def transform(self, X):
        """
        轉換數據
        
        Args:
            X: 輸入數據
        
        Returns:
            轉換後的數據
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted yet")
        
        X_copy = X.copy()
        
        if self.normalize:
            X_copy = (X_copy - X_copy.min()) / (X_copy.max() - X_copy.min() + 1e-8)
        
        if self.standardize:
            X_copy = self.scaler.transform(X_copy)
        
        return X_copy
    
    def fit_transform(self, X):
        """
        擬合並轉換數據
        
        Args:
            X: 輸入數據
        
        Returns:
            轉換後的數據
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        反向轉換（目前只支持標準化）
        
        Args:
            X: 轉換後的數據
        
        Returns:
            原始尺度的數據
        """
        if self.standardize:
            return self.scaler.inverse_transform(X)
        return X


def augment_data(X, y, rotation_range=15, shift_range=0.1, noise_level=0.01):
    """
    數據增強
    
    Args:
        X: 輸入數據，形狀(n_samples, n_features)
        y: 標籤
        rotation_range: 旋轉角度範圍
        shift_range: 平移範圍（相對於影像大小）
        noise_level: 噪聲水平
    
    Returns:
        增強後的數據和標籤
    """
    from scipy.ndimage import rotate
    
    # 假設影像是28x28或32x32
    side = int(np.sqrt(X.shape[1]))
    
    X_augmented = [X]
    y_augmented = [y]
    
    logger.info(f"Augmenting data with {rotation_range}° rotation and {shift_range} shift")
    
    for i in range(X.shape[0]):
        img = X[i].reshape(side, side)
        
        # 旋轉
        angle = np.random.uniform(-rotation_range, rotation_range)
        img_rotated = rotate(img, angle, reshape=False)
        X_augmented.append(img_rotated.flatten())
        y_augmented.append(y[i])
        
        # 加噪聲
        img_noisy = img + np.random.normal(0, noise_level, img.shape)
        img_noisy = np.clip(img_noisy, 0, 1)
        X_augmented.append(img_noisy.flatten())
        y_augmented.append(y[i])
    
    X_augmented = np.vstack(X_augmented)
    y_augmented = np.hstack(y_augmented)
    
    logger.info(f"Data augmentation completed: {X.shape[0]} -> {X_augmented.shape[0]}")
    
    return X_augmented, y_augmented


if __name__ == "__main__":
    # 測試
    X = np.random.rand(100, 784)
    y = np.random.randint(0, 10, 100)
    
    preprocessor = DataPreprocessor(normalize=True, standardize=True)
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Original data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Processed data range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
    print(f"Processed data mean: {X_processed.mean():.6f}, std: {X_processed.std():.6f}")
