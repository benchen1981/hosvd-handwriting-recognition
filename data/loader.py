"""
數據載入模塊 - 從sklearn載入標準數據集
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist
import logging

logger = logging.getLogger(__name__)


def load_mnist_data(test_size=0.2, random_state=42, normalize=True):
    """
    載入MNIST手寫數字數據集
    
    Args:
        test_size: 測試集比例
        random_state: 隨機種子
        normalize: 是否歸一化到[0,1]
    
    Returns:
        X_train, y_train, X_test, y_test: 訓練和測試數據
    """
    logger.info("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    # 展平為2D數組
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    logger.info(f"MNIST loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_fashion_mnist_data(test_size=0.2, random_state=42, normalize=True):
    """
    載入Fashion-MNIST數據集
    
    Args:
        test_size: 測試集比例
        random_state: 隨機種子
        normalize: 是否歸一化到[0,1]
    
    Returns:
        X_train, y_train, X_test, y_test: 訓練和測試數據
    """
    logger.info("Loading Fashion-MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    if normalize:
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
    
    # 展平為2D數組
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    logger.info(f"Fashion-MNIST loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_sklearn_digits(test_size=0.2, random_state=42, normalize=True):
    """
    載入sklearn的digits數據集（8x8手寫數字）
    
    Args:
        test_size: 測試集比例
        random_state: 隨機種子
        normalize: 是否歸一化
    
    Returns:
        X_train, y_train, X_test, y_test: 訓練和測試數據
    """
    logger.info("Loading sklearn Digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    if normalize:
        X = X.astype('float32') / X.max()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Digits loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, y_train, X_test, y_test


def load_data(dataset='mnist', test_size=0.2, random_state=42, normalize=True):
    """
    通用數據加載函數
    
    Args:
        dataset: 'mnist', 'fashion_mnist', 'digits'
        test_size: 測試集比例
        random_state: 隨機種子
        normalize: 是否歸一化
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    if dataset == 'mnist':
        return load_mnist_data(test_size, random_state, normalize)
    elif dataset == 'fashion_mnist':
        return load_fashion_mnist_data(test_size, random_state, normalize)
    elif dataset == 'digits':
        return load_sklearn_digits(test_size, random_state, normalize)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    # 測試
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
