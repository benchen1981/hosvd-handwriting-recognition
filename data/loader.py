"""
數據載入模塊 - 從sklearn載入標準數據集
"""

import numpy as np
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

# 嘗試導入 Keras 數據集，如果失敗則使用 sklearn digits
try:
    try:
        from tensorflow.keras.datasets import mnist, fashion_mnist
        _has_keras = True
    except ImportError:
        from keras.datasets import mnist, fashion_mnist
        _has_keras = True
except (ImportError, ModuleNotFoundError):
    _has_keras = False
    logger.warning("Keras/TensorFlow not available, will use sklearn digits dataset")


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
    
    if _has_keras:
        try:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            
            if normalize:
                X_train = X_train.astype('float32') / 255.0
                X_test = X_test.astype('float32') / 255.0
            
            # 展平為2D數組
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        except Exception as e:
            logger.warning(f"Failed to load Keras MNIST: {e}. Using sklearn digits instead.")
            return _load_mnist_from_sklearn(test_size, random_state, normalize)
    else:
        return _load_mnist_from_sklearn(test_size, random_state, normalize)
    
    logger.info(f"MNIST loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
    return X_train, y_train, X_test, y_test


def _load_mnist_from_sklearn(test_size=0.2, random_state=42, normalize=True):
    """
    使用 sklearn 的 digits 數據集作為 MNIST 替代品
    """
    logger.info("Loading digits dataset from sklearn...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    if normalize:
        X = X.astype('float32') / 16.0  # digits 值範圍 0-16
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"sklearn digits loaded: Train shape {X_train.shape}, Test shape {X_test.shape}")
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
    
    if _has_keras:
        try:
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
            
            if normalize:
                X_train = X_train.astype('float32') / 255.0
                X_test = X_test.astype('float32') / 255.0
            
            # 展平為2D數組
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        except Exception as e:
            logger.warning(f"Failed to load Keras Fashion-MNIST: {e}. Using sklearn digits instead.")
            return _load_mnist_from_sklearn(test_size, random_state, normalize)
    else:
        logger.warning("Keras/TensorFlow not available for Fashion-MNIST. Using sklearn digits instead.")
        return _load_mnist_from_sklearn(test_size, random_state, normalize)
    
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
    elif dataset in ('usps', 'ups', 'usps10'):
        # 支援 USPS (user-provided) 資料集；如果本地沒有，退回到 sklearn digits
        try:
            return _load_usps_local(test_size, random_state, normalize)
        except Exception as e:
            logger.warning(f"USPS load failed: {e}. Falling back to sklearn digits.")
            return load_sklearn_digits(test_size, random_state, normalize)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _load_usps_local(test_size=0.2, random_state=42, normalize=True):
    """
    嘗試從本地目錄或檔案載入 USPS 資料集。
    支援 data/usps.npz、data/usps.npy 或 data/usps.csv 格式（若使用者已下載 Kaggle 資料並放到 data/ 目錄）。
    若找不到或載入失敗，會拋出例外以讓 caller fallback。
    """
    logger.info("Attempting to load USPS dataset from local files...")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    # 判斷常見檔名
    candidates = [
        os.path.join(base_dir, 'usps.npz'),
        os.path.join(base_dir, 'usps.npy'),
        os.path.join(base_dir, 'usps.csv'),
        os.path.join(base_dir, 'USPS', 'usps.npy'),
    ]

    for p in candidates:
        if os.path.exists(p):
            logger.info(f"Found USPS file: {p}")
            if p.endswith('.npz'):
                data = np.load(p)
                X = data['X'] if 'X' in data else data['x'] if 'x' in data else data.get('arr_0')
                y = data['y'] if 'y' in data else data.get('arr_1')
            elif p.endswith('.npy'):
                arr = np.load(p)
                # assume (X,y) saved as object array
                if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 2:
                    X, y = arr[0], arr[1]
                else:
                    raise ValueError('Unknown .npy format for USPS')
            elif p.endswith('.csv'):
                import pandas as _pd
                df = _pd.read_csv(p)
                if 'label' in df.columns:
                    y = df['label'].values
                    X = df.drop(columns=['label']).values
                else:
                    raise ValueError('CSV must contain a label column')
            else:
                raise ValueError('Unsupported USPS file format')

            if normalize:
                # USPS grayscale often 0-255 or 0-1; attempt safe normalization
                X = X.astype('float32')
                if X.max() > 1.0:
                    X = X / 255.0

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info(f"USPS loaded from local: Train shape {X_train.shape}, Test shape {X_test.shape}")
            return X_train, y_train, X_test, y_test

    raise FileNotFoundError('No local USPS file found in data/ (usps.npz|usps.npy|usps.csv)')


if __name__ == "__main__":
    # 測試
    X_train, y_train, X_test, y_test = load_mnist_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
