"""
輔助工具模塊
"""

import os
import json
import pickle
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FileManager:
    """文件管理器"""
    
    @staticmethod
    def save_model(model, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """加載模型"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    @staticmethod
    def save_array(arr, filepath):
        """保存numpy數組"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, arr)
        logger.info(f"Array saved to {filepath}")
    
    @staticmethod
    def load_array(filepath):
        """加載numpy數組"""
        arr = np.load(filepath)
        logger.info(f"Array loaded from {filepath}")
        return arr
    
    @staticmethod
    def save_json(data, filepath):
        """保存JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved to {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """加載JSON文件"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON loaded from {filepath}")
        return data


class Logger:
    """日誌配置"""
    
    @staticmethod
    def setup_logger(name, log_level=logging.INFO):
        """設置日誌"""
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger


class ProgressTracker:
    """進度跟蹤器"""
    
    def __init__(self, total_steps, prefix=""):
        """
        初始化
        
        Args:
            total_steps: 總步數
            prefix: 前綴
        """
        self.total_steps = total_steps
        self.prefix = prefix
        self.current_step = 0
        self.start_time = None
    
    def start(self):
        """開始計時"""
        self.start_time = datetime.now()
    
    def update(self, step=1):
        """更新進度"""
        self.current_step += step
        self._print_progress()
    
    def _print_progress(self):
        """打印進度"""
        progress = self.current_step / self.total_steps
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\r{self.prefix} |{bar}| {progress*100:.1f}% ({self.current_step}/{self.total_steps}) "
              f"Time: {elapsed:.1f}s", end='', flush=True)


def validate_input(X, y=None, min_samples=1, min_features=1):
    """
    驗證輸入數據
    
    Args:
        X: 特徵矩陣
        y: 標籤（可選）
        min_samples: 最少樣本數
        min_features: 最少特徵數
    
    Returns:
        驗證結果（布爾值）
    """
    if not isinstance(X, np.ndarray):
        logger.error("X must be a numpy array")
        return False
    
    if len(X.shape) != 2:
        logger.error("X must be 2-dimensional")
        return False
    
    if X.shape[0] < min_samples:
        logger.error(f"X has fewer than {min_samples} samples")
        return False
    
    if X.shape[1] < min_features:
        logger.error(f"X has fewer than {min_features} features")
        return False
    
    if y is not None:
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if len(y) != X.shape[0]:
            logger.error("X and y must have the same number of samples")
            return False
    
    return True


def compute_statistics(X, return_dict=False):
    """
    計算數據統計
    
    Args:
        X: 數據矩陣
        return_dict: 是否返回字典格式
    
    Returns:
        統計信息
    """
    stats = {
        'shape': X.shape,
        'mean': X.mean(axis=0).mean(),
        'std': X.std(axis=0).mean(),
        'min': X.min(),
        'max': X.max(),
        'median': np.median(X),
    }
    
    if return_dict:
        return stats
    else:
        msg = f"Data shape: {stats['shape']}\n"
        msg += f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}\n"
        msg += f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}\n"
        msg += f"Median: {stats['median']:.4f}"
        return msg


def memory_usage(X):
    """
    計算內存使用量
    
    Args:
        X: 數據
    
    Returns:
        內存大小（MB）
    """
    return X.nbytes / (1024 ** 2)


if __name__ == "__main__":
    # 測試
    logger = Logger.setup_logger("test")
    logger.info("Test logging")
    
    X = np.random.rand(100, 784)
    print(compute_statistics(X))
    print(f"Memory usage: {memory_usage(X):.2f} MB")
