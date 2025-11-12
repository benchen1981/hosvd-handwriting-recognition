"""
HOSVD Handwriting Recognition System
高階奇異值分解手寫辨識系統

版本: 1.0.0
作者: 陳宥興
學號: 5114050015
"""

__version__ = "1.0.0"
__author__ = "Shen Zhen-Xun"
__email__ = "5114050015@nchu.edu.tw"

from . import config
from . import data
from . import models
from . import utils

__all__ = [
    'config',
    'data',
    'models', 
    'utils',
]

print(f"HOSVD Handwriting Recognition System v{__version__}")
print("Successfully initialized!")
