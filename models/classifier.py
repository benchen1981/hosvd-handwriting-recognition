"""
分類器模塊 - 集成多種分類算法
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import logging

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """分類器管道"""
    
    def __init__(self, classifier_type='knn', **kwargs):
        """
        初始化分類器
        
        Args:
            classifier_type: 'knn', 'svm', 'rf', 'mlp'
            **kwargs: 分類器參數
        """
        self.classifier_type = classifier_type
        self.classifier = self._create_classifier(classifier_type, **kwargs)
        self.fitted = False
    
    def _create_classifier(self, classifier_type, **kwargs):
        """創建分類器實例"""
        if classifier_type == 'knn':
            n_neighbors = kwargs.get('n_neighbors', 5)
            weights = kwargs.get('weights', 'uniform')
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
        elif classifier_type == 'svm':
            kernel = kwargs.get('kernel', 'rbf')
            C = kwargs.get('C', 1.0)
            gamma = kwargs.get('gamma', 'scale')
            return SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        
        elif classifier_type == 'rf':
            n_estimators = kwargs.get('n_estimators', 100)
            max_depth = kwargs.get('max_depth', None)
            random_state = kwargs.get('random_state', 42)
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        
        elif classifier_type == 'mlp':
            hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (256, 128, 64))
            learning_rate = kwargs.get('learning_rate', 'adaptive')
            max_iter = kwargs.get('max_iter', 200)
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate=learning_rate,
                max_iter=max_iter,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def fit(self, X, y):
        """
        訓練分類器
        
        Args:
            X: 訓練特徵
            y: 訓練標籤
        """
        logger.info(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X):
        """預測標籤"""
        if not self.fitted:
            raise RuntimeError("Classifier not fitted yet")
        return self.classifier.predict(X)
    
    def predict_proba(self, X):
        """預測概率"""
        if not self.fitted:
            raise RuntimeError("Classifier not fitted yet")
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X)
        else:
            logger.warning(f"{self.classifier_type} does not support predict_proba")
            return None
    
    def score(self, X, y):
        """計算精度"""
        if not self.fitted:
            raise RuntimeError("Classifier not fitted yet")
        return self.classifier.score(X, y)
    
    def get_feature_importance(self):
        """
        獲取特徵重要性（僅支持某些分類器）
        
        Returns:
            特徵重要性數組，如果不支持返回None
        """
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            return self.classifier.coef_
        else:
            return None


class EnsembleClassifier:
    """集成分類器 - 組合多個分類器"""
    
    def __init__(self, classifiers=None, weights=None):
        """
        初始化集成分類器
        
        Args:
            classifiers: 分類器列表
            weights: 各分類器的權重
        """
        self.classifiers = classifiers or []
        self.weights = weights or [1.0] * len(classifiers)
        self.fitted = False
    
    def add_classifier(self, classifier, weight=1.0):
        """添加分類器"""
        self.classifiers.append(classifier)
        self.weights.append(weight)
    
    def fit(self, X, y):
        """訓練所有分類器"""
        logger.info(f"Training ensemble with {len(self.classifiers)} classifiers...")
        for clf in self.classifiers:
            clf.fit(X, y)
        self.fitted = True
        return self
    
    def predict(self, X):
        """投票預測"""
        if not self.fitted:
            raise RuntimeError("Ensemble not fitted yet")
        
        predictions = []
        for clf, weight in zip(self.classifiers, self.weights):
            pred = clf.predict(X)
            predictions.append(pred * weight)
        
        # 計算加權投票結果
        ensemble_pred = predictions[0]
        for pred in predictions[1:]:
            ensemble_pred = ensemble_pred + pred
        
        return ensemble_pred.astype(int)
    
    def score(self, X, y):
        """計算精度"""
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy


def create_classifier(classifier_type='knn', **kwargs):
    """
    工廠函數 - 創建分類器
    
    Args:
        classifier_type: 分類器類型
        **kwargs: 分類器參數
    
    Returns:
        分類器實例
    """
    return ClassifierPipeline(classifier_type, **kwargs)


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    # 測試
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )
    
    # 測試KNN
    knn = ClassifierPipeline('knn', n_neighbors=5)
    knn.fit(X_train, y_train)
    print(f"KNN accuracy: {knn.score(X_test, y_test):.4f}")
    
    # 測試SVM
    svm = ClassifierPipeline('svm')
    svm.fit(X_train, y_train)
    print(f"SVM accuracy: {svm.score(X_test, y_test):.4f}")
    
    # 測試隨機森林
    rf = ClassifierPipeline('rf', n_estimators=100)
    rf.fit(X_train, y_train)
    print(f"RF accuracy: {rf.score(X_test, y_test):.4f}")
