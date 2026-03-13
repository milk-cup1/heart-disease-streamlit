from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class ModelEnsemble:
    """模型融合类"""
    
    def __init__(self, models):
        """初始化模型融合
        
        Args:
            models: 字典，键为模型名称，值为预训练模型
        """
        self.models = models
    
    def soft_voting(self, X, weights=None):
        """软投票融合
        
        Args:
            X: 输入特征
            weights: 权重列表，与模型顺序对应
            
        Returns:
            预测概率
        """
        # 获取所有模型的预测概率
        probabilities = []
        model_names = list(self.models.keys())
        
        for model_name in model_names:
            model = self.models[model_name]
            prob = model.predict_proba(X)[:, 1]
            probabilities.append(prob)
        
        # 计算加权平均
        probabilities = np.array(probabilities)
        
        if weights is None:
            # 等权重
            avg_prob = np.mean(probabilities, axis=0)
        else:
            # 自定义权重
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # 归一化
            avg_prob = np.dot(weights, probabilities)
        
        return avg_prob
    
    def stacking(self, X_train, y_train, X_test, meta_model=None):
        """Stacking融合
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            meta_model: 元模型，默认为逻辑回归
            
        Returns:
            预测概率
        """
        # 划分训练集为两部分，一部分用于训练基模型，一部分用于训练元模型
        X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42, stratify=y_train
        )
        
        # 训练基模型
        base_models = {}
        for model_name, model in self.models.items():
            base_model = model.__class__(**model.get_params())
            base_model.fit(X_train_base, y_train_base)
            base_models[model_name] = base_model
        
        # 生成基模型的预测作为元特征
        meta_features_train = []
        for model_name, model in base_models.items():
            prob = model.predict_proba(X_train_meta)[:, 1]
            meta_features_train.append(prob)
        meta_features_train = np.array(meta_features_train).T
        
        meta_features_test = []
        for model_name, model in base_models.items():
            prob = model.predict_proba(X_test)[:, 1]
            meta_features_test.append(prob)
        meta_features_test = np.array(meta_features_test).T
        
        # 训练元模型
        if meta_model is None:
            meta_model = LogisticRegression(class_weight='balanced', C=0.1)
        meta_model.fit(meta_features_train, y_train_meta)
        
        # 预测
        final_prob = meta_model.predict_proba(meta_features_test)[:, 1]
        
        return final_prob
    
    def evaluate_ensemble(self, X_test, y_test, method='soft_voting', weights=None, X_train=None, y_train=None):
        """评估融合模型
        
        Args:
            X_test: 测试集特征
            y_test: 测试集标签
            method: 融合方法，'soft_voting'或'stacking'
            weights: 软投票权重
            X_train: 训练集特征（stacking需要）
            y_train: 训练集标签（stacking需要）
            
        Returns:
            评估指标字典
        """
        if method == 'soft_voting':
            y_pred_proba = self.soft_voting(X_test, weights)
        elif method == 'stacking':
            if X_train is None or y_train is None:
                raise ValueError("Stacking方法需要提供训练集数据")
            y_pred_proba = self.stacking(X_train, y_train, X_test)
        else:
            raise ValueError("不支持的融合方法")
        
        # 计算评估指标
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
