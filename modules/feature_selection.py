from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import pandas as pd

def get_feature_importance(model, feature_names):
    """获取特征重要性
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        
    Returns:
        特征重要性字典
    """
    if hasattr(model, 'feature_importances_'):
        # 树模型
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # 线性模型
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # 排序
    importance_dict = {}
    for feature, importance in zip(feature_names, importances):
        importance_dict[feature] = importance
    
    # 按重要性排序
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance

def select_top_features(X, y, feature_names, k=10, method='tree'):
    """选择最重要的特征
    
    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称列表
        k: 选择的特征数量
        method: 选择方法，'tree'、'f_classif'或'mutual_info'
        
    Returns:
        选中的特征名称列表
    """
    if method == 'tree':
        # 使用ExtraTreesClassifier获取特征重要性
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:k]
        selected_features = [feature_names[i] for i in indices]
    
    elif method == 'f_classif':
        # 使用F检验
        selector = SelectKBest(f_classif, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = [feature for feature, selected in zip(feature_names, mask) if selected]
    
    elif method == 'mutual_info':
        # 使用互信息
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected_features = [feature for feature, selected in zip(feature_names, mask) if selected]
    
    else:
        raise ValueError("不支持的特征选择方法")
    
    return selected_features

def rfe_feature_selection(X, y, feature_names, k=10, estimator=None):
    """使用RFE进行特征选择
    
    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称列表
        k: 选择的特征数量
        estimator: 用于RFE的基模型
        
    Returns:
        选中的特征名称列表
    """
    if estimator is None:
        # 默认使用ExtraTreesClassifier
        estimator = ExtraTreesClassifier(n_estimators=100, random_state=42)
    
    # 创建RFE选择器
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X, y)
    
    # 获取选中的特征
    selected_features = [feature for feature, selected in zip(feature_names, rfe.support_) if selected]
    
    return selected_features

def analyze_feature_correlation(X, feature_names, threshold=0.8):
    """分析特征相关性，剔除高度相关的特征
    
    Args:
        X: 特征矩阵
        feature_names: 特征名称列表
        threshold: 相关系数阈值
        
    Returns:
        保留的特征名称列表
    """
    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(X, rowvar=False)
    correlation_df = pd.DataFrame(correlation_matrix, columns=feature_names, index=feature_names)
    
    # 找到高度相关的特征对
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(correlation_matrix[i, j]) > threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix[i, j]))
    
    # 剔除高度相关的特征
    to_remove = set()
    for feature1, feature2, corr in high_corr_pairs:
        # 简单策略：保留第一个出现的特征
        if feature2 not in to_remove:
            to_remove.add(feature2)
    
    # 保留的特征
    selected_features = [feature for feature in feature_names if feature not in to_remove]
    
    return selected_features, high_corr_pairs
