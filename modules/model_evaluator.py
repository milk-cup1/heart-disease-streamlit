from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    return y_pred, y_pred_proba, report, cm, auc_score

def find_optimal_threshold(y_true, y_proba):
    """寻找最优阈值"""
    from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
    
    # 计算PR曲线
    pr_precision, pr_recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # 计算每个阈值的F1分数
    f1_scores = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    # 找到最大F1对应的阈值
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    
    # 生成详细的阈值分析（0.1到0.9，步长0.05）
    detailed_thresholds = np.arange(0.1, 0.91, 0.05)
    detailed_metrics = []
    
    for t in detailed_thresholds:
        y_pred = (y_proba >= t).astype(int)
        detailed_precision = precision_score(y_true, y_pred, zero_division=0)
        detailed_recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        detailed_metrics.append({
            'threshold': t,
            'precision': detailed_precision,
            'recall': detailed_recall,
            'f1_score': f1
        })
    
    return optimal_threshold, best_f1, pr_precision, pr_recall, thresholds, f1_scores, detailed_metrics

def plot_confusion_matrix(cm, model_name):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    return fig

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.2f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """绘制特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        features, importance_values = zip(*feat_importances)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance_values, y=features, ax=ax)
        plt.title(f'{model_name} Feature Importance')
        return fig
    elif hasattr(model, 'coef_'):
        # 对于逻辑回归
        coef = model.coef_[0]
        feat_importances = sorted(zip(feature_names, abs(coef)), key=lambda x: x[1], reverse=True)
        
        features, importance_values = zip(*feat_importances)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance_values, y=features, ax=ax)
        plt.title(f'{model_name} Feature Importance (Coefficient Magnitude)')
        return fig
    else:
        return None

def calculate_sensitivity_specificity(y_test, y_pred):
    """计算敏感度和特异度"""
    from sklearn.metrics import recall_score
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    return sensitivity, specificity