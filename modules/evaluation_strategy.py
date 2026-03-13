from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def adversarial_validation(X_train, X_test):
    """对抗性验证：评估训练集和测试集的分布差异
    
    Args:
        X_train: 训练集特征
        X_test: 测试集特征
        
    Returns:
        accuracy: 分类器区分训练集和测试集的准确率
        fig: 可视化结果
    """
    # 创建标签：0表示训练集，1表示测试集
    y_train_label = np.zeros(len(X_train))
    y_test_label = np.ones(len(X_test))
    
    # 合并数据
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train_label, y_test_label])
    
    # 训练分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_combined, y_combined)
    
    # 预测
    y_pred = model.predict(X_combined)
    accuracy = accuracy_score(y_combined, y_pred)
    
    # 计算特征重要性
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 准确率可视化
    ax1.bar(['训练集', '测试集'], [len(X_train), len(X_test)], color=['blue', 'orange'])
    ax1.set_xlabel('数据集')
    ax1.set_ylabel('样本数量')
    ax1.set_title(f'对抗性验证结果\n准确率: {accuracy:.4f}')
    ax1.text(0.5, 0.9, f'准确率: {accuracy:.4f}', transform=ax1.transAxes, ha='center')
    
    # 特征重要性可视化
    ax2.bar(range(min(10, len(importances))), importances[indices[:10]])
    ax2.set_xlabel('特征索引')
    ax2.set_ylabel('重要性')
    ax2.set_title('区分训练集和测试集的重要特征')
    
    return accuracy, fig

def plot_learning_curve(estimator, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """绘制学习曲线
    
    Args:
        estimator: 评估器
        X: 特征矩阵
        y: 标签
        cv: 交叉验证策略
        n_jobs: 并行任务数
        train_sizes: 训练集大小
        
    Returns:
        fig: 学习曲线可视化
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    
    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制训练曲线
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
    
    # 绘制测试曲线
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, test_mean, 'o-', color='orange', label='交叉验证分数')
    
    ax.set_xlabel('训练样本数')
    ax.set_ylabel('分数')
    ax.set_title('学习曲线')
    ax.legend(loc='best')
    
    return fig
