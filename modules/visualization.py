import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi

def plot_model_comparison(comparison_df):
    """绘制模型对比条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].plot(kind='bar', ax=ax)
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    return fig

def plot_model_radar(comparison_df):
    """绘制模型性能雷达图"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    N = len(categories)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for idx, row in comparison_df.iterrows():
        values = row[categories].values.tolist()
        values += values[:1]  # 闭合图形
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
    plt.title('Model Performance Radar Chart')
    
    return fig

def plot_correlation_heatmap(df):
    """绘制相关性热力图"""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature Correlation Heatmap')
    return fig

def plot_feature_distribution(df, feature_name):
    """绘制特征分布"""
    fig, ax = plt.subplots()
    sns.histplot(df[feature_name], kde=True, ax=ax)
    plt.title(f'{feature_name} Distribution')
    return fig

def plot_target_distribution(df, target_col='target'):
    """绘制目标变量分布"""
    target_counts = df[target_col].value_counts()
    fig, ax = plt.subplots()
    target_counts.plot(kind='bar', ax=ax)
    plt.title('Target Variable Distribution')
    plt.xlabel('Target')
    plt.ylabel('Count')
    return fig