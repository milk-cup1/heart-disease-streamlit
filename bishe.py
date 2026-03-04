# 心脏病风险预测代码模板（基于机器学习）
# By milk-cup1 | 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap

# 1. 数据载入
# 使用UCI心脏病数据集（processed.cleveland.data）
# 特征名称：age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df = pd.read_csv('heart+disease/processed.cleveland.data', names=columns)

# 处理缺失值（将?替换为NaN）
df = df.replace('?', pd.NA)

# 将目标变量转换为二分类（0=无心脏病，1=有心脏病）
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# 2. 基本数据探索
print(df.head())
print(df.info())
print(df.describe())
print(df['target'].value_counts())  # target=1 表示有心脏病，0表示无

# 处理缺失值（删除含有缺失值的行）
df = df.dropna()

# 将ca和thal列转换为数值类型
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

print(f"处理后的数据形状: {df.shape}")

# 3. 特征与标签分离
X = df.drop('target', axis=1)
y = df['target']

# 4. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# 5. 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. 多模型训练与评估
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'\n模型: {name}')
    print(classification_report(y_test, y_pred))
    # 交叉验证得分
    cv_score = cross_val_score(model, X, y, cv=5).mean()
    print(f"5折交叉验证均值: {cv_score:.4f}")
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'cv_score': cv_score,
        'auc': roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    }

# 7. 混淆矩阵与ROC曲线
plt.figure(figsize=(16,5))
for idx, (name, result) in enumerate(results.items()):
    plt.subplot(1, len(results), idx + 1)
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(name)
    plt.xlabel('预测')
    plt.ylabel('实际')
plt.tight_layout()
plt.show()

# 可视化ROC
plt.figure(figsize=(8,6))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['model'].predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f'{name} (AUC={result["auc"]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('模型ROC曲线')
plt.legend()
plt.show()

# 8. 特征重要性（以随机森林为例）
rf = results['Random Forest']['model']
importances = rf.feature_importances_
feat_names = df.drop('target', axis=1).columns
imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
imp_df = imp_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=imp_df)
plt.title('特征重要性（Random Forest）')
plt.show()

# 9. SHAP可解释性分析（以XGBoost为例）
print('\nSHAP可解释性分析：')
xgb_model = results['XGBoost']['model']

# 创建SHAP解释器
explainer = shap.Explainer(xgb_model, X_train)
# 生成SHAP值
shap_values = explainer(X_test)

# 绘制SHAP摘要图
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test, feature_names=feat_names)
plt.title('SHAP特征重要性摘要')
plt.show()

# 绘制SHAP依赖图（选择最重要的特征）
plt.figure(figsize=(10,6))
shap.dependence_plot(0, shap_values, X_test, feature_names=feat_names)
plt.title('SHAP依赖图（最重要特征）')
plt.show()