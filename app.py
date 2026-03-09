import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 设置页面配置
st.set_page_config(
    page_title="心脏病风险预测",
    page_icon="❤️",
    layout="wide"
)

# 页面标题
st.title("❤️ 心脏病风险预测系统")

# 侧边栏
st.sidebar.title("功能选择")
option = st.sidebar.selectbox(
    "选择功能",
    ["数据探索", "模型训练", "模型预测", "SHAP可解释性分析"]
)

# 1. 数据加载和预处理
@st.cache_data
def load_data(dataset_name):
    try:
        if dataset_name == "UCI心脏病数据集":
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            # 尝试相对路径
            try:
                df = pd.read_csv('heart+disease/processed.cleveland.data', names=columns)
            except FileNotFoundError:
                # 如果相对路径失败，尝试使用绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, 'heart+disease', 'processed.cleveland.data')
                df = pd.read_csv(data_path, names=columns)
            
            # 基本处理：替换缺失值标记
            df = df.replace('?', pd.NA)
            # 将目标变量转换为二分类
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        elif dataset_name == "Framingham数据集":
            # 尝试相对路径
            try:
                df = pd.read_csv('framingham.csv')
            except FileNotFoundError:
                # 如果相对路径失败，尝试使用绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, 'framingham.csv')
                df = pd.read_csv(data_path)
            
            # 重命名目标变量
            df = df.rename(columns={'TenYearCHD': 'target'})
        return df
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        return None

# 数据预处理函数，避免数据泄露
def preprocess_data(X_train, X_test, y_train, dataset_name):
    # 处理 UCI 心脏病数据集
    if dataset_name == "UCI心脏病数据集":
        # 训练集处理：删除缺失值
        train_data = pd.concat([X_train, y_train], axis=1)
        train_data = train_data.dropna()
        X_train_clean = train_data.drop('target', axis=1)
        y_train_clean = train_data['target']
        
        # 将 ca 和 thal 列转换为数值类型
        X_train_clean['ca'] = pd.to_numeric(X_train_clean['ca'])
        X_train_clean['thal'] = pd.to_numeric(X_train_clean['thal'])
        
        # 测试集处理：删除缺失值
        X_test_clean = X_test.dropna()
        
        # 将 ca 和 thal 列转换为数值类型
        X_test_clean['ca'] = pd.to_numeric(X_test_clean['ca'])
        X_test_clean['thal'] = pd.to_numeric(X_test_clean['thal'])
        
        return X_train_clean, X_test_clean, y_train_clean
    
    # 处理 Framingham 数据集
    elif dataset_name == "Framingham数据集":
        # 训练集处理：按 target 分组填充
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # 按 target 分组进行填充
        for target_group in train_data['target'].unique():
            # 获取当前组的数据
            group_data = train_data[train_data['target'] == target_group]
            
            # 对数值型特征使用均值填充
            numeric_cols = group_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if train_data[col].isnull().sum() > 0:
                    fill_value = group_data[col].mean()
                    train_data.loc[train_data['target'] == target_group, col] = train_data.loc[train_data['target'] == target_group, col].fillna(fill_value)
            
            # 对分类型特征使用众数填充
            categorical_cols = group_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if train_data[col].isnull().sum() > 0:
                    fill_value = group_data[col].mode()[0]
                    train_data.loc[train_data['target'] == target_group, col] = train_data.loc[train_data['target'] == target_group, col].fillna(fill_value)
        
        # 检查是否还有缺失值
        if train_data.isnull().sum().sum() > 0:
            # 如果还有缺失值，使用全局统计量填充
            for col in train_data.columns:
                if train_data[col].isnull().sum() > 0:
                    if train_data[col].dtype in ['float64', 'int64']:
                        train_data[col] = train_data[col].fillna(train_data[col].mean())
                    else:
                        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
        
        X_train_clean = train_data.drop('target', axis=1)
        y_train_clean = train_data['target']
        
        # 测试集处理：使用训练集的全局统计量填充
        test_data = X_test.copy()
        
        # 计算训练集的全局统计量
        train_stats = {}
        numeric_cols = X_train_clean.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X_train_clean.select_dtypes(include=['object']).columns
        
        # 对数值型特征使用训练集的中位数（对离群值更鲁棒）
        for col in numeric_cols:
            train_stats[col] = X_train_clean[col].median()
        
        # 对分类型特征使用训练集的众数
        for col in categorical_cols:
            train_stats[col] = X_train_clean[col].mode()[0]
        
        # 使用训练集的全局统计量填充测试集
        for col in test_data.columns:
            if test_data[col].isnull().sum() > 0:
                test_data[col] = test_data[col].fillna(train_stats[col])
        
        return X_train_clean, test_data, y_train_clean

# 数据集选择
st.sidebar.subheader("数据集选择")
dataset_name = st.sidebar.selectbox(
    "选择数据集",
    ["UCI心脏病数据集", "Framingham数据集"]
)

# 加载数据
df = load_data(dataset_name)

# 检查数据加载是否成功
if df is None:
    st.stop()

# 保存原始数据用于展示
raw_df = df.copy()

# 2. 数据探索部分
if option == "数据探索":
    st.subheader("数据探索")
    
    # 创建选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["数据概览", "数据清洗过程", "数据质量分析", "相关性分析"])
    
    with tab1:
        # 显示数据基本信息
        st.write("### 数据集基本信息")
        st.write(df.info())
        
        # 显示前几行数据
        st.write("### 数据预览")
        st.write(df.head())
        
        # 显示统计描述
        st.write("### 数据统计描述")
        st.write(df.describe())
        
        # 显示目标变量分布
        st.write("### 目标变量分布")
        target_counts = df['target'].value_counts()
        st.bar_chart(target_counts)
        st.write(f"无心脏病: {target_counts[0]}人")
        st.write(f"有心脏病: {target_counts[1]}人")
    
    with tab2:
        st.write("### 数据清洗过程")
        
        # 数据形状对比
        st.write("#### 数据形状对比")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**原始数据**")
            st.write(f"行数: {raw_df.shape[0]}")
            st.write(f"列数: {raw_df.shape[1]}")
        with col2:
            st.write("**清洗后数据**")
            st.write(f"行数: {df.shape[0]}")
            st.write(f"列数: {df.shape[1]}")
        
        deleted_rows = raw_df.shape[0] - df.shape[0]
        deleted_ratio = (deleted_rows / raw_df.shape[0]) * 100
        st.write(f"删除的样本数: {deleted_rows} ({deleted_ratio:.2f}%)")
        
        # 缺失值对比
        st.write("#### 缺失值对比")
        
        # 原始数据缺失值
        raw_missing = raw_df.isnull().sum()
        raw_missing = raw_missing[raw_missing > 0]
        
        if len(raw_missing) > 0:
            st.write("**原始数据缺失值**")
            # 绘制缺失值条形图
            fig, ax = plt.subplots(figsize=(10, 6))
            raw_missing.plot(kind='bar', ax=ax)
            plt.title('Raw Data Missing Values Distribution')
            plt.ylabel('Missing Values Count')
            st.pyplot(fig)
            
            # 显示缺失值表格
            missing_df = pd.DataFrame({
                '缺失数量': raw_missing,
                '缺失比例': (raw_missing / raw_df.shape[0] * 100).round(2)
            })
            st.write(missing_df)
        else:
            st.write("**原始数据无缺失值**")
        
        # 清洗后数据缺失值
        clean_missing = df.isnull().sum().sum()
        if clean_missing == 0:
            st.write("**清洗后数据**")
            st.write("已删除所有含缺失值的行")
        
        # 数值特征分布对比
        st.write("#### 数值特征分布对比")
        
        # 选择关键数值特征
        if dataset_name == "UCI心脏病数据集":
            numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        else:
            numeric_features = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        for feature in numeric_features:
            st.write(f"**{feature} 分布对比**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("原始数据")
                # 排除缺失值后绘制
                if feature in raw_df.columns:
                    raw_data = raw_df[feature].dropna()
                    fig, ax = plt.subplots()
                    sns.histplot(raw_data, kde=True, ax=ax)
                    plt.title(f'Raw Data {feature} Distribution')
                    st.pyplot(fig)
            
            with col2:
                st.write("清洗后数据")
                if feature in df.columns:
                    fig, ax = plt.subplots()
                    sns.histplot(df[feature], kde=True, ax=ax)
                    plt.title(f'Cleaned Data {feature} Distribution')
                    st.pyplot(fig)
        
        # 类别特征频数对比
        st.write("#### 类别特征频数对比")
        
        if dataset_name == "UCI心脏病数据集":
            cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
        else:
            cat_features = ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']
        
        for feature in cat_features:
            if feature in raw_df.columns and feature in df.columns:
                st.write(f"**{feature} 分布对比**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("原始数据")
                    raw_counts = raw_df[feature].value_counts()
                    fig, ax = plt.subplots()
                    raw_counts.plot(kind='bar', ax=ax)
                    plt.title(f'Raw Data {feature} Distribution')
                    st.pyplot(fig)
                
                with col2:
                    st.write("清洗后数据")
                    clean_counts = df[feature].value_counts()
                    fig, ax = plt.subplots()
                    clean_counts.plot(kind='bar', ax=ax)
                    plt.title(f'Cleaned Data {feature} Distribution')
                    st.pyplot(fig)
    
    with tab3:
        st.write("### 数据质量分析")
        
        # 缺失值分析
        st.write("#### 缺失值分析")
        
        # 缺失值热力图
        if raw_df.isnull().sum().sum() > 0:
            st.write("**缺失值模式热力图**")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(raw_df.isnull(), cbar=False, cmap='viridis', ax=ax)
            plt.title('Missing Values Pattern Heatmap')
            st.pyplot(fig)
        
        # 异常值分析
        st.write("#### 异常值分析")
        
        for feature in numeric_features:
            if feature in df.columns:
                st.write(f"**{feature} 异常值分析**")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[feature], ax=ax)
                plt.title(f'{feature} Box Plot')
                st.pyplot(fig)
                
                # 计算IQR和异常值
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                st.write(f"异常值数量: {len(outliers)}")
                st.write(f"异常值范围: < {lower_bound:.2f} 或 > {upper_bound:.2f}")
        
        # 类别不平衡分析
        st.write("#### 类别不平衡分析")
        target_counts = df['target'].value_counts()
        total = len(df)
        class_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else float('inf')
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**目标变量分布**")
            fig, ax = plt.subplots()
            target_counts.plot(kind='bar', ax=ax)
            plt.title('Target Variable Distribution')
            st.pyplot(fig)
        
        with col2:
            st.write("**目标变量比例**")
            fig, ax = plt.subplots()
            target_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            plt.title('Target Variable Proportion')
            st.pyplot(fig)
        
        st.write(f"类别比例 (0:1): {class_ratio:.2f}:1")
        if class_ratio > 2:
            st.write("⚠️ 注意：数据存在类别不平衡问题，可能需要考虑过采样或欠采样方法")
        
        # 数据完整性总览
        st.write("#### 数据完整性总览")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总样本数", raw_df.shape[0])
        col2.metric("完整样本数", df.shape[0])
        col3.metric("删除样本数", raw_df.shape[0] - df.shape[0])
        col4.metric("特征数量", df.shape[1])
    
    with tab4:
        # 显示特征相关性热力图
        st.write("### 特征相关性热力图")
        # 处理缺失值后再计算相关系数
        df_clean = df.dropna()
        # 对 UCI 数据集，确保 ca 和 thal 列是数值类型
        if dataset_name == "UCI心脏病数据集":
            df_clean['ca'] = pd.to_numeric(df_clean['ca'], errors='coerce')
            df_clean['thal'] = pd.to_numeric(df_clean['thal'], errors='coerce')
            df_clean = df_clean.dropna()
        corr = df_clean.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# 3. 模型训练部分
elif option == "模型训练":
    st.subheader("模型训练")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分 - 先划分再预处理，避免数据泄露
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # 数据预处理 - 使用训练集的统计量处理测试集
    X_train_clean, X_test_clean, y_train_clean = preprocess_data(X_train, X_test, y_train, dataset_name)
    
    # 特征标准化 - 只在训练集上拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # 更新 y_test 以匹配处理后的 X_test
    y_test_clean = y_test[X_test_clean.index]
    
    # 选择模型
    model_choice = st.selectbox(
        "选择模型",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    # 训练模型
    @st.cache_data
    def train_model(model_name, X_train, y_train):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200, class_weight='balanced')  # 处理类别不平衡
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced')  # 处理类别不平衡
        elif model_name == "XGBoost":
            scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)  # 处理类别不平衡
        
        # 训练模型
        model.fit(X_train, y_train)
        return model
    
    if st.button("开始训练"):
        model = train_model(model_choice, X_train_scaled, y_train_clean)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估指标
        st.write("### 模型评估结果")
        st.text(classification_report(y_test_clean, y_pred))
        
        # 指标解释
        st.write("#### 指标解释")
        st.markdown("- **精确率（Precision）**：预测为正例的样本中，实际为正的比例。高精确率意味着误诊少。")
        st.markdown("- **召回率（Recall）**：实际为正例的样本中，被正确预测的比例。高召回率意味着漏诊少。")
        st.markdown("- **F1-score**：精确率和召回率的调和平均，综合衡量。")
        st.markdown("- **AUC**：ROC曲线下的面积，衡量模型区分正负样本的能力，值越大越好。")
        st.markdown("⚠️ 在心脏病预测场景中，召回率（Recall）尤为重要，因为漏诊的代价很高。")
        
        # 交叉验证（使用标准化后的数据）
        cv_score = cross_val_score(model, X_train_scaled, y_train_clean, cv=5).mean()
        st.write(f"5折交叉验证得分: {cv_score:.4f}")
        
        # AUC得分
        auc_score = roc_auc_score(y_test_clean, y_pred_proba)
        st.write(f"AUC得分: {auc_score:.4f}")
        
        # 混淆矩阵
        st.write("### 混淆矩阵")
        cm = confusion_matrix(y_test_clean, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_choice} Confusion Matrix')
        st.pyplot(fig)
        
        # ROC曲线
        st.write("### ROC曲线")
        fpr, tpr, _ = roc_curve(y_test_clean, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{model_choice} (AUC={auc_score:.2f})')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
        
        # 阈值选择分析
        st.write("### 阈值选择分析")
        threshold = st.slider("选择分类阈值", 0.0, 1.0, 0.5, 0.05)
        
        # 根据阈值更新预测
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        # 更新评估指标
        st.write("#### 阈值调整结果")
        st.text(classification_report(y_test_clean, y_pred_threshold))
        
        # 更新混淆矩阵
        cm_threshold = confusion_matrix(y_test_clean, y_pred_threshold)
        fig, ax = plt.subplots()
        sns.heatmap(cm_threshold, annot=True, fmt='d', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_choice} Confusion Matrix (Threshold={threshold})')
        st.pyplot(fig)
        
        # 计算敏感度和特异度
        from sklearn.metrics import recall_score, precision_score
        sensitivity = recall_score(y_test_clean, y_pred_threshold)
        specificity = recall_score(y_test_clean, y_pred_threshold, pos_label=0)
        st.write(f"敏感度（召回率）: {sensitivity:.4f}")
        st.write(f"特异度: {specificity:.4f}")
        
        # 阈值-敏感度/特异度曲线
        st.write("#### 阈值-敏感度/特异度曲线")
        thresholds = np.linspace(0, 1, 100)
        sensitivities = []
        specificities = []
        
        for t in thresholds:
            y_pred_t = (y_pred_proba >= t).astype(int)
            sensitivities.append(recall_score(y_test_clean, y_pred_t))
            specificities.append(recall_score(y_test_clean, y_pred_t, pos_label=0))
        
        fig, ax = plt.subplots()
        ax.plot(thresholds, sensitivities, label='Sensitivity (Recall)')
        ax.plot(thresholds, specificities, label='Specificity')
        ax.axvline(x=threshold, color='r', linestyle='--', label=f'Current Threshold: {threshold}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Effect of Threshold on Sensitivity and Specificity')
        ax.legend()
        st.pyplot(fig)
        
        # 特征重要性解释
        st.write("### 特征重要性分析")
        if model_choice in ["Random Forest", "XGBoost"]:
            importances = model.feature_importances_
            feat_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
            feat_importances = feat_importances.sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feat_importances, ax=ax)
            plt.title(f'{model_choice} Feature Importance')
            st.pyplot(fig)
            
            # 显示前5个重要特征
            top_features = feat_importances.head(5)
            st.write("#### 重要特征分析")
            for i, row in top_features.iterrows():
                st.write(f"- **{row['feature']}**: 重要性 = {row['importance']:.4f}")
        
        # 业务场景总结
        st.write("### Business Scenario Summary")
        st.markdown(f"> Based on the current model, at the default threshold of 0.5, the model can correctly identify {sensitivity:.2%} of actual heart disease patients (recall), but {1-specificity:.2%} of healthy people are misdiagnosed as patients.")
        st.markdown(f"> If we lower the threshold to 0.3, we can identify more patients (recall improvement), but the misdiagnosis rate will also increase.")
        st.markdown(f"> In practical applications, the appropriate threshold needs to be selected based on medical resources and patient tolerance.")

# 4. 模型预测部分
elif option == "模型预测":
    st.subheader("Model Prediction")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分 - 先划分再预处理，避免数据泄露
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # 数据预处理 - 使用训练集的统计量处理测试集
    X_train_clean, X_test_clean, y_train_clean = preprocess_data(X_train, X_test, y_train, dataset_name)
    
    # 特征标准化 - 只在训练集上拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    
    # 选择模型
    model_choice = st.selectbox(
        "选择模型",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    # 训练模型 - 只在训练集上训练
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200, class_weight='balanced')  # 处理类别不平衡
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced')  # 处理类别不平衡
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=len(y_train_clean[y_train_clean==0])/len(y_train_clean[y_train_clean==1]))  # 处理类别不平衡
    
    model.fit(X_train_scaled, y_train_clean)
    
    # 用户输入特征
    st.write("### Patient Information Input")
    
    # 根据选择的数据集生成不同的输入字段
    if dataset_name == "UCI心脏病数据集":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
            trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
            chol = st.number_input("Serum Cholesterol", min_value=100, max_value=400, value=200)
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            restecg = st.selectbox("Resting ECG Result", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x])
            thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        
        with col3:
            slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
            ca = st.number_input("Number of Vessels", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x])
        
        # 预测
        if st.button("Predict"):
            # 准备输入数据
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            input_scaled = scaler.transform(input_data)
            
            # 预测结果
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # 显示结果
            st.write("### Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ Prediction: **At risk of heart disease**")
            else:
                st.success(f"✅ Prediction: **No risk of heart disease**")
            st.write(f"Prediction Probability: {probability:.4f}")
    elif dataset_name == "Framingham数据集":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            male = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            age = st.number_input("Age", min_value=20, max_value=100, value=50)
            education = st.selectbox("Education Level", [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
            currentSmoker = st.selectbox("Current Smoker", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)
        
        with col2:
            BPMeds = st.selectbox("Blood Pressure Medication", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            prevalentStroke = st.selectbox("Previous Stroke", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            prevalentHyp = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200)
        
        with col3:
            sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
            diaBP = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=150, value=80)
            BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
            glucose = st.number_input("Glucose", min_value=40, max_value=300, value=80)
        
        # 预测
        if st.button("Predict"):
            # 准备输入数据
            input_data = np.array([[male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
            input_scaled = scaler.transform(input_data)
            
            # 预测结果
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # 显示结果
            st.write("### Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ Prediction: **At risk of CHD within 10 years**")
            else:
                st.success(f"✅ Prediction: **No risk of CHD within 10 years**")
            st.write(f"Prediction Probability: {probability:.4f}")

# 5. SHAP Interpretability Analysis
elif option == "SHAP可解释性分析":
    st.subheader("SHAP Interpretability Analysis")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分 - 先划分再预处理，避免数据泄露
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # 数据预处理 - 使用训练集的统计量处理测试集
    X_train_clean, X_test_clean, y_train_clean = preprocess_data(X_train, X_test, y_train, dataset_name)
    
    # 特征标准化 - 只在训练集上拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # 训练XGBoost模型 - 处理类别不平衡
    scale_pos_weight = len(y_train_clean[y_train_clean==0])/len(y_train_clean[y_train_clean==1]) if len(y_train_clean[y_train_clean==1]) > 0 else 1
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train_scaled, y_train_clean)
    
    # 创建SHAP解释器
    explainer = shap.Explainer(xgb_model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    
    # 绘制SHAP摘要图
    st.write("### SHAP Feature Importance Summary")
    # 清除之前的图形
    plt.clf()
    # 创建图形对象
    fig, ax = plt.subplots(figsize=(10, 6))
    # 直接调用shap.summary_plot
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    # 显示图形
    st.pyplot(fig)
    
    # 选择特征查看依赖图
    with st.expander("SHAP Feature Dependence Plot", expanded=False):
        # 为特征列创建英文名称映射
        if dataset_name == "UCI心脏病数据集":
            # 确保使用英文特征名称
            feature_display_names = [
                'Age',
                'Sex',
                'Chest Pain Type',
                'Resting Blood Pressure',
                'Serum Cholesterol',
                'Fasting Blood Sugar',
                'Resting ECG',
                'Maximum Heart Rate',
                'Exercise Induced Angina',
                'ST Depression',
                'ST Slope',
                'Number of Vessels',
                'Thalassemia'
            ]
            # 原始列名
            original_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        elif dataset_name == "Framingham数据集":
            # 确保使用英文特征名称
            feature_display_names = [
                'Sex',
                'Age',
                'Education Level',
                'Current Smoker',
                'Cigarettes Per Day',
                'Blood Pressure Medication',
                'Previous Stroke',
                'Hypertension',
                'Diabetes',
                'Total Cholesterol',
                'Systolic Blood Pressure',
                'Diastolic Blood Pressure',
                'BMI',
                'Heart Rate',
                'Glucose'
            ]
            # 原始列名
            original_columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        
        # 创建显示名称到原始列名的映射
        name_to_col = dict(zip(feature_display_names, original_columns))
        # 使用显示名称创建选择框
        selected_display_name = st.selectbox("Select Feature", feature_display_names)
        # 获取对应的原始列名
        feature = name_to_col[selected_display_name]
        
        # 清除之前的图形
        plt.clf()
        # 尝试使用更简单的方法来显示SHAP值
        st.write(f"### SHAP Values for {selected_display_name}")
        # 从Explanation对象中提取SHAP值
        shap_values_array = shap_values.values
        # 获取特征索引
        feature_idx = X.columns.get_loc(feature)
        # 显示前10个样本的SHAP值
        shap_df = pd.DataFrame({
            'Feature Value': X_test_scaled[:, feature_idx],
            'SHAP Value': shap_values_array[:, feature_idx]
        })
        st.write(shap_df.head(10))
        
        # 创建一个简单的散点图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(shap_df['Feature Value'], shap_df['SHAP Value'])
        ax.set_xlabel(selected_display_name)
        ax.set_ylabel('SHAP Value')
        ax.set_title(f'SHAP Dependence Plot for {selected_display_name}')
        st.pyplot(fig)

# 数据血缘
st.sidebar.markdown("---")
with st.sidebar.expander("数据处理流程", expanded=False):
    if dataset_name == "UCI心脏病数据集":
        st.write("1. 原始数据：从 processed.cleveland.data 读取，包含 303 条记录,13 个特征 + 1 个目标变量")
        st.write("2. 缺失值标识：将文件中的 ? 替换为 NaN，识别出缺失值")
        st.write("3. 缺失值处理：删除含有 NaN 的行，剩余 297 条记录")
        st.write("4. 数据类型转换：将 ca 和 thal 列转换为数值类型（原为字符串）")
        st.write("5. 目标变量二值化:将目标变量转换为二分类:0 表示无心脏病,1 表示有心脏病")
        st.write("6. 最终建模数据:297 条记录,13 个特征，目标为二分类")
    elif dataset_name == "Framingham数据集":
        st.write("1. 原始数据：从 framingham.csv 读取，包含 4240 条记录,15 个特征 + 1 个目标变量")
        st.write("2. 缺失值处理：删除含有 NaN 的行，剩余约 3658 条记录")
        st.write("3. 目标变量重命名：将 TenYearCHD 重命名为 target 以保持一致性")
        st.write("4. 最终建模数据:约 3658 条记录,15 个特征，目标为二分类")

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 心脏病风险预测系统")
st.sidebar.markdown("基于UCI心脏病数据集")
