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
                raw_df = pd.read_csv('heart+disease/processed.cleveland.data', names=columns)
            except FileNotFoundError:
                # 如果相对路径失败，尝试使用绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, 'heart+disease', 'processed.cleveland.data')
                raw_df = pd.read_csv(data_path, names=columns)
            
            df = raw_df.copy()
            df = df.replace('?', pd.NA)
            df = df.dropna()
            df['ca'] = pd.to_numeric(df['ca'])
            df['thal'] = pd.to_numeric(df['thal'])
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        elif dataset_name == "Framingham数据集":
            # 尝试相对路径
            try:
                raw_df = pd.read_csv('framingham.csv')
            except FileNotFoundError:
                # 如果相对路径失败，尝试使用绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(current_dir, 'framingham.csv')
                raw_df = pd.read_csv(data_path)
            
            df = raw_df.copy()
            df = df.dropna()
            df = df.rename(columns={'TenYearCHD': 'target'})
        return df, raw_df
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        return None, None

# 数据集选择
st.sidebar.subheader("数据集选择")
dataset_name = st.sidebar.selectbox(
    "选择数据集",
    ["UCI心脏病数据集", "Framingham数据集"]
)

# 加载数据
df, raw_df = load_data(dataset_name)

# 检查数据加载是否成功
if df is None or raw_df is None:
    st.stop()

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
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# 3. 模型训练部分
elif option == "模型训练":
    st.subheader("模型训练")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 选择模型
    model_choice = st.selectbox(
        "选择模型",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    # 训练模型
    @st.cache_data
    def train_model(model_name, X_train, y_train):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        # 训练模型
        model.fit(X_train, y_train)
        return model
    
    if st.button("开始训练"):
        model = train_model(model_choice, X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估指标
        st.write("### 模型评估结果")
        st.text(classification_report(y_test, y_pred))
        
        # 指标解释
        st.write("#### 指标解释")
        st.markdown("- **精确率（Precision）**：预测为正例的样本中，实际为正的比例。高精确率意味着误诊少。")
        st.markdown("- **召回率（Recall）**：实际为正例的样本中，被正确预测的比例。高召回率意味着漏诊少。")
        st.markdown("- **F1-score**：精确率和召回率的调和平均，综合衡量。")
        st.markdown("- **AUC**：ROC曲线下的面积，衡量模型区分正负样本的能力，值越大越好。")
        st.markdown("⚠️ 在心脏病预测场景中，召回率（Recall）尤为重要，因为漏诊的代价很高。")
        
        # 交叉验证（使用标准化后的数据）
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
        st.write(f"5折交叉验证得分: {cv_score:.4f}")
        
        # AUC得分
        auc_score = roc_auc_score(y_test, y_pred_proba)
        st.write(f"AUC得分: {auc_score:.4f}")
        
        # 混淆矩阵
        st.write("### 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_choice} Confusion Matrix')
        st.pyplot(fig)
        
        # ROC曲线
        st.write("### ROC曲线")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
        st.text(classification_report(y_test, y_pred_threshold))
        
        # 更新混淆矩阵
        cm_threshold = confusion_matrix(y_test, y_pred_threshold)
        fig, ax = plt.subplots()
        sns.heatmap(cm_threshold, annot=True, fmt='d', ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_choice} 混淆矩阵 (阈值={threshold})')
        st.pyplot(fig)
        
        # 计算敏感度和特异度
        from sklearn.metrics import recall_score, precision_score
        sensitivity = recall_score(y_test, y_pred_threshold)
        specificity = recall_score(y_test, y_pred_threshold, pos_label=0)
        st.write(f"敏感度（召回率）: {sensitivity:.4f}")
        st.write(f"特异度: {specificity:.4f}")
        
        # 阈值-敏感度/特异度曲线
        st.write("#### 阈值-敏感度/特异度曲线")
        thresholds = np.linspace(0, 1, 100)
        sensitivities = []
        specificities = []
        
        for t in thresholds:
            y_pred_t = (y_pred_proba >= t).astype(int)
            sensitivities.append(recall_score(y_test, y_pred_t))
            specificities.append(recall_score(y_test, y_pred_t, pos_label=0))
        
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
        st.write("### 业务场景总结")
        st.markdown(f"> 根据当前模型,在默认阈值0.5下，模型能够正确识别出{sensitivity:.2%}的实际心脏病患者（召回率），但有{1-specificity:.2%}的健康人被误诊为患者。")
        st.markdown(f"> 如果我们将阈值降低到0.3,可以识别出更多的患者（召回率提升），但误诊率也会上升。")
        st.markdown(f"> 在实际应用中，需要根据医疗资源和患者承受能力选择合适的阈值。")

# 4. 模型预测部分
elif option == "模型预测":
    st.subheader("模型预测")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 选择模型
    model_choice = st.selectbox(
        "选择模型",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )
    
    # 训练模型
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_choice == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_scaled, y)
    
    # 用户输入特征
    st.write("### 输入患者信息")
    
    # 根据选择的数据集生成不同的输入字段
    if dataset_name == "UCI心脏病数据集":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("年龄", min_value=20, max_value=100, value=50)
            sex = st.selectbox("性别", [0, 1], format_func=lambda x: "女" if x==0 else "男")
            cp = st.selectbox("胸痛类型", [0, 1, 2, 3], format_func=lambda x: {0: "典型心绞痛", 1: "非典型心绞痛", 2: "非心绞痛", 3: "无症状"}[x])
            trestbps = st.number_input("静息血压", min_value=80, max_value=200, value=120)
            chol = st.number_input("血清胆固醇", min_value=100, max_value=400, value=200)
        
        with col2:
            fbs = st.selectbox("空腹血糖 > 120mg/dl", [0, 1], format_func=lambda x: "否" if x==0 else "是")
            restecg = st.selectbox("静息心电图结果", [0, 1, 2], format_func=lambda x: {0: "正常", 1: "ST-T异常", 2: "左心室肥厚"}[x])
            thalach = st.number_input("最大心率", min_value=60, max_value=220, value=150)
            exang = st.selectbox("运动诱发心绞痛", [0, 1], format_func=lambda x: "否" if x==0 else "是")
            oldpeak = st.number_input("ST段压低", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        
        with col3:
            slope = st.selectbox("ST段斜率", [0, 1, 2], format_func=lambda x: {0: "上升", 1: "平坦", 2: "下降"}[x])
            ca = st.number_input("血管造影数", min_value=0, max_value=4, value=0)
            thal = st.selectbox("地中海贫血", [0, 1, 2, 3], format_func=lambda x: {0: "正常", 1: "固定缺陷", 2: "可逆缺陷", 3: "未知"}[x])
        
        # 预测
        if st.button("预测"):
            # 准备输入数据
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            input_scaled = scaler.transform(input_data)
            
            # 预测结果
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # 显示结果
            st.write("### 预测结果")
            if prediction == 1:
                st.error(f"⚠️ 预测结果：**有心脏病风险**")
            else:
                st.success(f"✅ 预测结果：**无心脏病风险**")
            st.write(f"预测概率：{probability:.4f}")
    elif dataset_name == "Framingham数据集":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            male = st.selectbox("性别", [0, 1], format_func=lambda x: "女" if x==0 else "男")
            age = st.number_input("年龄", min_value=20, max_value=100, value=50)
            education = st.selectbox("教育程度", [1, 2, 3, 4], format_func=lambda x: f"{x}级")
            currentSmoker = st.selectbox("当前吸烟者", [0, 1], format_func=lambda x: "是" if x==1 else "否")
            cigsPerDay = st.number_input("每天吸烟数", min_value=0, max_value=100, value=0)
        
        with col2:
            BPMeds = st.selectbox("服用降压药", [0, 1], format_func=lambda x: "是" if x==1 else "否")
            prevalentStroke = st.selectbox("既往卒中", [0, 1], format_func=lambda x: "是" if x==1 else "否")
            prevalentHyp = st.selectbox("高血压", [0, 1], format_func=lambda x: "是" if x==1 else "否")
            diabetes = st.selectbox("糖尿病", [0, 1], format_func=lambda x: "是" if x==1 else "否")
            totChol = st.number_input("总胆固醇", min_value=100, max_value=500, value=200)
        
        with col3:
            sysBP = st.number_input("收缩压", min_value=80, max_value=250, value=120)
            diaBP = st.number_input("舒张压", min_value=50, max_value=150, value=80)
            BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            heartRate = st.number_input("心率", min_value=40, max_value=200, value=75)
            glucose = st.number_input("血糖", min_value=40, max_value=300, value=80)
        
        # 预测
        if st.button("预测"):
            # 准备输入数据
            input_data = np.array([[male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
            input_scaled = scaler.transform(input_data)
            
            # 预测结果
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # 显示结果
            st.write("### 预测结果")
            if prediction == 1:
                st.error(f"⚠️ 预测结果：**10年内有冠心病风险**")
            else:
                st.success(f"✅ 预测结果：**10年内无冠心病风险**")
            st.write(f"预测概率：{probability:.4f}")

# 5. SHAP可解释性分析部分
elif option == "SHAP可解释性分析":
    st.subheader("SHAP可解释性分析")
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练XGBoost模型
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    
    # 创建SHAP解释器
    explainer = shap.Explainer(xgb_model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    
    # 绘制SHAP摘要图
    st.write("### SHAP特征重要性摘要")
    # 清除之前的图形
    plt.clf()
    # 直接调用shap.summary_plot
    shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
    # 显示图形
    st.pyplot()
    
    # 选择特征查看依赖图
    st.write("### SHAP特征依赖图")
    feature = st.selectbox("选择特征", X.columns)
    
    # 清除之前的图形
    plt.clf()
    # 从Explanation对象中提取SHAP值
    shap_values_array = shap_values.values
    # 调用shap.dependence_plot，使用特征名称
    shap.dependence_plot(feature, shap_values_array, X_test_scaled, feature_names=X.columns, show=False)
    # 显示图形
    st.pyplot()

# 数据血缘
st.sidebar.markdown("---")
st.sidebar.subheader("数据处理流程")

if dataset_name == "UCI心脏病数据集":
    st.sidebar.write("1. 原始数据：从 processed.cleveland.data 读取，包含 303 条记录,13 个特征 + 1 个目标变量。")
    st.sidebar.write("2. 缺失值标识：将文件中的 ? 替换为 NaN，识别出缺失值。")
    st.sidebar.write("3. 缺失值处理：删除含有 NaN 的行，剩余 297 条记录。")
    st.sidebar.write("4. 数据类型转换：将 ca 和 thal 列转换为数值类型（原为字符串）。")
    st.sidebar.write("5. 目标变量二值化:将目标变量转换为二分类:0 表示无心脏病,1 表示有心脏病。")
    st.sidebar.write("6. 最终建模数据:297 条记录,13 个特征，目标为二分类。")
elif dataset_name == "Framingham数据集":
    st.sidebar.write("1. 原始数据：从 framingham.csv 读取，包含 4240 条记录,15 个特征 + 1 个目标变量。")
    st.sidebar.write("2. 缺失值处理：删除含有 NaN 的行，剩余约 3658 条记录。")
    st.sidebar.write("3. 目标变量重命名：将 TenYearCHD 重命名为 target 以保持一致性。")
    st.sidebar.write("4. 最终建模数据:约 3658 条记录,15 个特征，目标为二分类。")

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 心脏病风险预测系统")
st.sidebar.markdown("基于UCI心脏病数据集")
