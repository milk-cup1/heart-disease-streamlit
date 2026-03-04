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
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_name == "UCI心脏病数据集":
        # 加载UCI心脏病数据集
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        data_path = os.path.join(current_dir, 'heart+disease', 'processed.cleveland.data')
        df = pd.read_csv(data_path, names=columns)
        # 处理缺失值
        df = df.replace('?', pd.NA)
        df = df.dropna()
        # 将ca和thal列转换为数值类型
        df['ca'] = pd.to_numeric(df['ca'])
        df['thal'] = pd.to_numeric(df['thal'])
        # 转换目标变量为二分类
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    elif dataset_name == "Framingham数据集":
        # 加载Framingham数据集
        data_path = os.path.join(current_dir, 'framingham.csv')
        df = pd.read_csv(data_path)
        # 处理缺失值
        df = df.dropna()
        # 重命名目标变量以保持一致性
        df = df.rename(columns={'TenYearCHD': 'target'})
    return df

# 数据集选择
st.sidebar.subheader("数据集选择")
dataset_name = st.sidebar.selectbox(
    "选择数据集",
    ["UCI心脏病数据集", "Framingham数据集"]
)

# 加载数据
df = load_data(dataset_name)

# 2. 数据探索部分
if option == "数据探索":
    st.subheader("数据探索")
    
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
    if st.button("开始训练"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        elif model_choice == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 评估指标
        st.write("### 模型评估结果")
        st.text(classification_report(y_test, y_pred))
        
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
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title(f'{model_choice} 混淆矩阵')
        st.pyplot(fig)
        
        # ROC曲线
        st.write("### ROC曲线")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{model_choice} (AUC={auc_score:.2f})')
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC曲线')
        ax.legend()
        st.pyplot(fig)

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

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 心脏病风险预测系统")
st.sidebar.markdown("基于UCI心脏病数据集")
