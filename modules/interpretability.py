import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

def enhanced_shap_analysis(model, X_train, X_test, feature_names):
    """增强的SHAP可解释性分析"""
    # 创建解释器
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # 1. 全局特征重要性（条形图）
    st.write("#### 全局特征重要性")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                     plot_type="bar", show=False)
    st.pyplot(fig)
    plt.clf()
    
    # 2. 蜂群图（展示特征影响的正负方向）
    st.write("#### 特征影响分布（蜂群图）")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    st.pyplot(fig)
    plt.clf()
    
    # 3. 单个样本的解释（选择几个典型样本）
    st.write("#### 单个样本解释")
    
    # 选择高风险和低风险样本
    y_proba = model.predict_proba(X_test)[:, 1]
    high_risk_idx = np.argmax(y_proba)
    low_risk_idx = np.argmin(y_proba)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**高风险样本解释**")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap_values[high_risk_idx], show=False)
        st.pyplot(fig)
        plt.clf()
    
    with col2:
        st.write("**低风险样本解释**")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap_values[low_risk_idx], show=False)
        st.pyplot(fig)
        plt.clf()
    
    # 4. 特征依赖图（展示特征交互）
    st.write("#### 特征交互分析")
    top_features = np.argsort(np.abs(shap_values.values).mean(0))[-3:]
    
    for idx in top_features:
        feature_name = feature_names[idx]
        st.write(f"**{feature_name} 的SHAP依赖图**")
        # 创建图形对象
        fig, ax = plt.subplots(figsize=(8, 5))
        # 使用shap的绘图函数，指定ax参数
        shap.dependence_plot(idx, shap_values.values, X_test,
                           feature_names=feature_names, ax=ax, show=False)
        # 显示图形
        st.pyplot(fig)
        plt.clf()