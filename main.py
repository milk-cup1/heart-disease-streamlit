import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模块化导入
from modules.data_loader import load_data, split_and_preprocess_data, save_model, load_model, get_numeric_features, get_categorical_features
from modules.model_trainer import train_model, compare_models_detailed, hyperparameter_tuning
from modules.model_evaluator import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance, calculate_sensitivity_specificity, find_optimal_threshold
from modules.visualization import plot_model_comparison, plot_model_radar, plot_correlation_heatmap, plot_target_distribution, plot_feature_distribution
from modules.interpretability import enhanced_shap_analysis
from modules.model_ensemble import ModelEnsemble
from modules.feature_selection import get_feature_importance, select_top_features, analyze_feature_correlation, rfe_feature_selection
from modules.evaluation_strategy import adversarial_validation, plot_learning_curve

# 页面配置
st.set_page_config(page_title="心脏病风险预测系统", layout="wide")

# 数据血缘追踪功能
def show_data_lineage():
    """显示数据血缘信息"""
    with st.sidebar.expander("📊 数据血缘追踪", expanded=False):
        st.markdown("""
        **UCI心脏病数据集处理流程:**
        1. 原始数据: 303条记录,13个特征
        2. 缺失值处理: 使用分组均值填充(避免数据泄露)
        3. 特征工程: 标准化处理
        4. 最终数据: 303条记录(保留全部样本)
        
        **Framingham数据集处理流程:**
        1. 原始数据: 4240条记录,15个特征
        2. 缺失值处理: 按目标变量分组填充
        3. 特征工程: 标准化处理
        4. 最终数据: 保留全部样本
        """)

# 主函数
def main():
    st.title("❤️ 心脏病风险预测与可视化交互式评估系统")
    
    # 侧边栏导航
    option = st.sidebar.radio(
        "导航",
        ["数据探索", "模型训练与对比", "模型预测", "SHAP可解释性分析"]
    )
    
    # 数据集选择
    dataset_name = st.sidebar.selectbox(
        "选择数据集",
        ["UCI心脏病数据集", "Framingham数据集"]
    )
    
    # 显示数据血缘
    show_data_lineage()
    
    # 加载数据
    df = load_data(dataset_name)
    
    if df is None:
        st.error("数据加载失败")
        st.stop()
    
    # 特征与标签分离
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 数据集划分和预处理
    # SMOTE采样比例选择
    smote_ratio = st.sidebar.selectbox(
        "SMOTE采样比例",
        [0.5, 1.0, 2.0, 3.0, 5.0],
        format_func=lambda x: f"{x}:1"
    )
    
    # 重采样方法选择
    resampling_method = st.sidebar.selectbox(
        "重采样方法",
        ["SMOTE", "Borderline-SMOTE", "ADASYN", "SMOTE+Tomek", "SMOTE+ENN"]
    )
    
    X_train_processed, X_test_processed, y_train, y_test, preprocessor = split_and_preprocess_data(
        X, y, dataset_name, use_smote=True, smote_ratio=smote_ratio, resampling_method=resampling_method
    )
    
    # 保存原始特征名称用于可视化
    feature_names = X.columns.tolist()
    
    # 根据选项加载对应模块
    if option == "数据探索":
        st.subheader("数据探索")
        
        # 创建选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["数据概览", "数据质量分析", "相关性分析", "特征分布"])
        
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
            fig = plot_target_distribution(df)
            st.pyplot(fig)
        
        with tab2:
            st.write("### 数据质量分析")
            
            # 缺失值分析
            st.write("#### 缺失值分析")
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            
            if len(missing_values) > 0:
                st.write("**缺失值统计**")
                st.write(missing_values)
            else:
                st.write("**无缺失值**")
            
            # 数据完整性总览
            st.write("#### 数据完整性总览")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("总样本数", df.shape[0])
            col2.metric("特征数量", df.shape[1])
            col3.metric("无心脏病样本", len(df[df['target'] == 0]))
            col4.metric("有心脏病样本", len(df[df['target'] == 1]))
        
        with tab3:
            # 显示特征相关性热力图
            st.write("### 特征相关性热力图")
            # 处理缺失值后再计算相关系数
            df_clean = df.dropna()
            # 对 UCI 数据集，确保 ca 和 thal 列是数值类型
            if dataset_name == "UCI心脏病数据集":
                df_clean['ca'] = pd.to_numeric(df_clean['ca'], errors='coerce')
                df_clean['thal'] = pd.to_numeric(df_clean['thal'], errors='coerce')
                df_clean = df_clean.dropna()
            fig = plot_correlation_heatmap(df_clean)
            st.pyplot(fig)
        
        with tab4:
            st.write("### 特征分布")
            # 选择关键数值特征
            if dataset_name == "UCI心脏病数据集":
                numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            else:
                numeric_features = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
            
            for feature in numeric_features:
                st.write(f"**{feature} 分布**")
                fig = plot_feature_distribution(df, feature)
                st.pyplot(fig)
    
    elif option == "模型训练与对比":
        st.subheader("模型训练与对比")
        
        # 选择模型
        model_choice = st.selectbox(
            "选择模型",
            ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Neural Network"]
        )
        
        # 超参数调优选项
        use_hyperparameter_tuning = st.checkbox("启用超参数调优")
        
        if st.button("开始训练"):
            if use_hyperparameter_tuning:
                # 根据选择的模型类型进行超参数调优
                if model_choice == "Logistic Regression":
                    random_search = hyperparameter_tuning(X_train_processed, y_train, model_type='lr')
                    model = random_search.best_estimator_
                    st.write(f"最佳参数: {random_search.best_params_}")
                    st.write(f"最佳交叉验证AUC: {random_search.best_score_:.4f}")
                elif model_choice == "Random Forest":
                    random_search = hyperparameter_tuning(X_train_processed, y_train, model_type='rf')
                    model = random_search.best_estimator_
                    st.write(f"最佳参数: {random_search.best_params_}")
                    st.write(f"最佳交叉验证AUC: {random_search.best_score_:.4f}")
                else:  # XGBoost
                    random_search = hyperparameter_tuning(X_train_processed, y_train, model_type='xgb')
                    model = random_search.best_estimator_
                    st.write(f"最佳参数: {random_search.best_params_}")
                    st.write(f"最佳交叉验证AUC: {random_search.best_score_:.4f}")
            else:
                model = train_model(model_choice, X_train_processed, y_train)
            
            # 评估模型
            y_pred, y_pred_proba, report, cm, auc_score = evaluate_model(model, X_test_processed, y_test)
            
            # 保存模型
            save_model(model, preprocessor, model_choice, dataset_name)
            st.success(f"模型已保存: {model_choice}")
            
            # 显示评估结果
            st.write("### 模型评估结果")
            st.text(report)
            
            # 显示混淆矩阵
            st.write("### 混淆矩阵")
            fig = plot_confusion_matrix(cm, model_choice)
            st.pyplot(fig)
            
            # 显示ROC曲线
            st.write("### ROC曲线")
            fig = plot_roc_curve(y_test, y_pred_proba, model_choice)
            st.pyplot(fig)
            
            # 阈值调优
            st.write("### 阈值调优")
            optimal_threshold, best_f1, pr_precision, pr_recall, thresholds, f1_scores, detailed_metrics = find_optimal_threshold(y_test, y_pred_proba)
            
            st.write(f"**最优阈值**: {optimal_threshold:.4f}")
            st.write(f"**最佳F1分数**: {best_f1:.4f}")
            
            # 显示详细的阈值分析
            st.write("#### 详细阈值分析 (0.1到0.9，步长0.05)")
            detailed_df = pd.DataFrame(detailed_metrics)
            st.dataframe(detailed_df.style.format({
                'threshold': '{:.2f}',
                'precision': '{:.4f}',
                'recall': '{:.4f}',
                'f1_score': '{:.4f}'
            }).highlight_max(subset=['f1_score'], color='lightgreen'))
            
            # 绘制PR曲线
            st.write("#### PR曲线")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(pr_recall, pr_precision, label=f'{model_choice} PR curve')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            st.pyplot(fig)
            
            # 绘制阈值-F1曲线
            st.write("#### 阈值-F1曲线")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(thresholds, f1_scores, label='F1 Score')
            ax.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.4f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('F1 Score')
            ax.set_title('Threshold vs F1 Score')
            ax.legend()
            st.pyplot(fig)
            
            # 绘制阈值-精确率/召回率曲线
            st.write("#### 阈值-精确率/召回率曲线")
            fig, ax = plt.subplots(figsize=(10, 6))
            detailed_df = pd.DataFrame(detailed_metrics)
            ax.plot(detailed_df['threshold'], detailed_df['precision'], label='Precision')
            ax.plot(detailed_df['threshold'], detailed_df['recall'], label='Recall')
            ax.plot(detailed_df['threshold'], detailed_df['f1_score'], label='F1 Score')
            ax.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.4f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title('Threshold vs Metrics')
            ax.legend()
            st.pyplot(fig)
            
            # 使用最优阈值进行预测
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            st.write("#### 最优阈值下的评估结果")
            from sklearn.metrics import classification_report
            st.text(classification_report(y_test, y_pred_optimal))
            
            # 显示特征重要性
            st.write("### 特征重要性分析")
            fig = plot_feature_importance(model, X.columns.tolist(), model_choice)
            if fig:
                st.pyplot(fig)
            
            # 特征选择
            st.write("### 特征选择")
            feature_selection_method = st.selectbox(
                "选择特征选择方法",
                ["基于树模型", "F检验", "互信息", "RFE递归特征消除"]
            )
            k = st.slider("选择特征数量", min_value=5, max_value=len(X.columns), value=10)
            
            if st.button("运行特征选择"):
                # 获取处理后的特征名称
                processed_feature_names = []
                # 数值特征
                numeric_features = get_numeric_features(dataset_name)
                processed_feature_names.extend(numeric_features)
                # 类别特征（独热编码后）
                categorical_features = get_categorical_features(dataset_name)
                if categorical_features:
                    # 获取独热编码器
                    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                    # 获取类别特征的所有可能值
                    for i, feature in enumerate(categorical_features):
                        categories = onehot_encoder.categories_[i]
                        for cat in categories:
                            processed_feature_names.append(f"{feature}_{cat}")
                
                # 选择特征
                if feature_selection_method == "RFE递归特征消除":
                    # 使用RFE进行特征选择
                    selected_features = rfe_feature_selection(
                        X_train_processed, y_train, processed_feature_names, k=k
                    )
                else:
                    # 映射选择方法
                    method_map = {
                        "基于树模型": "tree",
                        "F检验": "f_classif",
                        "互信息": "mutual_info"
                    }
                    
                    # 选择特征
                    selected_features = select_top_features(
                        X_train_processed, y_train, processed_feature_names, 
                        k=k, method=method_map[feature_selection_method]
                    )
                
                st.write(f"**选中的特征** ({len(selected_features)}个):")
                st.write(selected_features)
                
                # 分析特征相关性
                st.write("### 特征相关性分析")
                selected_features_corr, high_corr_pairs = analyze_feature_correlation(
                    X_train_processed, processed_feature_names, threshold=0.8
                )
                
                st.write(f"**相关性分析后保留的特征** ({len(selected_features_corr)}个):")
                st.write(selected_features_corr)
                
                if high_corr_pairs:
                    st.write("**高度相关的特征对**:")
                    corr_df = pd.DataFrame(high_corr_pairs, columns=["特征1", "特征2", "相关系数"])
                    st.dataframe(corr_df.style.format({"相关系数": "{:.4f}"}))
                else:
                    st.write("**无高度相关的特征对**")
                
                # 使用选择的特征重新训练模型
                st.write("### 使用选择的特征重新训练模型")
                if st.button("使用选择的特征训练"):
                    # 获取选中特征的索引
                    feature_indices = [processed_feature_names.index(f) for f in selected_features]
                    X_train_selected = X_train_processed[:, feature_indices]
                    X_test_selected = X_test_processed[:, feature_indices]
                    
                    # 重新训练模型
                    model_selected = train_model(model_choice, X_train_selected, y_train)
                    
                    # 评估模型
                    y_pred_selected, y_pred_proba_selected, report_selected, cm_selected, auc_score_selected = evaluate_model(model_selected, X_test_selected, y_test)
                    
                    st.write("**使用选择特征后的模型评估结果**")
                    st.text(report_selected)
                    
                    # 显示混淆矩阵
                    st.write("### 混淆矩阵")
                    fig = plot_confusion_matrix(cm_selected, f"{model_choice} (特征选择后)")
                    st.pyplot(fig)
                    
                    # 显示ROC曲线
                    st.write("### ROC曲线")
                    fig = plot_roc_curve(y_test, y_pred_proba_selected, f"{model_choice} (特征选择后)")
                    st.pyplot(fig)
        
        # 模型对比分析
        st.write("---")
        st.subheader("模型性能对比分析")
        
        if st.button("运行模型对比"):
            comparison_df, trained_models = compare_models_detailed(X_train_processed, X_test_processed, y_train, y_test)
            
            # 展示对比表格
            st.write("### 模型性能对比")
            st.dataframe(comparison_df.style.highlight_max(axis=0))
            
            # 可视化对比
            st.write("### 模型性能对比图")
            fig = plot_model_comparison(comparison_df)
            st.pyplot(fig)
            
            # 模型性能雷达图
            st.write("### 模型性能雷达图")
            fig = plot_model_radar(comparison_df)
            st.pyplot(fig)
            
            # 模型性能分析
            st.write("### 模型性能分析")
            best_model = comparison_df.loc[comparison_df['AUC'].idxmax()]
            st.write(f"**最佳模型**: {best_model['Model']}")
            st.write(f"**最高AUC**: {best_model['AUC']:.4f}")
            st.write(f"**交叉验证AUC均值**: {best_model['CV_AUC_Mean']:.4f} (标准差: {best_model['CV_AUC_Std']:.4f})")
            
            # 评估策略
            st.write("---")
            st.subheader("评估策略分析")
            
            # 对抗性验证
            if st.button("运行对抗性验证"):
                st.write("### 对抗性验证结果")
                accuracy, fig = adversarial_validation(X_train_processed, X_test_processed)
                st.pyplot(fig)
                
                if accuracy > 0.8:
                    st.warning(f"⚠️ 训练集和测试集分布差异较大，准确率: {accuracy:.4f}")
                else:
                    st.success(f"✅ 训练集和测试集分布差异较小，准确率: {accuracy:.4f}")
            
            # 学习曲线分析
            if st.button("绘制学习曲线"):
                st.write("### 学习曲线分析")
                # 选择一个模型绘制学习曲线
                model = train_model("XGBoost", X_train_processed, y_train)
                fig = plot_learning_curve(model, X_train_processed, y_train)
                st.pyplot(fig)
            
            # 模型融合评估
            st.write("---")
            st.subheader("模型融合评估")
            
            if st.button("运行模型融合"):
                # 训练所有模型
                lr_model = train_model("Logistic Regression", X_train_processed, y_train)
                rf_model = train_model("Random Forest", X_train_processed, y_train)
                xgb_model = train_model("XGBoost", X_train_processed, y_train)
                
                # 创建模型字典
                models = {
                    "Logistic Regression": lr_model,
                    "Random Forest": rf_model,
                    "XGBoost": xgb_model
                }
                
                # 尝试添加其他模型
                try:
                    lgbm_model = train_model("LightGBM", X_train_processed, y_train)
                    models["LightGBM"] = lgbm_model
                except Exception as e:
                    pass
                
                try:
                    catboost_model = train_model("CatBoost", X_train_processed, y_train)
                    models["CatBoost"] = catboost_model
                except Exception as e:
                    pass
                
                try:
                    nn_model = train_model("Neural Network", X_train_processed, y_train)
                    models["Neural Network"] = nn_model
                except Exception as e:
                    pass
                ensemble = ModelEnsemble(models)
                
                # 软投票融合（等权重）
                soft_voting_metrics = ensemble.evaluate_ensemble(X_test_processed, y_test, method='soft_voting')
                
                # 软投票融合（自定义权重）
                custom_weights = [0.3, 0.3, 0.4]  # 逻辑回归0.3，随机森林0.3，XGBoost 0.4
                soft_voting_custom_metrics = ensemble.evaluate_ensemble(X_test_processed, y_test, method='soft_voting', weights=custom_weights)
                
                # Stacking融合
                stacking_metrics = ensemble.evaluate_ensemble(X_test_processed, y_test, method='stacking', X_train=X_train_processed, y_train=y_train)
                
                # 显示融合结果
                st.write("### 融合模型性能对比")
                
                # 创建结果表格
                ensemble_results = pd.DataFrame([
                    {"方法": "软投票（等权重）", **soft_voting_metrics},
                    {"方法": "软投票（自定义权重）", **soft_voting_custom_metrics},
                    {"方法": "Stacking", **stacking_metrics}
                ])
                
                st.dataframe(ensemble_results.style.highlight_max(axis=0))
                
                # 与最佳单模型对比
                st.write("### 融合模型与最佳单模型对比")
                best_single_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
                
                comparison_data = {
                    "指标": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
                    "最佳单模型": [
                        best_single_model['Accuracy'],
                        best_single_model['Precision'],
                        best_single_model['Recall'],
                        best_single_model['F1-Score'],
                        best_single_model['AUC']
                    ],
                    "软投票（自定义权重）": [
                        soft_voting_custom_metrics['accuracy'],
                        soft_voting_custom_metrics['precision'],
                        soft_voting_custom_metrics['recall'],
                        soft_voting_custom_metrics['f1_score'],
                        soft_voting_custom_metrics['auc']
                    ],
                    "Stacking": [
                        stacking_metrics['accuracy'],
                        stacking_metrics['precision'],
                        stacking_metrics['recall'],
                        stacking_metrics['f1_score'],
                        stacking_metrics['auc']
                    ]
                }
                
                comparison_df2 = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df2)
                
                # 可视化对比
                st.write("### 模型融合性能对比图")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
                labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
                
                x = np.arange(len(labels))
                width = 0.25
                
                ax.bar(x - width, [soft_voting_metrics[m] for m in metrics], width, label='软投票（等权重）')
                ax.bar(x, [soft_voting_custom_metrics[m] for m in metrics], width, label='软投票（自定义权重）')
                ax.bar(x + width, [stacking_metrics[m] for m in metrics], width, label='Stacking')
                
                ax.set_xlabel('评估指标')
                ax.set_ylabel('得分')
                ax.set_title('模型融合性能对比')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                
                st.pyplot(fig)
    
    elif option == "模型预测":
        st.subheader("模型预测")
        
        # 选择模型
        model_choice = st.selectbox(
            "选择模型",
            ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Neural Network"]
        )
        
        # 尝试加载模型，如果不存在则训练
        model_data = load_model(model_choice, dataset_name)
        if model_data:
            model, preprocessor = model_data
            st.success(f"已加载预训练模型: {model_choice}")
        else:
            # 训练模型
            model = train_model(model_choice, X_train_processed, y_train)
            # 保存模型
            save_model(model, preprocessor, model_choice, dataset_name)
            st.success(f"模型已训练并保存: {model_choice}")
        
        # 用户输入特征
        st.write("### 患者信息输入")
        
        # 根据选择的数据集生成不同的输入字段
        if dataset_name == "UCI心脏病数据集":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("年龄", min_value=20, max_value=100, value=50)
                sex = st.selectbox("性别", [0, 1], format_func=lambda x: "女性" if x==0 else "男性")
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
                slope = st.selectbox("ST段斜率", [0, 1, 2], format_func=lambda x: {0: "上斜", 1: "平坦", 2: "下斜"}[x])
                ca = st.number_input("血管数量", min_value=0, max_value=4, value=0)
                thal = st.selectbox("地中海贫血", [0, 1, 2, 3], format_func=lambda x: {0: "正常", 1: "固定缺陷", 2: "可逆缺陷", 3: "未知"}[x])
            
            # 预测
            if st.button("预测"):
                # 准备输入数据
                input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                         columns=feature_names)
                # 使用预处理管道处理输入数据
                input_processed = preprocessor.transform(input_data)
                
                # 预测结果
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0][1]
                
                # 显示结果
                st.write("### 预测结果")
                if prediction == 1:
                    st.error(f"⚠️ 预测: **有心脏病风险**")
                else:
                    st.success(f"✅ 预测: **无心脏病风险**")
                st.write(f"预测概率: {probability:.4f}")
        elif dataset_name == "Framingham数据集":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                male = st.selectbox("性别", [0, 1], format_func=lambda x: "女性" if x==0 else "男性")
                age = st.number_input("年龄", min_value=20, max_value=100, value=50)
                education = st.selectbox("教育水平", [1, 2, 3, 4], format_func=lambda x: f"水平 {x}")
                currentSmoker = st.selectbox("当前吸烟者", [0, 1], format_func=lambda x: "是" if x==1 else "否")
                cigsPerDay = st.number_input("每天吸烟量", min_value=0, max_value=100, value=0)
            
            with col2:
                BPMeds = st.selectbox("血压药物", [0, 1], format_func=lambda x: "是" if x==1 else "否")
                prevalentStroke = st.selectbox("既往中风", [0, 1], format_func=lambda x: "是" if x==1 else "否")
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
                input_data = pd.DataFrame([[male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                                         columns=feature_names)
                # 使用预处理管道处理输入数据
                input_processed = preprocessor.transform(input_data)
                
                # 预测结果
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0][1]
                
                # 显示结果
                st.write("### 预测结果")
                if prediction == 1:
                    st.error(f"⚠️ 预测: **10年内有CHD风险**")
                else:
                    st.success(f"✅ 预测: **10年内无CHD风险**")
                st.write(f"预测概率: {probability:.4f}")
    
    elif option == "SHAP可解释性分析":
        st.subheader("SHAP可解释性分析")
        
        # 训练XGBoost模型 - 处理类别不平衡
        from xgboost import XGBClassifier
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
        xgb_model.fit(X_train_processed, y_train)
        
        # 获取预处理后的特征名称
        processed_feature_names = []
        # 数值特征
        numeric_features = get_numeric_features(dataset_name)
        processed_feature_names.extend(numeric_features)
        # 类别特征（独热编码后）
        categorical_features = get_categorical_features(dataset_name)
        if categorical_features:
            # 获取独热编码器
            onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            # 获取类别特征的所有可能值
            for i, feature in enumerate(categorical_features):
                categories = onehot_encoder.categories_[i]
                for cat in categories:
                    processed_feature_names.append(f"{feature}_{cat}")
        
        # 执行增强的SHAP分析
        enhanced_shap_analysis(xgb_model, X_train_processed, X_test_processed, processed_feature_names)

# 页脚
st.sidebar.markdown("---")
st.sidebar.markdown("© 2026 心脏病风险预测系统")
st.sidebar.markdown("基于UCI心脏病数据集")

if __name__ == "__main__":
    main()