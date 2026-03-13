import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

def add_medical_interaction_features(df, dataset_name):
    """构造医学相关的交互特征"""
    if dataset_name == "UCI心脏病数据集":
        # 年龄分段
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5])
        
        # 血压分级
        df['bp_group'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 160, 200], labels=[1, 2, 3, 4])
        
        # 胆固醇分级
        df['chol_group'] = pd.cut(df['chol'], bins=[0, 200, 240, 280, 400], labels=[1, 2, 3, 4])
        
        # 交互特征
        df['age_bp'] = df['age'] * df['trestbps']
        df['age_chol'] = df['age'] * df['chol']
        df['age_thalach'] = df['age'] * df['thalach']
        df['bp_chol'] = df['trestbps'] * df['chol']
        df['bp_thalach'] = df['trestbps'] * df['thalach']
        df['chol_thalach'] = df['chol'] * df['thalach']
        df['age_sex'] = df['age'] * df['sex']
        
    elif dataset_name == "Framingham数据集":
        # 年龄分段
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], labels=[1, 2, 3, 4, 5])
        
        # BMI分组
        df['bmi_group'] = pd.cut(df['BMI'], bins=[0, 18.5, 24, 28, 32, 50], labels=[1, 2, 3, 4, 5])
        
        # 血压分级
        df['bp_group'] = pd.cut(df['sysBP'], bins=[0, 120, 140, 160, 200], labels=[1, 2, 3, 4])
        
        # 交互特征
        df['age_bp'] = df['age'] * df['sysBP']
        df['bmi_age'] = df['BMI'] * df['age']
        df['smoke_age'] = df['currentSmoker'] * df['age']
        df['age_chol'] = df['age'] * df['totChol']
        df['bp_chol'] = df['sysBP'] * df['totChol']
        df['age_glucose'] = df['age'] * df['glucose']
        df['bmi_bp'] = df['BMI'] * df['sysBP']
    
    return df

def load_data(dataset_name):
    """加载数据集"""
    try:
        if dataset_name == "UCI心脏病数据集":
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            # 尝试相对路径
            try:
                df = pd.read_csv('heart+disease/processed.cleveland.data', names=columns)
            except FileNotFoundError:
                # 如果相对路径失败，尝试使用绝对路径
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_path = os.path.join(current_dir, 'framingham.csv')
                df = pd.read_csv(data_path)
            
            # 重命名目标变量
            df = df.rename(columns={'TenYearCHD': 'target'})
        
        # 处理NA值，确保所有列类型正确
        for col in df.columns:
            # 对于数值列，填充NA为NaN
            if df[col].dtype in ['int64', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 对于类别列，填充NA为最常见值
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else pd.NA)
        
        # 构造医学相关的交互特征
        df = add_medical_interaction_features(df, dataset_name)
        
        return df
    except Exception as e:
        raise Exception(f"加载数据时出错: {str(e)}")

def preprocess_data(X_train, X_test, y_train, dataset_name):
    """数据预处理"""
    # 处理 UCI 心脏病数据集
    if dataset_name == "UCI心脏病数据集":
        # 训练集处理：使用更智能的填充方法，而不是直接删除
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # 先处理数值类型转换
        for col in ['ca', 'thal']:
            if col in train_data.columns:
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        
        # 按目标变量分组填充缺失值（比直接删除更好）
        for target_group in train_data['target'].unique():
            group_mask = train_data['target'] == target_group
            group_data = train_data[group_mask]
            
            # 数值列用均值填充
            numeric_cols = group_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if train_data.loc[group_mask, col].isnull().sum() > 0:
                    fill_value = group_data[col].mean()
                    train_data.loc[group_mask, col] = train_data.loc[group_mask, col].fillna(fill_value)
        
        # 如果还有缺失值，用全局中位数填充
        if train_data.isnull().sum().sum() > 0:
            for col in train_data.columns:
                if train_data[col].isnull().sum() > 0:
                    train_data[col] = train_data[col].fillna(train_data[col].median())
        
        X_train_clean = train_data.drop('target', axis=1)
        y_train_clean = train_data['target']
        
        # 测试集处理：使用训练集的统计量填充
        X_test_clean = X_test.copy()
        X_test_clean['ca'] = pd.to_numeric(X_test_clean['ca'], errors='coerce')
        X_test_clean['thal'] = pd.to_numeric(X_test_clean['thal'], errors='coerce')
        
        # 使用训练集的统计量填充测试集
        for col in X_test_clean.columns:
            if X_test_clean[col].isnull().sum() > 0:
                X_test_clean[col] = X_test_clean[col].fillna(X_train_clean[col].median())
        
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

def winsorize_outliers(df, feature_cols, lower_quantile=0.01, upper_quantile=0.99):
    """分位数截断处理异常值"""
    df_clean = df.copy()
    for col in feature_cols:
        lower = df_clean[col].quantile(lower_quantile)
        upper = df_clean[col].quantile(upper_quantile)
        df_clean[col] = df_clean[col].clip(lower, upper)
    return df_clean

def get_categorical_features(dataset_name):
    """获取类别特征列表"""
    if dataset_name == "UCI心脏病数据集":
        return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'bp_group', 'chol_group']
    elif dataset_name == "Framingham数据集":
        return ['male', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'age_group', 'bmi_group', 'bp_group']
    return []

def get_numeric_features(dataset_name):
    """获取数值特征列表"""
    if dataset_name == "UCI心脏病数据集":
        return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'age_bp', 'age_chol', 'age_thalach', 'bp_chol', 'bp_thalach', 'chol_thalach', 'age_sex']
    elif dataset_name == "Framingham数据集":
        return ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'age_bp', 'bmi_age', 'smoke_age', 'age_chol', 'bp_chol', 'age_glucose', 'bmi_bp']
    return []

class Winsorizer(BaseEstimator, TransformerMixin):
    """分位数截断转换器"""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        # 确保X是数值型数组
        X = np.array(X, dtype=np.float64)
        for i in range(X.shape[1]):
            column = X[:, i]
            # 计算分位数，忽略NaN
            self.bounds_[i] = (np.nanpercentile(column, self.lower_quantile * 100),
                             np.nanpercentile(column, self.upper_quantile * 100))
        return self
    
    def transform(self, X):
        # 确保X是数值型数组
        X = np.array(X, dtype=np.float64)
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            lower, upper = self.bounds_[i]
            # 处理NA值
            X_transformed[:, i] = np.clip(X_transformed[:, i], lower, upper)
        return X_transformed

def create_preprocessing_pipeline(dataset_name):
    """创建预处理管道"""
    from sklearn.impute import SimpleImputer
    
    categorical_features = get_categorical_features(dataset_name)
    numeric_features = get_numeric_features(dataset_name)
    
    # 数值特征处理：填充缺失值 + 分位数截断 + 标准化
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', Winsorizer()),
        ('scaler', StandardScaler())
    ])
    
    # 类别特征处理：填充缺失值 + 独热编码
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def split_and_preprocess_data(X, y, dataset_name, test_size=0.2, random_state=0, use_smote=True, smote_ratio=1.0, resampling_method='SMOTE'):
    """划分并预处理数据"""
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 创建并应用预处理管道
    preprocessor = create_preprocessing_pipeline(dataset_name)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 应用重采样
    if use_smote:
        # 计算正例比例
        pos_ratio = sum(y_train) / len(y_train)
        # 只在正例比例较低时应用重采样
        if pos_ratio < 0.3:
            # 计算需要的正例数量
            neg_count = len(y_train) - sum(y_train)
            desired_pos_count = int(neg_count * smote_ratio)
            # 计算采样比例
            sampling_strategy = desired_pos_count / sum(y_train)
            
            # 根据选择的重采样方法进行处理
            if resampling_method == 'SMOTE':
                sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
            elif resampling_method == 'Borderline-SMOTE':
                sampler = BorderlineSMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
            elif resampling_method == 'ADASYN':
                sampler = ADASYN(random_state=random_state, sampling_strategy=sampling_strategy)
            elif resampling_method == 'SMOTE+Tomek':
                sampler = SMOTETomek(random_state=random_state, sampling_strategy=sampling_strategy)
            elif resampling_method == 'SMOTE+ENN':
                sampler = SMOTEENN(random_state=random_state, sampling_strategy=sampling_strategy)
            else:
                sampler = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
            
            X_train_processed, y_train = sampler.fit_resample(X_train_processed, y_train)
            print(f"{resampling_method}应用后: 训练集大小={len(X_train_processed)}, 正例比例={sum(y_train)/len(y_train):.3f}, 采样比例={smote_ratio}:1")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def save_model(model, preprocessor, model_name, dataset_name):
    """保存模型和预处理管道"""
    model_path = f"models/{dataset_name}_{model_name}.pkl"
    os.makedirs('models', exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'preprocessor': preprocessor}, f)
    
    return model_path

def load_model(model_name, dataset_name):
    """加载模型和预处理管道"""
    model_path = f"models/{dataset_name}_{model_name}.pkl"
    
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['preprocessor']