from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# 尝试导入LightGBM和CatBoost
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from sklearn.neural_network import MLPClassifier
except ImportError:
    MLPClassifier = None

def train_model(model_name, X_train, y_train):
    """训练模型"""
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=200, class_weight='balanced', C=0.1)  # 增加正则化
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=80, 
            class_weight='balanced',
            # 增加正则化，减少过拟合
            min_samples_leaf=15,
            min_samples_split=15,
            max_depth=10
        )
    elif model_name == "XGBoost":
        # 计算类别权重
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        # 添加更多参数来处理不平衡数据和过拟合
        model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight,
            # 减少树的深度，避免过拟合
            max_depth=4,
            # 降低学习率
            learning_rate=0.01,
            # 增加迭代次数
            n_estimators=1000,
            # 增加正则化
            reg_alpha=1.0,
            reg_lambda=10.0,
            # 增加子采样，减少过拟合
            subsample=0.8,
            colsample_bytree=0.8,
            # 增加最小子节点权重
            min_child_weight=3
        )
    elif model_name == "LightGBM" and LGBMClassifier is not None:
        # 计算类别权重
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        model = LGBMClassifier(
            class_weight='balanced',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=10.0,
            min_child_samples=15,
            is_unbalance=False,
            scale_pos_weight=scale_pos_weight
        )
    elif model_name == "CatBoost" and CatBoostClassifier is not None:
        # 计算类别权重
        class_weights = {0: 1, 1: len(y_train[y_train==0])/len(y_train[y_train==1])} if len(y_train[y_train==1]) > 0 else {0: 1, 1: 1}
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=10,
            class_weights=class_weights,
            subsample=0.8,
            colsample_bylevel=0.8,
            verbose=0
        )
    elif model_name == "Neural Network" and MLPClassifier is not None:
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            learning_rate_init=0.001,
            random_state=42
        )
    else:
        # 默认使用XGBoost
        scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
        model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight,
            max_depth=4,
            learning_rate=0.01,
            n_estimators=1000,
            reg_alpha=1.0,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3
        )
    
    # 训练模型
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(X_train, y_train, model_type='xgb'):
    """超参数调优"""
    if model_type == 'xgb':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(class_weight='balanced')
    else:  # lr
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        model = LogisticRegression(max_iter=200, class_weight='balanced')
    
    # 使用随机搜索（更快）
    random_search = RandomizedSearchCV(
        model, param_grid, n_iter=20, cv=5,
        scoring='roc_auc', random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search

def compare_models_detailed(X_train, X_test, y_train, y_test):
    """详细的模型对比分析"""
    # 计算类别权重
    scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=200,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # 使用分层交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # 训练模型
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算各项指标
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba),
            'CV_AUC_Mean': cv_scores.mean(),
            'CV_AUC_Std': cv_scores.std()
        })
        
        trained_models[name] = model
    
    return pd.DataFrame(results), trained_models