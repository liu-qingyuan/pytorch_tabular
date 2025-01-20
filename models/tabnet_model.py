import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import torch

from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_tabnet_experiment(data_path):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Set random seed
    set_seed(42)

    # Read data
    df = pd.read_excel(data_path)
    features = [c for c in df.columns if c.startswith("Feature")]
    X = df[features].copy()
    y = df["Label"].copy()

    # 标准化特征
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())

    # Initialize results storage
    fold_metrics = []

    # 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}/10")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create training dataframe
        train_df = X_train.copy()
        train_df['target'] = y_train
        val_df = X_val.copy()
        val_df['target'] = y_val

        # Calculate class weights
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weights = {i: total/(len(class_counts)*v) for i, v in class_counts.items()}
        print(f"Class weights: {class_weights}")

        # Configure model
        data_config = DataConfig(
            target=['target'],
            continuous_cols=features,
            categorical_cols=[],
        )

        trainer_config = TrainerConfig(
            batch_size=128,  # 增大batch size
            max_epochs=200,  # 增加训练轮数
            early_stopping="valid_loss",
            early_stopping_patience=50,  # 增加耐心值
            auto_lr_find=True,  # 启用自动学习率查找
        )

        optimizer_config = OptimizerConfig()

        model_config = TabNetModelConfig(
            task="classification",
            n_d=16,  # 增大网络容量
            n_a=16,
            n_steps=10,
            gamma=1.3,
            n_independent=4,
            n_shared=4,
            virtual_batch_size=128,
            mask_type="entmax",  # 使用entmax激活
        )

        # Train model
        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        
        model.fit(train=train_df, validation=val_df)

        # Get predictions
        pred_df = model.predict(val_df)
        
        print("Prediction columns:", pred_df.columns.tolist())
        
        # 获取预测结果
        if 'target_prediction' in pred_df.columns:
            y_pred = pred_df['target_prediction'].values
            y_pred_proba = pred_df['target_1_probability'].values
        else:
            raise KeyError(f"Cannot find prediction column in {pred_df.columns}")

        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        acc_0 = accuracy_score(y_val[y_val==0], y_pred[y_val==0])
        acc_1 = accuracy_score(y_val[y_val==1], y_pred[y_val==1])
        g_mean = np.sqrt(acc_0 * acc_1)

        print(f"\nFold {fold+1} Results:")
        print(f"AUC: {auc:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"ACC: {acc:.4f}")
        print(f"ACC_0: {acc_0:.4f}")
        print(f"ACC_1: {acc_1:.4f}")
        print(f"G-mean: {g_mean:.4f}")

        # Store fold results
        fold_result = {
            'Fold': fold+1,
            'AUC': auc,
            'F1': f1,
            'ACC': acc,
            'ACC_0': acc_0,
            'ACC_1': acc_1,
            'G-mean': g_mean
        }
        fold_metrics.append(fold_result)

    # Save per-fold results
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv('results/TabNet-Health.csv', index=False)

    # Calculate and save final averaged results
    final_metrics = fold_df.mean().round(4)
    final_df = pd.DataFrame(final_metrics).T
    final_df.to_csv('results/TabNet-Health-Final.csv', index=False)

    print("\nFinal Average Results:")
    for metric, value in final_metrics.items():
        if metric != 'Fold':
            print(f"{metric}: {value:.4f}")

    return fold_df, final_metrics 