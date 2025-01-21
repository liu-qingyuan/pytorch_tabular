import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import torch
import json

from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    TabNetModelConfig,
    DANetConfig,
    AutoIntConfig,
    TabTransformerConfig
)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model_tuner import TabularModelTuner

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tune_selected_models(data_path):
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

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    print("\nData Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # Configure data
    data_config = DataConfig(
        target=['target'],
        continuous_cols=features,
        categorical_cols=[],
    )

    # 配置训练器
    trainer_config = TrainerConfig(
        batch_size=1024,
        max_epochs=200,  # 增加训练轮数
        early_stopping="valid_loss",
        early_stopping_patience=20,  # 增加早停耐心值
        learning_rate=0.001,
        optimizer="Adam",
        optimizer_params={"weight_decay": 0.01},  # 添加L2正则化
        scheduler="ReduceLROnPlateau",  # 添加学习率调度器
        scheduler_params={
            "mode": "max",
            "patience": 5,
            "factor": 0.5,
            "min_lr": 1e-6
        }
    )

    optimizer_config = OptimizerConfig()

    # Configure head with more options
    head_config = LinearHeadConfig(
        layers="256-128-64",  # 添加更多层
        dropout=0.2,
        initialization="kaiming",
        use_batch_norm=True
    ).__dict__

    # 定义要调优的模型配置
    model_configs = [
        # 1. DANet
        DANetConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config,
            metrics=["accuracy", "auroc"],
            metrics_params=[{}, {}],
            metrics_prob_input=[False, True],
            n_layers=6,
            abstlay_dim_1=32,
            abstlay_dim_2=64,
            dropout_rate=0.1
        ),
        # 2. TabTransformer
        TabTransformerConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config,
            metrics=["accuracy", "auroc"],
            metrics_params=[{}, {}],
            metrics_prob_input=[False, True],
            input_embed_dim=32,
            num_heads=8,
            num_attn_blocks=3,
            embedding_dropout=0.1
        ),
        # 3. AutoInt
        AutoIntConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config,
            metrics=["accuracy", "auroc"],
            metrics_params=[{}, {}],
            metrics_prob_input=[False, True],
            attn_embed_dim=32,
            num_heads=2,
            num_attn_blocks=3,
            attn_dropouts=0.0,
            has_residuals=True,
            embedding_dim=16
        ),
        # 4. TabNet
        TabNetModelConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config,
            metrics=["accuracy", "auroc"],
            metrics_params=[{}, {}],
            metrics_prob_input=[False, True],
            n_d=32,
            n_a=32,
            n_steps=3,
            gamma=1.5,
            n_independent=1,
            n_shared=2
        )
    ]

    # 为每个模型定义更大的搜索空间
    search_spaces = [
        # DANet
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__learning_rate": [0.1, 0.01, 0.001, 0.0001],
            "model_config__n_layers": [4, 6, 8, 10],
            "model_config__abstlay_dim_1": [32, 64, 128, 256],
            "model_config__abstlay_dim_2": [64, 128, 256, 512],
            "model_config__dropout_rate": [0.0, 0.1, 0.2, 0.3],
            "trainer_config__batch_size": [512, 1024, 2048]
        },
        # TabTransformer
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__learning_rate": [0.1, 0.01, 0.001, 0.0001],
            "model_config__input_embed_dim": [32, 64, 128, 256],
            "model_config__num_attn_blocks": [2, 3, 4, 6],
            "model_config__num_heads": [4, 8, 16],
            "model_config__embedding_dropout": [0.0, 0.1, 0.2, 0.3],
            "model_config__attention_dropout": [0.0, 0.1, 0.2],
            "model_config__ff_dropout": [0.0, 0.1, 0.2],
            "trainer_config__batch_size": [512, 1024, 2048]
        },
        # AutoInt
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__learning_rate": [0.1, 0.01, 0.001, 0.0001],
            "model_config__attn_embed_dim": [32, 64, 128, 256],
            "model_config__num_heads": [2, 4, 8, 16],
            "model_config__num_attn_blocks": [2, 3, 4, 6],
            "model_config__attn_dropouts": [0.0, 0.1, 0.2, 0.3],
            "model_config__has_residuals": [True, False],
            "model_config__embedding_dim": [16, 32, 64, 128],
            "model_config__deep_layers": [True, False],
            "model_config__attention_pooling": [True, False],
            "trainer_config__batch_size": [512, 1024, 2048]
        },
        # TabNet
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__learning_rate": [0.1, 0.01, 0.001, 0.0001],
            "model_config__n_d": [8, 16, 32, 64],
            "model_config__n_a": [8, 16, 32, 64],
            "model_config__n_steps": [3, 5, 7],
            "model_config__gamma": [1.0, 1.3, 1.5, 1.7],
            "model_config__n_independent": [1, 2, 3],
            "model_config__n_shared": [1, 2, 3],
            "model_config__virtual_batch_size": [128, 256, 512],
            "trainer_config__batch_size": [512, 1024, 2048]
        }
    ]

    # 加载历史最佳配置（如果存在）
    best_configs_file = 'results/best_configs.json'
    if os.path.exists(best_configs_file):
        with open(best_configs_file, 'r') as f:
            historical_best = json.load(f)
        print("\nLoaded historical best configurations")
    else:
        historical_best = {}

    print("\nStarting Hyperparameter Tuning...")
    
    tuner = TabularModelTuner(
        data_config=data_config,
        model_config=model_configs,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    tuner_df = tuner.tune(
        train=train_df,
        validation=test_df,
        search_space=search_spaces,
        strategy="random_search",
        n_trials=30,  # 增加调参次数
        metric="auroc",
        mode="max",
        progress_bar=True,
        verbose=True
    )

    # Save tuning results
    tuner_df.trials_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)
    
    # 创建一个结果汇总DataFrame
    summary_results = []
    
    # 保存每个模型的最佳配置
    best_configs = {}
    
    # 按模型分组显示结果
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    for model_idx, model_name in enumerate(model_names):
        model_results = tuner_df.trials_df[tuner_df.trials_df['model'].str.startswith(f"{model_idx}-")]
        
        # 获取最佳AUC结果
        best_auc_row = model_results.sort_values("auroc", ascending=False).iloc[0]
        best_auc = best_auc_row["auroc"]
        best_auc_acc = best_auc_row["accuracy"]
        
        # 保存最佳配置
        best_configs[model_name] = {
            'params': {col: best_auc_row[col] for col in best_auc_row.index if 'model_config__' in col or 'optimizer_config__' in col or 'trainer_config__' in col},
            'performance': {
                'auc': best_auc,
                'accuracy': best_auc_acc
            }
        }
        
        # 如果性能提升，更新历史最佳
        if model_name not in historical_best or best_auc > historical_best[model_name]['performance']['auc']:
            historical_best[model_name] = best_configs[model_name]
            print(f"\nNew best configuration found for {model_name}!")
        
        # 添加到汇总结果
        summary_results.append({
            "Model": model_name,
            "Best_AUC": best_auc,
            "Accuracy_at_Best_AUC": best_auc_acc,
            "Best_Accuracy": best_auc_acc,
            "AUC_at_Best_Accuracy": best_auc
        })
        
        print(f"\n\nResults for {model_name}:")
        print(f"Best AUC: {best_auc:.4f} (Accuracy: {best_auc_acc:.4f})")
        print(f"Best Accuracy: {best_auc_acc:.4f} (AUC: {best_auc:.4f})")
        
        print("\nTop 3 by AUC:")
        print(model_results.sort_values("auroc", ascending=False).head(3))
        
        print("\nTop 3 by Accuracy:")
        print(model_results.sort_values("accuracy", ascending=False).head(3))
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/model_summary.csv', index=False)
    
    # 保存最佳配置
    with open(best_configs_file, 'w') as f:
        json.dump(historical_best, f, indent=4)
    print(f"\nBest configurations saved to {best_configs_file}")
    
    # 在控制台输出性能汇总和历史最佳
    print("\n" + "="*80)
    print("本次调参结果与历史最佳比较:")
    print("="*80)
    for model_name in model_names:
        print(f"\n{model_name}:")
        print("  本次最佳:")
        print(f"    AUC: {best_configs[model_name]['performance']['auc']:.4f}")
        print(f"    Accuracy: {best_configs[model_name]['performance']['accuracy']:.4f}")
        print("  历史最佳:")
        print(f"    AUC: {historical_best[model_name]['performance']['auc']:.4f}")
        print(f"    Accuracy: {historical_best[model_name]['performance']['accuracy']:.4f}")
    print("="*80)

    return tuner_df, summary_df, historical_best

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"  # 修改为正确的数据路径
    tuner_df, summary_df, historical_best = tune_selected_models(data_path) 