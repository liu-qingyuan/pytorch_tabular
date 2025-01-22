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
import joblib

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
    # Save the scaler
    joblib.dump(scaler, "results/scaler.pkl")
    print("✅ Saved scaler to results/scaler.pkl")

    # Split data into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 准备训练集和测试集
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    # Save test set
    test_df.to_csv("results/test_set.csv", index=False)
    print("✅ Saved test set to results/test_set.csv")

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
        early_stopping_patience=20  # 增加早停耐心值
    )

    optimizer_config = OptimizerConfig(
        optimizer="Adam",
        optimizer_params={"lr": 0.001, "weight_decay": 0.01},
        lr_scheduler="ReduceLROnPlateau",
        lr_scheduler_params={"mode": "min", "patience": 5, "factor": 0.1},
        lr_scheduler_monitor_metric="valid_loss"
    )

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
            embedding_initialization="kaiming_uniform",
            embedding_bias=False,
            share_embedding=False,
            share_embedding_strategy="fraction",
            shared_embedding_fraction=0.25,
            num_heads=8,
            num_attn_blocks=6,
            embedding_dropout=0.1,
            transformer_head_dim=None,
            attn_dropout=0.1,
            add_norm_dropout=0.1,
            ff_dropout=0.1,
            ff_hidden_multiplier=4,
            transformer_activation="GEGLU",
            batch_norm_continuous_input=True
        ),
        # 3. AutoInt
        AutoIntConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config,
            metrics=["accuracy", "auroc"],
            metrics_params=[{}, {}],
            metrics_prob_input=[False, True],
            embedding_dim=63,  # 与 layers 的第一层维度匹配
            attn_embed_dim=16,  # 与 layers 的最后一层维度匹配
            # num_heads=2,
            num_attn_blocks=3,
            attn_dropouts=0.0,
            has_residuals=True,
            batch_norm_continuous_input=True,
            embedding_initialization="kaiming_uniform",
            embedding_bias=True,
            share_embedding=True,
            share_embedding_strategy="add",
            shared_embedding_fraction=0.25,
            deep_layers=False,
            # layers="63-24-16",  # 从嵌入维度(32)到注意力维度(16)
            use_batch_norm=True,
            attention_pooling=True
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

    # 为每个模型定义搜索空间
    search_spaces = [
        # DANet
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__optimizer_params": [
                {"weight_decay": 0.01},
                {"weight_decay": 0.001}
            ],
            "model_config__n_layers": [4, 6, 8],  # 增加层数选项
            "model_config__abstlay_dim_1": [16, 32, 64],  # 增加维度选项
            "model_config__abstlay_dim_2": [32, 64, 128],  # 增加维度选项
            "model_config__k": [3, 5, 7],  # 增加 k 值选项
            "model_config__dropout_rate": [0.0, 0.1, 0.2],  # 增加 dropout 选项
            "model_config__block_activation": ["ReLU", "LeakyReLU", "GELU"],  # 增加激活函数选项
            "model_config__virtual_batch_size": [128, 256, 512]  # 增加批量大小选项
        },
        # TabTransformer
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__optimizer_params": [
                {"weight_decay": 0.01},
                {"weight_decay": 0.001}
            ],
            "model_config__input_embed_dim": [16, 32, 64],  # 增加嵌入维度选项
            "model_config__num_attn_blocks": [2, 4, 6],  # 增加注意力块数量
            "model_config__num_heads": [4, 8, 16],  # 增加头数选项
            "model_config__transformer_head_dim": [None, 16, 32],  # 增加头维度选项
            "model_config__attn_dropout": [0.0, 0.1, 0.2],  # 增加注意力dropout选项
            "model_config__add_norm_dropout": [0.0, 0.1, 0.2],  # 增加归一化dropout选项
            "model_config__ff_dropout": [0.0, 0.1, 0.2],  # 增加前馈dropout选项
            "model_config__ff_hidden_multiplier": [2, 4, 8],  # 增加隐藏层倍数选项
            "model_config__transformer_activation": ["GEGLU", "ReGLU", "SwiGLU"],  # 使用支持的激活函数
            "model_config__embedding_initialization": ["kaiming_uniform", "kaiming_normal"],
            "model_config__embedding_bias": [True, False],
            "model_config__embedding_dropout": [0.0, 0.1, 0.2],
            "model_config__share_embedding": [True, False],
            "model_config__share_embedding_strategy": ["add", "fraction"],
            "model_config__shared_embedding_fraction": [0.25, 0.5, 0.75]
        },
        # AutoInt
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__optimizer_params": [
                {"weight_decay": 0.01},
                {"weight_decay": 0.001}
            ],
            "model_config__embedding_dim": [63, 32, 16],  # 固定为输入维度
            "model_config__attn_embed_dim": [16, 32, 64],  # 注意力层维度选项
            "model_config__num_heads": [2, 4, 8],
            "model_config__num_attn_blocks": [2, 3, 4],
            "model_config__attn_dropouts": [0.0, 0.1, 0.2],
            "model_config__has_residuals": [True, False],
            "model_config__embedding_initialization": ["kaiming_uniform", "kaiming_normal"],
            "model_config__embedding_bias": [True, False],
            "model_config__share_embedding": [True, False],
            "model_config__share_embedding_strategy": ["add", "fraction"],
            "model_config__shared_embedding_fraction": [0.25, 0.5],
            "model_config__deep_layers": [False],
            "model_config__use_batch_norm": [True],
            "model_config__attention_pooling": [True, False],
            "model_config__activation": ["ReLU", "LeakyReLU"],  # 添加激活函数选项
            "model_config__initialization": ["kaiming", "xavier"],  # 添加初始化方法选项
            "model_config__dropout": [0.0, 0.1, 0.2]  # 添加dropout选项
        },
        # TabNet
        {
            "optimizer_config__optimizer": ["Adam", "AdamW"],
            "optimizer_config__optimizer_params": [
                {"weight_decay": 0.01},
                {"weight_decay": 0.001}
            ],
            "model_config__n_d": [8, 16, 32, 64],  # 预测层维度 (4-64)
            "model_config__n_a": [8, 16, 32, 64],  # 注意力层维度 (4-64)
            "model_config__n_steps": [3, 5, 7, 10],  # 网络步骤数 (3-10)
            "model_config__gamma": [1.0, 1.3, 1.5, 2.0],  # 注意力更新的缩放因子 (1.0-2.0)
            "model_config__n_independent": [1, 2, 3],  # 独立 GLU 层数
            "model_config__n_shared": [1, 2, 3],  # 共享 GLU 层数
            "model_config__virtual_batch_size": [128, 256, 512],  # Ghost Batch Normalization 的批量大小
            "model_config__mask_type": ["sparsemax", "entmax"],  # 掩码函数类型
            "model_config__batch_norm_continuous_input": [True],
            "model_config__embedding_dropout": [0.0, 0.1, 0.2]  # 嵌入层的 dropout 率
        }
    ]

    # 加载历史最佳配置（如果存在）
    best_configs_file = 'results/best_configs.json'
    historical_best = {}
    if os.path.exists(best_configs_file):
        try:
            with open(best_configs_file, 'r') as f:
                historical_best = json.load(f)
            print("\nLoaded historical best configurations")
        except json.JSONDecodeError:
            print("\nWarning: best_configs.json is corrupted, starting fresh")
            historical_best = {}

    print("\nStarting Hyperparameter Tuning...")
    
    all_tuning_results = []
    best_models = {}
    summary_results = []
    
    # 按顺序调优每个模型
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    for model_idx, (model_config, search_space) in enumerate(zip(model_configs, search_spaces)):
        model_name = model_names[model_idx]
        print(f"\nTuning {model_name}...")
        
        tuner = TabularModelTuner(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )

        tuner_df = tuner.tune(
            train=train_df,
            validation=test_df,  # 使用测试集作为验证集
            search_space=search_space,
            strategy="random_search",
            n_trials=2,  # 每个模型2次尝试
            metric="auroc",  # 使用auroc作为主要优化指标
            mode="max",
            progress_bar=True,
            verbose=True,
            return_best_model=True  # 确保返回最佳模型
        )
        
        # 保存每个模型的调优结果
        tuner_df.trials_df['model'] = model_name
        all_tuning_results.append(tuner_df.trials_df)
        
        if tuner_df.best_model is not None:
            best_models[model_name] = tuner_df.best_model
            # 保存最佳模型
            model_save_path = f"results/{model_name}_best.pt"
            tuner_df.best_model.save_model(model_save_path)
            print(f"✅ Saved best model to {model_save_path}")
        
        # 获取最佳结果
        best_trial = tuner_df.trials_df.sort_values("auroc", ascending=False).iloc[0]
        
        # 获取最佳性能指标
        best_auc = best_trial["auroc"]
        best_acc = best_trial.get("accuracy", None)  # 尝试获取accuracy
        
        print(f"\nResults for {model_name}:")
        print(f"Best AUC: {best_auc:.4f}")
        if best_acc is not None:
            print(f"Accuracy at best AUC: {best_acc:.4f}")
        
        print("\nTop 3 configurations by AUC:")
        print(tuner_df.trials_df.sort_values("auroc", ascending=False).head(3))
        
        # 如果性能提升，更新历史最佳
        if model_name not in historical_best or best_auc > historical_best[model_name]['performance']['auc']:
            # 转换参数值为Python原生类型
            params_dict = {}
            for col in best_trial.index:
                if 'model_config__' in col or 'optimizer_config__' in col:
                    value = best_trial[col]
                    # 转换numpy/pandas类型为Python原生类型
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    elif isinstance(value, dict):
                        # 处理字典中的numpy类型
                        value = {k: v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v 
                               for k, v in value.items()}
                    params_dict[col] = value
            
            historical_best[model_name] = {
                'params': params_dict,
                'performance': {
                    'auc': float(best_auc),
                    'accuracy': float(best_acc) if best_acc is not None else 0.0,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            print(f"\n🏆 New best configuration found for {model_name}!")
        
        # 添加到汇总结果
        summary_results.append({
            "model_name": model_name,
            "best_auroc": best_auc,
            "accuracy_at_best_auroc": best_acc if best_acc is not None else 0.0,
            "historical_best_auroc": historical_best[model_name]['performance']['auc'],
            "historical_best_accuracy": historical_best[model_name]['performance']['accuracy'],
            "historical_best_timestamp": historical_best[model_name]['performance'].get('timestamp', 'N/A')
        })
    
    # 合并所有调优结果
    combined_tuning_df = pd.concat(all_tuning_results, ignore_index=True)
    combined_tuning_df.to_csv('results/all_trials.csv', index=False)
    print("✅ Saved all trials to results/all_trials.csv")
    
    # 保存历史最佳配置
    with open(best_configs_file, 'w') as f:
        json.dump(historical_best, f, indent=4)
    print("✅ Saved historical best configurations to results/best_configs.json")
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/summary.csv', index=False)
    print("✅ Saved summary to results/summary.csv")
    
    # 打印汇总结果
    print("\nModel Performance Summary:")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"\nModel: {row['model_name']}")
        print(f"Current Best AUROC: {row['best_auroc']:.4f} (Accuracy: {row['accuracy_at_best_auroc']:.4f})")
        print(f"Historical Best AUROC: {row['historical_best_auroc']:.4f} (Accuracy: {row['historical_best_accuracy']:.4f})")
        print(f"Historical Best Achieved: {row['historical_best_timestamp']}")
    print("=" * 80)
    
    return combined_tuning_df, summary_df, best_models, historical_best

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"
    tuner_df, summary_df, best_models, historical_best = tune_selected_models(data_path) 