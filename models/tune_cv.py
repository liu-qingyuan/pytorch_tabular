import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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

    print("\nData Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())

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

    print("\nStarting 10-Fold Cross-Validation Hyperparameter Tuning...")
    
    # 初始化10折交叉验证
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    all_fold_results = []
    best_models = {}
    summary_results = []
    
    # 按顺序调优每个模型
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    
    for model_idx, (model_config, search_space) in enumerate(zip(model_configs, search_spaces)):
        model_name = model_names[model_idx]
        print(f"\n{'='*50}")
        print(f"Tuning {model_name}...")
        print(f"{'='*50}")
        
        fold_results = []
        fold_models = []
        
        # 对每个fold进行训练和验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            print(f"\n--- Fold {fold}/10 ---")
            
            # 准备当前fold的数据
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_df = X_train.copy()
            train_df['target'] = y_train
            val_df = X_val.copy()
            val_df['target'] = y_val
            
            # 为每个fold设置不同的随机种子
            set_seed(42 + fold)
            
            tuner = TabularModelTuner(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config
            )

            tuner_df = tuner.tune(
                train=train_df,
                validation=val_df,
                search_space=search_space,
                strategy="random_search",
                n_trials=300,  # 增加到300次尝试
                metric="auroc",
                mode="max",
                progress_bar=True,
                verbose=True,
                return_best_model=True
            )
            
            # 保存每个fold的调优结果
            tuner_df.trials_df['fold'] = fold
            tuner_df.trials_df['model'] = model_name
            fold_results.append(tuner_df.trials_df)
            
            if tuner_df.best_model is not None:
                fold_models.append(tuner_df.best_model)
                # 保存fold模型
                model_save_path = f"results/{model_name}_fold{fold}"
                tuner_df.best_model.save_model(model_save_path)
                print(f"✅ Saved fold {fold} model to {model_save_path}")
            
            # 获取当前fold的最佳结果
            best_trial = tuner_df.trials_df.sort_values("auroc", ascending=False).iloc[0]
            best_auc = best_trial["auroc"]
            best_acc = best_trial.get("accuracy", None)
            
            print(f"\nFold {fold} Results:")
            print(f"AUC: {best_auc:.4f}")
            if best_acc is not None:
                print(f"Accuracy: {best_acc:.4f}")
        
        # 合并所有fold的结果
        model_results = pd.concat(fold_results, ignore_index=True)
        all_fold_results.append(model_results)
        
        # 计算交叉验证平均性能
        mean_auc = model_results.groupby('fold')['auroc'].max().mean()
        std_auc = model_results.groupby('fold')['auroc'].max().std()
        mean_acc = model_results.groupby('fold')['accuracy'].max().mean() if 'accuracy' in model_results.columns else None
        std_acc = model_results.groupby('fold')['accuracy'].max().std() if 'accuracy' in model_results.columns else None
        
        print(f"\n{model_name} Cross-Validation Results:")
        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        if mean_acc is not None:
            print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # 更新历史最佳配置
        if model_name not in historical_best or mean_auc > historical_best[model_name]['performance']['auc']:
            # 获取最佳fold的配置
            best_fold_idx = model_results.groupby('fold')['auroc'].max().idxmax()
            best_trial = model_results[model_results['fold'] == best_fold_idx].sort_values("auroc", ascending=False).iloc[0]
            
            params_dict = {}
            for col in best_trial.index:
                if 'model_config__' in col or 'optimizer_config__' in col:
                    value = best_trial[col]
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    elif isinstance(value, dict):
                        value = {k: v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v 
                               for k, v in value.items()}
                    params_dict[col] = value
            
            historical_best[model_name] = {
                'params': params_dict,
                'performance': {
                    'auc': float(mean_auc),
                    'auc_std': float(std_auc),
                    'accuracy': float(mean_acc) if mean_acc is not None else 0.0,
                    'accuracy_std': float(std_acc) if std_acc is not None else 0.0,
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            print(f"\n🏆 New best configuration found for {model_name}!")
        
        # 保存最佳模型集成
        best_models[model_name] = fold_models
        
        # 添加到汇总结果
        summary_results.append({
            "model_name": model_name,
            "mean_auroc": mean_auc,
            "std_auroc": std_auc,
            "mean_accuracy": mean_acc if mean_acc is not None else 0.0,
            "std_accuracy": std_acc if std_acc is not None else 0.0,
            "historical_best_auroc": historical_best[model_name]['performance']['auc'],
            "historical_best_accuracy": historical_best[model_name]['performance']['accuracy'],
            "historical_best_timestamp": historical_best[model_name]['performance'].get('timestamp', 'N/A')
        })
    
    # 合并所有调优结果
    combined_tuning_df = pd.concat(all_fold_results, ignore_index=True)
    combined_tuning_df.to_csv('results/all_cv_trials.csv', index=False)
    print("\n✅ Saved all cross-validation trials to results/all_cv_trials.csv")
    
    # 保存历史最佳配置
    with open(best_configs_file, 'w') as f:
        json.dump(historical_best, f, indent=4)
    print("✅ Saved historical best configurations to results/best_configs.json")
    
    # 创建汇总DataFrame
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv('results/cv_summary.csv', index=False)
    print("✅ Saved cross-validation summary to results/cv_summary.csv")
    
    # 打印汇总结果
    print("\nModel Performance Summary:")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        print(f"\nModel: {row['model_name']}")
        print(f"CV AUROC: {row['mean_auroc']:.4f} ± {row['std_auroc']:.4f}")
        print(f"CV Accuracy: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
        print(f"Historical Best AUROC: {row['historical_best_auroc']:.4f}")
        print(f"Historical Best Achieved: {row['historical_best_timestamp']}")
    print("=" * 80)
    
    return combined_tuning_df, summary_df, best_models, historical_best

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"
    tuner_df, summary_df, best_models, historical_best = tune_selected_models(data_path) 