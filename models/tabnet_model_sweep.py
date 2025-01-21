import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import torch
from torchmetrics.classification import AUROC

from pytorch_tabular import TabularModel, model_sweep
from pytorch_tabular.models import TabNetModelConfig, CategoryEmbeddingModelConfig, GANDALFConfig, FTTransformerConfig, AutoIntConfig, DANetConfig, GatedAdditiveTreeEnsembleConfig, NodeConfig, TabTransformerConfig
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

def run_model_sweep_experiment(data_path):
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

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create training and test dataframes
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    print("\nTrain Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # Configure data
    data_config = DataConfig(
        target=['target'],
        continuous_cols=features,
        categorical_cols=[],
    )

    trainer_config = TrainerConfig(
        batch_size=128,
        max_epochs=100,
        early_stopping="valid_loss",
        early_stopping_patience=20,
        auto_lr_find=True,
        accelerator="cpu"
    )

    optimizer_config = OptimizerConfig()

    # Configure head
    head_config = LinearHeadConfig(
        layers="",
        dropout=0.1,
        initialization="kaiming",
    ).__dict__

    # Run model sweep with custom model list
    model_list = [
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
        )
    ]

    sweep_df, best_model = model_sweep(
        task="classification",
        train=train_df,
        test=test_df,
        data_config=data_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        model_list=model_list,
        metrics=["accuracy", "auroc"],
        metrics_params=[{}, {}],
        metrics_prob_input=[False, True],
        rank_metric=("auroc", "higher_is_better"),
        progress_bar=True,
        verbose=True
    )

    # Save sweep results
    sweep_df.to_csv('results/model_sweep_results.csv', index=False)
    
    print("\nModel Sweep Results:")
    print(sweep_df[['model', '# Params', 'test_accuracy', 'test_auroc', 'time_taken_per_epoch']])

    # 选择性能最好的模型进行参数调优
    top_models = sweep_df[sweep_df['model'] == 'AutoIntModel'].copy()
    
    # 为AutoInt模型定义搜索空间
    search_spaces = [{
        "optimizer_config__optimizer": ["Adam", "SGD"],
        "model_config__attn_embed_dim": [32, 64],
        "model_config__num_heads": [2, 4, 8],
        "model_config__num_attn_blocks": [2, 3, 4],
        "model_config__attn_dropouts": [0.0, 0.1, 0.2],
        "model_config__has_residuals": [True, False],
        "model_config__embedding_dim": [16, 32]
    }]

    model_configs = [AutoIntConfig(
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
    )]

    print(f"\nStarting Hyperparameter Tuning for AutoInt model...")
    
    tuner = TabularModelTuner(
        data_config=data_config,
        model_config=model_configs[0],  # 只使用一个模型配置
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    tuner_df = tuner.tune(
        train=train_df,
        validation=test_df,
        search_space=search_spaces[0],  # 只使用一个搜索空间
        strategy="random_search",
        n_trials=10,
        metric="auroc",
        mode="max",
        progress_bar=True,
        verbose=True
    )

    # Save tuning results
    tuner_df.trials_df.to_csv('results/hyperparameter_tuning_results.csv', index=False)
    
    print("\nTop 3 configurations by AUC:")
    print(tuner_df.trials_df.sort_values("auroc", ascending=False).head(3))
    
    print("\nTop 3 configurations by Accuracy:")
    print(tuner_df.trials_df.sort_values("accuracy", ascending=False).head(3))
    
    # Save the best model
    tuner_df.best_model.save_model("best_model", inference_only=True)
    
    return sweep_df, tuner_df.best_model 