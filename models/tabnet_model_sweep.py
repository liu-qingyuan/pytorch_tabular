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
            head_config=head_config
        ),
        CategoryEmbeddingModelConfig(
            task="classification",
            layers="256-128-64",
            head="LinearHead",
            head_config=head_config
        ),
        DANetConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config
        ),
        FTTransformerConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config
        ),
        GANDALFConfig(
            task="classification",
            gflu_stages=6,
            head="LinearHead",
            head_config=head_config
        ),
        GatedAdditiveTreeEnsembleConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config
        ),
        NodeConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config
        ),
        TabNetModelConfig(
            task="classification",
            n_d=32,
            n_a=32,
            n_steps=3,
            gamma=1.5,
            n_independent=1,
            n_shared=2,
            head="LinearHead",
            head_config=head_config
        ),
        TabTransformerConfig(
            task="classification",
            head="LinearHead",
            head_config=head_config
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
        common_model_args=dict(head="LinearHead", head_config=head_config),
        metrics=["accuracy", "f1_score"],
        metrics_params=[{}, {"average": "macro"}],
        metrics_prob_input=[False, True],
        rank_metric=("accuracy", "higher_is_better"),
        progress_bar=True,
        verbose=True,
        ignore_oom=True
    )

    # Save sweep results
    sweep_df.to_csv('results/model_sweep_results.csv', index=False)
    
    print("\nModel Sweep Results:")
    print(sweep_df[['model', '# Params', 'test_accuracy', 'test_f1_score', 'time_taken_per_epoch']])

    # Calculate metrics using scikit-learn for all models
    print("\nModel Performance with scikit-learn metrics:")
    results_list = []
    
    # Create a dictionary to store AUC values
    auc_values = {}
    
    for model_name in sweep_df['model'].unique():
        if "(OOM)" in model_name:  # Skip OOM models
            continue
            
        # Get the model configuration based on model name
        model_config = None
        if model_name == "CategoryEmbeddingModel":
            model_config = CategoryEmbeddingModelConfig(
                task="classification",
                layers="256-128-64",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "GANDALFModel":
            model_config = GANDALFConfig(
                task="classification",
                gflu_stages=6,
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "TabNetModel":
            model_config = TabNetModelConfig(
                task="classification",
                n_d=32,
                n_a=32,
                n_steps=3,
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "FTTransformerModel":
            model_config = FTTransformerConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "AutoIntModel":
            model_config = AutoIntConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "DANetModel":
            model_config = DANetConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "GatedAdditiveTreeEnsembleModel":
            model_config = GatedAdditiveTreeEnsembleConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "NodeModel":
            model_config = NodeConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
        elif model_name == "TabTransformerModel":
            model_config = TabTransformerConfig(
                task="classification",
                head="LinearHead",
                head_config=head_config,
            )
            
        if model_config is None:
            print(f"Skipping unknown model: {model_name}")
            continue
        
        # Train model
        model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        
        try:
            model.fit(train=train_df, validation=test_df)
            
            # Get predictions
            pred_df = model.predict(test_df)
            y_pred_proba = pred_df['target_1_probability'].values
            y_pred = pred_df['target_prediction'].values
            
            # Calculate metrics
            sk_accuracy = accuracy_score(y_test, y_pred)
            sk_f1 = f1_score(y_test, y_pred, average='macro')
            sk_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store AUC value
            auc_values[model_name] = sk_auc
            
            results_list.append({
                'Model': model_name,
                'Accuracy': sk_accuracy,
                'F1-Score': sk_f1,
                'AUC-ROC': sk_auc
            })
            
            print(f"\n{model_name}:")
            print(f"Accuracy: {sk_accuracy:.4f}")
            print(f"F1-Score: {sk_f1:.4f}")
            print(f"AUC-ROC: {sk_auc:.4f}")
            
        except Exception as e:
            print(f"\nError training {model_name}: {str(e)}")
            continue
    
    # Add AUC values to sweep_df
    sweep_df['test_auroc'] = sweep_df['model'].map(auc_values)
    
    # Create and save detailed results
    detailed_results = pd.DataFrame(results_list)
    detailed_results.to_csv('results/detailed_model_metrics.csv', index=False)
    print("\nDetailed results saved to 'results/detailed_model_metrics.csv'")
    
    # Print final results including AUC
    print("\nFinal Model Sweep Results:")
    print(sweep_df[['model', '# Params', 'test_accuracy', 'test_f1_score', 'test_auroc', 'time_taken_per_epoch']])
    
    return sweep_df, best_model 