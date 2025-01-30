# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Standard libraries
import os
import json
import copy
from pathlib import Path
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Union

# Data processing
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    ParameterSampler,
    ParameterGrid,
    BaseCrossValidator
)

# Deep learning
import torch
import torch.nn.functional as F
import joblib

# Progress tracking
from rich.progress import Progress

# PyTorch Tabular imports
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig
)
from pytorch_tabular.models import (
    TabNetModelConfig,
    DANetConfig,
    AutoIntConfig,
    TabTransformerConfig
)
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.tabular_model_tuner import TabularModelTuner
from pytorch_tabular.utils import OOMException, OutOfMemoryHandler, get_logger, suppress_lightning_logs

# Get logger
logger = get_logger(__name__)

def multi_metric_agg(list_of_dicts):
    """
    多指标聚合函数：将多折的指标结果聚合为一个字典
    
    Args:
        list_of_dicts: 形如 [{"loss":0.5,"accuracy":0.8}, {"loss":0.6,"accuracy":0.81}, ...]
    
    Returns:
        dict: 聚合后的结果，如 {"loss":0.55,"accuracy":0.805}
    """
    if not list_of_dicts:
        return {}
    # 取所有key
    keys = list_of_dicts[0].keys()
    agg_dict = {}
    for k in keys:
        vals = []
        for d in list_of_dicts:
            vals.append(d.get(k, np.inf))  # 如果OOM则可能是np.inf
        agg_dict[k] = np.mean(vals)
    return agg_dict

class MultiMetricTabularModelTuner(TabularModelTuner):
    """支持多指标的TabularModelTuner
    
    继承TabularModelTuner，改写在CV场景下如何处理多指标。
    可以指定primary_metric用于选最优，其它指标也会被记录。
    """
    def __init__(
        self,
        primary_metric: str = "loss",    # 用于最终选最优的指标
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.primary_metric = primary_metric

    def tune(
        self,
        train,
        search_space,
        metric,      # 这里必须是一个能返回dict的可调用：compute_metrics
        mode: str,
        validation=None,
        n_trials=None,
        cv=None,
        cv_kwargs={},
        return_best_model=True,
        verbose=False,
        progress_bar=True,
        random_state=42,
        ignore_oom=True,
        **kwargs,
    ):
        """支持多指标的超参数调优
        
        基本流程和原TabularModelTuner一致，但在cv模式下:
        1) cross_validate直接返回多折平均后的dict结果
        2) 把该dict中所有key都更新到params
        3) 用 self.primary_metric 来判断最优
        """
        assert mode in ["max", "min"], "mode must be one of ['max', 'min']"
        if self.suppress_lightning_logger:
            from pytorch_tabular.utils import suppress_lightning_logs
            suppress_lightning_logs()

        if cv is not None and validation is not None:
            warnings.warn("Both validation and cv are provided. We'll use CV and ignore validation.")
            validation = None

        # 如果只传了一个model_config和一个search_space，也要包成list统一处理
        if not isinstance(self.model_config, list):
            model_configs = [self.model_config]
        else:
            model_configs = self.model_config

        if not isinstance(search_space, list):
            search_space = [search_space]

        trials = []
        best_model = None
        best_score = float("inf") if mode == "min" else float("-inf")

        with Progress() as progress_bar_inst:
            # 可能有多个model_config
            for idx, (mconfig_i, sspace_i) in enumerate(zip(model_configs, search_space)):

                # 给params里加一个model字段，区分不同模型
                sspace_i = {"model": [f"{idx}-{mconfig_i.__class__.__name__}"], **sspace_i}

                # 根据搜索策略
                strategy = kwargs.pop("strategy", "random_search")
                if strategy == "grid_search":
                    iterator = list(ParameterGrid(sspace_i))
                    if n_trials is not None:
                        warnings.warn("n_trials is ignored for grid_search.")
                else:
                    # 默认random_search
                    if n_trials is None:
                        raise ValueError("Need n_trials for random_search")
                    iterator = list(ParameterSampler(sspace_i, n_iter=n_trials, random_state=random_state))

                # 如果要显示进度条
                if progress_bar:
                    iterator = progress_bar_inst.track(iterator, description=f"Model {idx} Searching")

                # datamodule只需要初始化一次
                datamodule = None

                for i, params in enumerate(iterator):
                    # 复制配置
                    trainer_config_t = copy.deepcopy(self.trainer_config)
                    optimizer_config_t = copy.deepcopy(self.optimizer_config)
                    model_config_t = copy.deepcopy(mconfig_i)

                    # 用_tuner的_update_configs函数写入config
                    optimizer_config_t, model_config_t = self._update_configs(
                        optimizer_config_t, model_config_t, params
                    )

                    # 初始化TabularModel
                    tabular_model_t = TabularModel(
                        data_config=self.data_config,
                        model_config=model_config_t,
                        optimizer_config=optimizer_config_t,
                        trainer_config=trainer_config_t,
                        **self.tabular_model_init_kwargs,
                    )

                    # 构建datamodule
                    if not datamodule:
                        prep_dl_kwargs, prep_model_kwargs, train_kwargs = tabular_model_t._split_kwargs(kwargs)
                        if "seed" not in prep_dl_kwargs:
                            prep_dl_kwargs["seed"] = random_state
                        datamodule = tabular_model_t.prepare_dataloader(
                            train=train,
                            validation=validation,
                            **prep_dl_kwargs
                        )
                        valid_df = validation if validation is not None else datamodule.validation_dataset.data
                    else:
                        prep_model_kwargs, train_kwargs = {}, {}

                    # ========================
                    # 进入CV流程
                    # ========================
                    if cv is not None:
                        cv_verbose = cv_kwargs.pop("verbose", False)
                        cv_kwargs.pop("handle_oom", None)
                        with OutOfMemoryHandler(handle_oom=True) as handler:
                            # cross_validate 直接返回平均后的dict
                            cv_dict, _ = tabular_model_t.cross_validate(
                                cv=cv,
                                train=train,
                                metric=metric,  # compute_metrics(y_true, y_pred) => dict
                                verbose=cv_verbose,
                                handle_oom=False,
                                **cv_kwargs
                            )
                        if handler.oom_triggered:
                            # OOM处理
                            if not ignore_oom:
                                raise OOMException("OOM happened!")
                            else:
                                # 用inf替代
                                cv_dict = {self.primary_metric: np.inf}
                                params.update({"model": f"{params['model']} (OOM)"})

                    # ========================
                    # 否则走 常规train+evaluate
                    # ========================
                    else:
                        model = tabular_model_t.prepare_model(
                            datamodule=datamodule,
                            **prep_model_kwargs
                        )
                        with OutOfMemoryHandler(handle_oom=True) as handler:
                            tabular_model_t.train(
                                model=model,
                                datamodule=datamodule,
                                handle_oom=False,
                                **train_kwargs
                            )
                        if handler.oom_triggered:
                            if not ignore_oom:
                                raise OOMException("OOM happened!")
                            else:
                                cv_dict = {self.primary_metric: np.inf}
                                params.update({"model": f"{params['model']} (OOM)"})
                        else:
                            # predict验证集 => 计算多指标
                            preds = tabular_model_t.predict(valid_df)
                            y_true = valid_df[tabular_model_t.config.target]
                            cv_dict = metric(y_true, preds)  # dict

                    # 把多指标都写到params
                    for k, v in cv_dict.items():
                        params[k] = v

                    # ============判断是否是更优模型==========
                    current_score = cv_dict.get(self.primary_metric, np.inf if mode == "min" else -np.inf)
                    if best_model is None:
                        best_model = copy.deepcopy(tabular_model_t)
                        best_score = current_score
                    else:
                        if mode == "min" and current_score < best_score:
                            best_model = copy.deepcopy(tabular_model_t)
                            best_score = current_score
                        elif mode == "max" and current_score > best_score:
                            best_model = copy.deepcopy(tabular_model_t)
                            best_score = current_score

                    params["trial_id"] = i
                    trials.append(params)

                    if verbose:
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in cv_dict.items()])
                        print(f"[Trial {i}] {params['model']} => {metrics_str}")

        # 全部结束后，汇总到DataFrame
        trials_df = pd.DataFrame(trials)
        trials = trials_df.pop("trial_id")

        # 找到最佳行
        if mode == "max":
            best_idx = trials_df[self.primary_metric].idxmax()
        else:
            best_idx = trials_df[self.primary_metric].idxmin()

        best_params = trials_df.iloc[best_idx].to_dict()
        best_score = best_params.pop(self.primary_metric)
        trials_df.insert(0, "trial_id", trials)

        if verbose:
            logger.info("Model Tuner Finished")
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_params.items() if k in cv_dict.keys()])
            logger.info(f"Best Model: {best_params['model']} - Metrics: {metrics_str}")

        if return_best_model and best_model is not None:
            # 把datamodule挂回去
            best_model.datamodule = datamodule
            return self.OUTPUT(trials_df, best_params, best_score, best_model)
        else:
            return self.OUTPUT(trials_df, best_params, best_score, None)

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_model_files(save_model_dir="save_model", max_trials_to_keep=5):
    """清理模型文件，只保留最近的几次试验"""
    if not os.path.exists(save_model_dir):
        return
    
    # 获取所有试验目录
    trial_dirs = []
    for trial_dir in os.listdir(save_model_dir):
        trial_path = os.path.join(save_model_dir, trial_dir)
        if os.path.isdir(trial_path):
            # 获取目录的修改时间
            mtime = os.path.getmtime(trial_path)
            trial_dirs.append((mtime, trial_path))
    
    # 按时间排序
    trial_dirs.sort(reverse=True)
    
    # 删除旧的试验目录
    if len(trial_dirs) > max_trials_to_keep:
        for _, trial_path in trial_dirs[max_trials_to_keep:]:
            import shutil
            shutil.rmtree(trial_path)
        print(f"✅ Cleaned up old trial models, keeping {max_trials_to_keep} most recent trials")

def clean_save_model_dir(save_model_dir="save_model"):
    """清空save_model目录"""
    if os.path.exists(save_model_dir):
        import shutil
        shutil.rmtree(save_model_dir)
        os.makedirs(save_model_dir)
        print(f"✅ Cleaned up {save_model_dir} directory")

def compute_metrics(y_true, y_pred):
    """使用PyTorch计算多个评估指标的函数
    
    Args:
        y_true: 真实标签
        y_pred: 预测结果DataFrame，包含probability和prediction列
    
    Returns:
        dict: 包含多个指标的字典
    """
    # 准备数据
    prob_cols = [col for col in y_pred.columns if 'probability' in col]
    
    # 检查概率列的数量
    if len(prob_cols) == 1:
        # 如果只有一个概率列，创建两个类的概率
        probs = y_pred[prob_cols[0]].values
        probs = np.column_stack([1 - probs, probs])
    else:
        # 如果有多个概率列，直接使用
        probs = y_pred[prob_cols].values
    
    # 确保形状正确
    if len(probs.shape) == 1:
        probs = probs.reshape(-1, 1)
        probs = np.column_stack([1 - probs, probs])
    
    # 确保y_true和probs的样本数量匹配
    assert len(y_true) == len(probs), f"Sample size mismatch: y_true({len(y_true)}) != probs({len(probs)})"
    
    # 转换为tensor
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    target_tensor = torch.tensor(y_true.values, dtype=torch.long)
    
    # 确保target_tensor是一维的
    if len(target_tensor.shape) > 1:
        target_tensor = target_tensor.squeeze()
    
    # 计算交叉熵损失
    loss = F.cross_entropy(probs_tensor, target_tensor).item()
    
    # 计算accuracy
    predictions = probs_tensor.argmax(dim=1)
    accuracy = (predictions == target_tensor).float().mean().item()
    
    # 计算AUROC
    # 使用正类的概率
    pos_probs = probs_tensor[:, 1]
    
    # 将标签转换为one-hot编码
    target_one_hot = F.one_hot(target_tensor, num_classes=2).float()
    
    # 计算TPR和FPR
    thresholds = torch.linspace(0, 1, steps=100)
    tpr = torch.zeros_like(thresholds)
    fpr = torch.zeros_like(thresholds)
    
    for i, threshold in enumerate(thresholds):
        y_pred = (pos_probs >= threshold).float()
        
        # True Positives
        tp = (y_pred * target_one_hot[:, 1]).sum()
        # False Positives
        fp = (y_pred * (1 - target_one_hot[:, 1])).sum()
        # True Negatives
        tn = ((1 - y_pred) * (1 - target_one_hot[:, 1])).sum()
        # False Negatives
        fn = ((1 - y_pred) * target_one_hot[:, 1]).sum()
        
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 按照FPR的顺序对TPR进行排序
    sorted_indices = torch.argsort(fpr)
    fpr = fpr[sorted_indices]
    tpr = tpr[sorted_indices]
    
    # 使用梯形法则计算AUC，注意顺序
    auroc = torch.trapz(tpr, fpr).item()
    
    # 确保AUROC在[0,1]范围内
    auroc = max(0, min(1, auroc))
    
    return {
        'loss': loss,          # 主要优化指标
        'auroc': auroc,        # 次要指标
        'accuracy': accuracy   # 次要指标
    }

def tune_selected_models(data_path):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # 清空save_model目录
    clean_save_model_dir()
    
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
        early_stopping_patience=20,
        checkpoints="valid_loss",  # 根据验证集loss选择最佳模型
        checkpoints_path="save_model",
        checkpoints_mode="min"  # loss越小越好
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
    
    # 准备完整的训练数据集
    train_df = X.copy()
    train_df['target'] = y
    
    # 按顺序调优每个模型
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    
    for model_idx, (model_config, search_space) in enumerate(zip(model_configs, search_spaces)):
        model_name = model_names[model_idx]
        print(f"\n{'='*50}")
        print(f"Tuning {model_name}...")
        print(f"{'='*50}")
        
        # 每个模型开始前清空save_model目录
        clean_save_model_dir()
        
        # 为每个模型设置随机种子
        set_seed(42)
        
        # 使用新的MultiMetricTabularModelTuner替代原来的TabularModelTuner
        tuner = MultiMetricTabularModelTuner(
            primary_metric="loss",  # 使用loss作为主要优化指标
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )

        tuner_output = tuner.tune(
            train=train_df,
            search_space=search_space,
            strategy="random_search",
            n_trials=500,  # 增加到500次尝试
            metric=compute_metrics,  # 使用自定义指标函数
            mode="min",    # 最小化损失
            cv=kfold,  # 使用10折交叉验证
            cv_kwargs={"verbose": True},  # 显示详细信息
            progress_bar=True,
            verbose=True,
            return_best_model=True
        )
        
        # 保存调优结果
        tuner_output.trials_df['model'] = model_name
        all_fold_results.append(tuner_output.trials_df)
        
        # 保存最佳模型后清理save_model目录
        if tuner_output.best_model is not None:
            # 保存模型前，清理旧的模型文件
            model_save_path = f"results/{model_name}_best"
            if os.path.exists(model_save_path):
                import shutil
                shutil.rmtree(model_save_path)
            # 保存新的最佳模型
            tuner_output.best_model.save_model(model_save_path)
            print(f"✅ Saved best model to {model_save_path}")
            # 清理save_model目录
            clean_save_model_dir()
            
            # 保存到最佳模型字典
            best_models[model_name] = tuner_output.best_model
        
        # 获取最佳结果
        best_trial = tuner_output.trials_df.sort_values("loss", ascending=True).iloc[0]
        print("\n最佳试验配置:")
        for col in best_trial.index:
            if col not in ['loss', 'auroc', 'accuracy', 'model', 'trial_id']:
                print(f"{col}: {best_trial[col]}")
        
        print(f"\n{model_name} 最佳交叉验证结果:")
        print(f"Loss: {best_trial['loss']:.4f}")
        print(f"AUROC: {best_trial['auroc']:.4f}")
        print(f"Accuracy: {best_trial['accuracy']:.4f}")
        
        # 更新历史最佳配置
        if model_name not in historical_best or best_trial['loss'] < historical_best[model_name]['performance'].get('loss', float('inf')):
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
                    'loss': float(best_trial['loss']),
                    'auroc': float(best_trial['auroc']),
                    'accuracy': float(best_trial['accuracy']),
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            print(f"\n🏆 找到 {model_name} 的新最佳配置!")
            print(f"Loss: {best_trial['loss']:.4f}")
            print(f"AUROC: {best_trial['auroc']:.4f}")
            print(f"Accuracy: {best_trial['accuracy']:.4f}")
        
        # 添加到汇总结果
        summary_results.append({
            "model_name": model_name,
            "best_loss": best_trial['loss'],
            "best_auroc": best_trial['auroc'],
            "best_accuracy": best_trial['accuracy'],
            "timestamp": historical_best[model_name]['performance']['timestamp']
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
    print("=" * 80)
    print("\n所有模型的最佳结果汇总:")
    for _, row in summary_df.iterrows():
        print(f"\n模型: {row['model_name']}")
        print(f"最佳 Loss: {row['best_loss']:.4f}")
        print(f"最佳 AUROC: {row['best_auroc']:.4f}")
        print(f"最佳 Accuracy: {row['best_accuracy']:.4f}")
        print(f"达到时间: {row['timestamp']}")
    print("=" * 80)
    
    # 返回所有结果
    return pd.concat(all_fold_results), summary_df, best_models, historical_best

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"
    tuner_df, summary_df, best_models, historical_best = tune_selected_models(data_path) 