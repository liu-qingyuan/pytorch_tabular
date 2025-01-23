import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
from torchmetrics import AUROC, Accuracy  # 修正：直接从 torchmetrics 导入
import joblib
import os
from sklearn.model_selection import StratifiedKFold
import torch
import json

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_metrics_from_result(metrics_result):
    """从评估结果中提取指标"""
    # 如果是列表，取第一个元素
    if isinstance(metrics_result, list) and len(metrics_result) > 0:
        metrics_result = metrics_result[0]
    
    # 如果已经是字典格式，直接返回需要的指标
    if isinstance(metrics_result, dict):
        return {
            'test_accuracy': metrics_result['test_accuracy'],
            'test_auroc': metrics_result['test_auroc']
        }
    
    # 否则解析表格格式
    lines = str(metrics_result).split('\n')
    metrics_dict = {}
    for line in lines:
        if '│' not in line:  # Skip lines without table separator
            continue
        columns = line.split('│')
        if len(columns) < 3:  # Skip invalid lines
            continue
        metric_name = columns[1].strip()
        if metric_name == 'test_accuracy':
            metrics_dict['test_accuracy'] = float(columns[2].strip())
        elif metric_name == 'test_auroc':
            metrics_dict['test_auroc'] = float(columns[2].strip())
    
    if not metrics_dict:
        raise ValueError(f"无法从结果中解析出指标:\n{metrics_result}")
    
    return metrics_dict

def load_model_and_predict(model_name, data_path, fold="ensemble"):
    """加载模型进行预测
    
    Args:
        model_name: 模型名称 (DANetModel/TabTransformerModel/AutoIntModel/TabNetModel)
        data_path: 数据路径
        fold: 指定fold编号(1-10) 或 "ensemble"（集成预测）
        
    Returns:
        tuple: (predictions_df, fold_metrics_df) 如果是ensemble模式
        DataFrame: predictions_df 如果是单fold模式
    """
    print("\n1. 加载数据和预处理配置...")
    # 1. 设置基础随机种子
    set_seed(42)
    
    # 2. 加载预处理配置
    scaler = joblib.load("results/scaler.pkl")
    print("加载的scaler均值样例:", scaler.mean_[:3])
    print("加载的scaler方差样例:", scaler.var_[:3])
    
    # 3. 加载原始数据（未标准化）
    raw_df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
    
    # 4. 获取特征列（保持与训练时相同的顺序）
    features = [c for c in raw_df.columns if c.startswith("Feature")]  # 不排序，保持原始顺序
    if not features:
        raise ValueError(f"数据中没有找到特征列（以'Feature'开头的列）")
    print(f"加载数据: {len(raw_df)} 行, {len(features)} 个特征")
    print("特征列:", features[:5], "...")  # 打印前几个特征列名，用于验证
    
    # 5. 按原始顺序提取特征并标准化（与训练时一致）
    X = raw_df[features].copy()
    y = raw_df["Label"].copy() if "Label" in raw_df.columns else None
    
    # 6. 使用保存的scaler进行标准化（在划分fold之前）
    try:
        X = pd.DataFrame(scaler.transform(X), columns=features)
        print("✓ 完成数据标准化")
    except ValueError as e:
        print("❌ 标准化失败！检查特征列顺序...")
        print("期望的特征顺序:", features)
        print("Scaler的特征数量:", len(scaler.mean_))
        raise
    
    # 7. 使用相同的交叉验证划分（在标准化后的数据上划分）
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # 8. 获取所有fold的预测结果
    all_fold_preds = np.zeros(len(X))
    fold_weights = np.zeros(len(X))
    
    def get_probabilities(pred_df):
        """从预测结果中获取概率值"""
        if isinstance(pred_df, pd.DataFrame):
            if 'target_1_probability' in pred_df.columns:
                return pred_df['target_1_probability'].values
            elif pred_df.shape[1] == 2 and all(c.endswith('probability') for c in pred_df.columns):
                return pred_df.iloc[:, 1].values
            else:
                raise ValueError(f"无法解析预测结果，列名: {pred_df.columns.tolist()}")
        else:
            raise ValueError("预测结果必须是DataFrame格式")
    
    print("\n2. 开始模型预测...")
    if fold == "ensemble":
        # 对每个fold进行预测
        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            try:
                # 为每个fold设置不同的随机种子，与训练时一致
                set_seed(42 + fold_idx)
                
                # 加载当前fold的模型
                model_path = f"results/{model_name}_fold{fold_idx}"
                print(f"\n加载模型: {model_path}")
                
                model = TabularModel.load_model(model_path)
                print("模型类别:", type(model.model))
                
                # 准备验证数据（与训练时保持一致）
                X_val = X.iloc[val_idx]
                val_df = X_val.copy()
                if y is not None:
                    val_df['target'] = y.iloc[val_idx]
                
                # 对验证集进行预测
                pred = model.predict(val_df)
                probs = get_probabilities(pred)
                
                # 将预测结果放入对应位置
                all_fold_preds[val_idx] = probs
                fold_weights[val_idx] = 1
                
                # 计算当前fold的性能指标
                if y is not None:
                    # 使用模型自带的评估方法
                    metrics_result = model.evaluate(val_df)
                    metrics = get_metrics_from_result(metrics_result)
                    fold_metrics.append({
                        'fold': fold_idx,
                        'samples': len(val_idx),
                        'auc': metrics['test_auroc'],
                        'acc': metrics['test_accuracy']
                    })
                
                print(f"✅ Fold {fold_idx} 预测完成")
                # print(f"验证集大小: {len(val_idx)}")
                # print(f"预测值范围: [{probs.min():.4f}, {probs.max():.4f}]")
                # print(f"预测值均值: {probs.mean():.4f}")
                if y is not None:
                    print(f"AUC: {metrics['test_auroc']:.4f}, ACC: {metrics['test_accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ Fold {fold_idx} 预测失败: {str(e)}")
                raise
        
        # 检查是否所有样本都有预测值
        if not np.all(fold_weights > 0):
            raise ValueError("某些样本没有得到预测值，请检查交叉验证划分")
        
        print("\n3. 预测完成")
        
        if fold_metrics:
            print("\n📊 各Fold性能汇总:")
            fold_df = pd.DataFrame(fold_metrics)
            print(fold_df.to_string(index=False))
            print("\n平均性能:")
            print(f"Mean AUC: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
            print(f"Mean ACC: {fold_df['acc'].mean():.4f} ± {fold_df['acc'].std():.4f}")
        
        # 返回预测结果和fold性能指标
        predictions_df = pd.DataFrame({
            'probability': all_fold_preds,
            'prediction': (all_fold_preds > 0.5).astype(int)
        })
        
        if fold_metrics:
            fold_metrics_df = pd.DataFrame(fold_metrics)
            return predictions_df, fold_metrics_df
        return predictions_df
    else:
        # 单个fold的预测
        fold = int(fold)
        if not 1 <= fold <= 10:
            raise ValueError("fold必须在1-10之间")
        
        # 设置对应fold的随机种子
        set_seed(42 + fold)
        
        # 获取指定fold的验证集索引
        for fold_idx, (_, val_idx) in enumerate(kfold.split(X, y), 1):
            if fold_idx == fold:
                break
        else:
            raise ValueError(f"找不到fold {fold}")
        
        # 加载模型并预测
        try:
            model_path = f"results/{model_name}_fold{fold}"
            print(f"\n加载模型: {model_path}")
            
            model = TabularModel.load_model(model_path)
            print("模型类别:", type(model.model))
            
            # 只对该fold的验证集进行预测（已经标准化过）
            X_val = X.iloc[val_idx]
            pred = model.predict(X_val)
            probs = get_probabilities(pred)
            
            print("\n3. 预测完成")
            
            # 计算性能指标
            if y is not None:
                # 准备验证数据
                val_df = pd.DataFrame({
                    'target': y.iloc[val_idx],
                    **{f: X_val[f] for f in features}
                })
                # 使用模型自带的评估方法
                metrics_result = model.evaluate(val_df)
                metrics = get_metrics_from_result(metrics_result)
                print(f"AUC: {metrics['test_auroc']:.4f}, ACC: {metrics['test_accuracy']:.4f}")
            
            return pd.DataFrame({
                'probability': probs,
                'prediction': (probs > 0.5).astype(int)
            })
            
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}")
            raise

def calculate_metrics(y_true, y_pred_proba, model):
    """使用与训练时相同的指标计算方式"""
    try:
        # 准备数据（与训练时保持一致）
        val_df = pd.DataFrame()
        val_df['target'] = y_true  # 使用与训练时相同的目标列名
        
        # 使用0.0而不是0来确保浮点数类型
        for i, feat in enumerate(model.config.continuous_cols, 1):
            val_df[feat] = np.zeros(len(y_true), dtype=np.float32)  # 显式指定float32类型
        
        # 使用模型的evaluate方法（与训练时相同的评估方式）
        metrics_result = model.evaluate(val_df)
        metrics = get_metrics_from_result(metrics_result)
        
        # 从metrics字典中获取结果
        return metrics['test_auroc'], metrics['test_accuracy'], metrics.get('test_loss', None)
    except Exception as e:
        print(f"❌ 指标计算失败: {str(e)}")
        print(f"y_true 类型: {type(y_true)}, 形状: {getattr(y_true, 'shape', 'unknown')}")
        print(f"y_pred_proba 类型: {type(y_pred_proba)}, 形状: {getattr(y_pred_proba, 'shape', 'unknown')}")
        return None, None, None

def verify_performance(model_name, data_path="data/AI4healthcare.xlsx"):
    """验证模型性能一致性"""
    # 加载原始数据
    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
    y_true = df["Label"].copy()
    
    # 获取预测结果
    preds = load_model_and_predict(model_name, data_path)
    
    # 打印预测分布信息
    print("\n预测分布:")
    print(preds['probability'].describe())
    print("\n标签分布:")
    print(y_true.value_counts(normalize=True))
    
    # 加载第一个fold的模型用于评估
    try:
        model = TabularModel.load_model(f"results/{model_name}_fold1")
        # 使用与训练时相同的指标计算方式
        auc, acc, loss = calculate_metrics(y_true, preds['probability'], model)
        if auc is None:
            print("❌ 性能验证失败：指标计算错误")
            return
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return
    
    # 加载历史最佳结果
    with open('results/best_configs.json', 'r') as f:
        historical_best = json.load(f)
    hist_metrics = historical_best[model_name]['performance']
    hist_loss = hist_metrics['loss']
    hist_auc = hist_metrics['auc']
    hist_acc = hist_metrics['accuracy']
    
    print(f"\n🔍 {model_name} 性能验证结果:")
    print(f"当前 Loss: {loss:.4f} | 历史最佳 Loss: {hist_loss:.4f}")
    print(f"当前 AUC: {auc:.4f} | 历史最佳 AUC: {hist_auc:.4f}")
    print(f"当前 ACC: {acc:.4f} | 历史最佳 ACC: {hist_acc:.4f}")
    
    # 检查性能是否接近历史最佳
    loss_diff = abs(loss - hist_loss) if loss is not None else float('inf')
    if loss_diff > 0.01:  # 允许1%的差异
        print("\n⚠️ 警告: Loss与历史最佳相差较大!")
        print(f"Loss差异: {loss_diff:.4f}")
        print("可能的原因:")
        print("1. 数据预处理不一致")
        print("2. 交叉验证划分不匹配")
        print("3. 模型加载错误")
        print("4. 特征顺序不一致")
        print("5. 指标计算方式不一致")
    else:
        print("\n✅ 性能验证通过: Loss与历史最佳接近")
    
    # 保存详细的验证结果
    result_df = pd.DataFrame({
        'true_label': y_true,
        'probability': preds['probability'],
        'prediction': preds['prediction'],
        'fold_assignment': np.zeros(len(y_true))  # 记录每个样本属于哪个fold
    })
    
    # 记录fold分配
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(kfold.split(df[sorted([c for c in df.columns if c.startswith("Feature")])], y_true), 1):
        result_df.loc[val_idx, 'fold_assignment'] = fold_idx
    
    # 添加验证信息
    result_df['model_name'] = model_name
    result_df['validation_time'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    result_df['historical_best_loss'] = hist_loss
    result_df['historical_best_auc'] = hist_auc
    result_df['historical_best_acc'] = hist_acc
    result_df['current_loss'] = loss
    result_df['current_auc'] = auc
    result_df['current_acc'] = acc
    
    # 保存结果
    result_df.to_csv(f"results/{model_name}_predictions.csv", index=False)
    print(f"\n✅ 保存预测结果到 results/{model_name}_predictions.csv")
    
    # 分析每个fold的性能
    print("\n📊 各Fold性能分析:")
    fold_metrics = []
    for fold in range(1, 11):
        fold_mask = result_df['fold_assignment'] == fold
        if fold_mask.any():
            try:
                # 加载对应fold的模型
                fold_model = TabularModel.load_model(f"results/{model_name}_fold{fold}")
                fold_auc, fold_acc, fold_loss = calculate_metrics(
                    result_df.loc[fold_mask, 'true_label'],
                    result_df.loc[fold_mask, 'probability'],
                    fold_model
                )
                if fold_auc is not None:
                    fold_metrics.append({
                        'fold': fold,
                        'samples': fold_mask.sum(),
                        'loss': fold_loss,
                        'auc': fold_auc,
                        'acc': fold_acc
                    })
            except Exception as e:
                print(f"❌ Fold {fold} 评估失败: {str(e)}")
    
    if fold_metrics:
        fold_df = pd.DataFrame(fold_metrics)
        print(fold_df)
        
        # 检查fold间的性能差异
        loss_std = fold_df['loss'].std()
        if loss_std > 0.05:  # 如果fold间loss标准差超过0.05
            print("\n⚠️ 警告: Fold间性能差异较大!")
            print(f"Loss标准差: {loss_std:.4f}")
    
    # 绘制预测分布
    plot_predictions(preds, result_df['fold_assignment'], save_path=f"results/{model_name}_prediction_distribution.png")

def batch_predict(model_name, data_path, batch_size=1024):
    """分批预测避免内存不足"""
    df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
    total_samples = len(df)
    
    print(f"\n开始批量预测 {total_samples} 个样本...")
    print(f"批次大小: {batch_size}")
    
    all_preds = []
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_df = df.iloc[start_idx:end_idx]
        
        # 保存临时批次数据
        temp_batch_path = "temp_batch.csv"
        batch_df.to_csv(temp_batch_path, index=False)
        
        # 预测当前批次
        try:
            batch_pred = load_model_and_predict(model_name, temp_batch_path)
            all_preds.append(batch_pred)
            print(f"✓ 完成批次 {start_idx//batch_size + 1}/{(total_samples-1)//batch_size + 1}")
        except Exception as e:
            print(f"❌ 批次 {start_idx//batch_size + 1} 预测失败: {str(e)}")
        
        # 清理临时文件
        if os.path.exists(temp_batch_path):
            os.remove(temp_batch_path)
    
    # 合并所有批次结果
    final_preds = pd.concat(all_preds, ignore_index=True)
    print(f"\n✅ 批量预测完成")
    return final_preds

class ModelMonitor:
    def __init__(self, log_file="results/model_monitor.csv"):
        self.log_file = log_file
        self.current_run = {}  # 存储当前运行的所有模型性能
        self.performance_log = []
        
        # 加载已有日志（如果需要）
        if os.path.exists(log_file):
            self.performance_log = pd.read_csv(log_file).to_dict('records')
    
    def track_performance(self, model_name, fold_metrics, data_info=""):
        """记录模型性能
        
        Args:
            model_name: 模型名称
            fold_metrics: 包含fold性能的DataFrame，需要包含'auc'和'acc'列
            data_info: 数据描述信息
        """
        try:
            # 记录当前模型的性能
            self.current_run[model_name] = {
                "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "data_info": data_info,
                "metrics": fold_metrics.to_dict('records')
            }
            
            # 检查性能变化
            self._check_performance_drift(model_name, fold_metrics)
                
        except Exception as e:
            print(f"❌ 性能跟踪失败 ({model_name}): {str(e)}")
    
    def _save_log(self):
        """保存性能日志"""
        # 将当前运行的所有结果保存到日志
        records = []
        for model_name, data in self.current_run.items():
            for fold_metric in data['metrics']:
                record = {
                    "timestamp": data['timestamp'],
                    "model": model_name,
                    "fold": fold_metric['fold'],
                    "data_info": data['data_info'],
                    "auc": fold_metric['auc'],
                    "accuracy": fold_metric['acc'],
                    "samples": fold_metric['samples']
                }
                records.append(record)
        
        # 保存到文件
        pd.DataFrame(records).to_csv(self.log_file, index=False)
    
    def _check_performance_drift(self, model_name, fold_metrics):
        """检查性能漂移"""
        mean_auc = fold_metrics['auc'].mean()
        std_auc = fold_metrics['auc'].std()
        
        # 检查每个fold的性能是否异常
        for _, row in fold_metrics.iterrows():
            if abs(row['auc'] - mean_auc) > 2 * std_auc:
                print(f"\n⚠️ {model_name} Fold {row['fold']} 性能异常:")
                print(f"AUC: {row['auc']:.4f}")
                print(f"平均 AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    def generate_report(self):
        """生成性能报告"""
        if not self.current_run:
            print("没有可用的性能记录")
            return
        
        # 将所有当前运行的结果整理成DataFrame
        records = []
        for model_name, data in self.current_run.items():
            for fold_metric in data['metrics']:
                record = {
                    "timestamp": data['timestamp'],
                    "model": model_name,
                    "fold": fold_metric['fold'],
                    "data_info": data['data_info'],
                    "auc": fold_metric['auc'],
                    "accuracy": fold_metric['acc'],
                    "samples": fold_metric['samples']
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        
        # 按模型分组统计
        model_stats = df.groupby('model').agg({
            'auc': ['mean', 'std', 'min', 'max'],
            'accuracy': ['mean', 'std', 'min', 'max'],
            'samples': 'sum'
        }).round(4)
        
        print("\n📊 模型性能统计报告")
        print("=" * 80)
        print(model_stats)
        
        # 保存结果
        self._save_log()

def plot_predictions(preds, fold_assignment, save_path="results/prediction_distribution.png"):
    """可视化预测分布"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(preds['probability'], bins=50, alpha=0.7)
    plt.title("预测概率分布")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats = preds['probability'].describe()
    info = f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}\n"
    info += f"Min: {stats['min']:.3f}\nMax: {stats['max']:.3f}"
    plt.text(0.95, 0.95, info,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加fold信息
    plt.bar(range(len(preds['probability'])), fold_assignment, color='red', alpha=0.5)
    
    plt.savefig(save_path)
    plt.close()
    print(f"✅ 保存预测分布图到: {save_path}")

if __name__ == "__main__":
    # 示例：验证所有模型性能
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    
    print("\n开始模型验证...")
    monitor = ModelMonitor()
    
    for model_name in model_names:
        try:
            # 获取预测结果和fold性能
            preds = load_model_and_predict(model_name, "data/AI4healthcare.xlsx")
            
            # 记录性能到监控器
            if isinstance(preds, tuple) and len(preds) == 2:
                predictions, fold_metrics = preds
                monitor.track_performance(model_name, fold_metrics, "全量数据验证")
            
        except Exception as e:
            print(f"\n❌ {model_name} 验证失败: {str(e)}")
    
    # 生成监控报告
    monitor.generate_report() 