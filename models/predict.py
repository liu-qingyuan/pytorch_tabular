import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pytorch_tabular import TabularModel
import joblib
import os
import json
from sklearn.metrics import roc_auc_score, accuracy_score

def verify_model_performance(model_name, predictions, test_df):
    """
    验证模型性能是否与历史记录一致
    
    Args:
        model_name: 模型名称
        predictions: 模型预测结果
        test_df: 测试数据集
    """
    # 加载历史最佳配置
    best_configs_file = 'results/best_configs.json'
    if not os.path.exists(best_configs_file):
        print(f"❌ Historical best configurations file not found at {best_configs_file}")
        return
    
    with open(best_configs_file, 'r') as f:
        historical_best = json.load(f)
    
    if model_name not in historical_best:
        print(f"❌ No historical record found for {model_name}")
        return
    
    # 获取历史性能
    hist_auc = historical_best[model_name]['performance']['auc']
    hist_acc = historical_best[model_name]['performance']['accuracy']
    
    # 计算当前性能
    # 找到概率列
    prob_cols = [col for col in predictions.columns if 'probability' in col]
    if not prob_cols:
        print(f"❌ No probability column found in predictions. Available columns: {predictions.columns.tolist()}")
        return
    prob_col = prob_cols[0]
    
    # 找到预测列（可能是'prediction'或'target_prediction'）
    pred_cols = [col for col in predictions.columns if 'prediction' in col and 'probability' not in col]
    if not pred_cols:
        print(f"❌ No prediction column found in predictions. Available columns: {predictions.columns.tolist()}")
        # 如果没有预测列，根据概率生成预测
        predictions['prediction'] = (predictions[prob_col] > 0.5).astype(int)
        pred_col = 'prediction'
    else:
        pred_col = pred_cols[0]
    
    current_auc = roc_auc_score(test_df['target'], predictions[prob_col])
    current_acc = accuracy_score(test_df['target'], predictions[pred_col])
    
    print(f"\n📊 Performance Comparison for {model_name}:")
    print("=" * 50)
    print(f"Metric    Current    Historical    Difference")
    print("-" * 50)
    print(f"AUC       {current_auc:.4f}    {hist_auc:.4f}        {current_auc - hist_auc:+.4f}")
    print(f"Accuracy  {current_acc:.4f}    {hist_acc:.4f}        {current_acc - hist_acc:+.4f}")
    print("=" * 50)

def load_model_and_predict(data_path, model_name, verify_performance=True):
    """
    加载模型并进行预测
    
    Args:
        data_path: 需要预测的数据文件路径
        model_name: 模型名称 (DANetModel, TabTransformerModel, AutoIntModel, TabNetModel)
        verify_performance: 是否验证模型性能
    
    Returns:
        predictions: 预测结果DataFrame
    """
    try:
        # 1. 加载模型（使用正确的模型目录路径）
        model_path = f"results/best_{model_name}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        model = TabularModel.load_model(model_path)
        print(f"\n📊 Model config for {model_name}:")
        print(f"Task: {model.config.task}")
        print(f"Metrics: {model.config.metrics}")
        
        # 2. 加载数据
        df = pd.read_csv(data_path)
        
        # 3. 从模型配置中获取特征列
        continuous_features = model.config.continuous_cols
        categorical_features = model.config.categorical_cols
        all_features = continuous_features + categorical_features
        
        print(f"\n特征列验证:")
        print(f"连续特征 ({len(continuous_features)}): {continuous_features[:5]}...")
        print(f"类别特征 ({len(categorical_features)}): {categorical_features}")
        
        # 验证所有特征都存在
        missing_features = [f for f in all_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        if 'target' in df.columns:
            print("\n目标变量分布:")
            print(df['target'].value_counts(normalize=True))
        
        # 4. 准备输入数据
        X = df[all_features].copy()
        
        # 5. 加载和应用标准化器
        scaler_path = "results/scaler.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        
        # 检查数据的基本统计信息
        print("\n标准化前统计:")
        print(X[continuous_features].describe().round(3).loc[['mean', 'std']].head())
        
        # 只对连续特征进行标准化
        X[continuous_features] = scaler.transform(X[continuous_features])
        
        print("\n标准化后统计:")
        print(X[continuous_features].describe().round(3).loc[['mean', 'std']].head())
        
        # 6. 进行预测
        print("\n开始预测...")
        predictions = model.predict(X)
        
        print("\n预测结果列:", predictions.columns.tolist())
        
        # 7. 验证模型性能（如果需要）
        if verify_performance and 'target' in df.columns:
            verify_model_performance(model_name, predictions, df)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error with {model_name}:")
        import traceback
        traceback.print_exc()
        return None

def ensemble_predictions(predictions_list, model_names):
    """
    集成多个模型的预测结果
    
    Args:
        predictions_list: 每个模型的预测结果列表
        model_names: 对应的模型名称列表
    
    Returns:
        ensemble_df: 集成后的预测结果
    """
    if not predictions_list:
        print("❌ No valid predictions to ensemble")
        return None
    
    print("\n🔄 Creating ensemble predictions...")
    
    # 收集所有模型的预测概率
    all_probs = []
    for preds in predictions_list:
        prob_col = [col for col in preds.columns if 'probability' in col][0]
        all_probs.append(preds[prob_col].values)
    
    # 计算平均概率
    avg_probs = np.mean(all_probs, axis=0)
    
    # 创建集成预测结果
    ensemble_df = pd.DataFrame({
        'ensemble_probability': avg_probs,
        'ensemble_prediction': (avg_probs > 0.5).astype(int)
    })
    
    # 添加每个模型的预测结果
    for model_name, preds in zip(model_names, predictions_list):
        prob_col = [col for col in preds.columns if 'probability' in col][0]
        pred_col = 'prediction' if 'prediction' in preds.columns else None
        
        if prob_col:
            ensemble_df[f'{model_name}_probability'] = preds[prob_col]
        if pred_col:
            ensemble_df[f'{model_name}_prediction'] = preds[pred_col]
    
    # 保存集成结果
    save_path = "results/ensemble_predictions.csv"
    ensemble_df.to_csv(save_path, index=False)
    print(f"✅ Ensemble predictions saved to {save_path}")
    
    # 打印集成结果统计
    print("\n📊 Ensemble Prediction Distribution:")
    print("\nProbability Distribution:")
    print(ensemble_df['ensemble_probability'].describe())
    print("\nPrediction Distribution:")
    print(ensemble_df['ensemble_prediction'].value_counts(normalize=True))
    
    return ensemble_df

if __name__ == "__main__":
    # 加载测试集
    test_df = pd.read_csv("results/test_set.csv")
    print("\n📊 Verifying saved models performance...")
    
    # 定义模型列表
    model_names = ["DANetModel", "TabTransformerModel", "AutoIntModel", "TabNetModel"]
    
    # 对每个模型进行性能验证
    for model_name in model_names:
        try:
            predictions = load_model_and_predict(
                data_path="results/test_set.csv",  # 使用保存的测试集
                model_name=model_name,
                verify_performance=True  # 开启性能验证
            )
        except Exception as e:
            print(f"❌ Failed to verify {model_name}: {str(e)}")
            continue 