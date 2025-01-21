from models.tabnet_model_sweep import run_model_sweep_experiment

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"  # Updated data path
    sweep_df, best_model = run_model_sweep_experiment(data_path)
    
    print("\nBest Model by Accuracy:")
    best_acc_model = sweep_df.iloc[sweep_df['test_accuracy'].argmax()]
    print(f"Model: {best_acc_model['model']}")
    print(f"Model Parameters: {best_acc_model['# Params']}")
    print(f"Test Accuracy: {best_acc_model['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best_acc_model['test_f1_score']:.4f}")
    print(f"Test AUROC: {best_acc_model['test_auroc']:.4f}")
    print(f"Time per epoch: {best_acc_model['time_taken_per_epoch']:.4f} seconds")
    
    print("\nBest Model by AUC:")
    best_auc_model = sweep_df.iloc[sweep_df['test_auroc'].argmax()]
    print(f"Model: {best_auc_model['model']}")
    print(f"Model Parameters: {best_auc_model['# Params']}")
    print(f"Test Accuracy: {best_auc_model['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best_auc_model['test_f1_score']:.4f}")
    print(f"Test AUROC: {best_auc_model['test_auroc']:.4f}")
    print(f"Time per epoch: {best_auc_model['time_taken_per_epoch']:.4f} seconds")
    
    print("\nAll Models Sorted by AUC:")
    print(sweep_df.sort_values('test_auroc', ascending=False)[['model', 'test_auroc', 'test_accuracy', 'time_taken_per_epoch']]) 