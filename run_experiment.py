from models.tabnet_model import run_tabnet_experiment

if __name__ == "__main__":
    data_path = "data/AI4healthcare.xlsx"
    fold_results, final_results = run_tabnet_experiment(data_path) 