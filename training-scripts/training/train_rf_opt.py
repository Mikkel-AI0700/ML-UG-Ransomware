import optuna as opt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def _optuna_training (
    opt_trial,
    tr_x: pd.DataFrame,
    tr_y: np.ndarray,
    ts_x: pd.DataFrame,
    ts_y: np.ndarray,
    strat_kfold: StratifiedKFold
):
    current_iteration_precision_scores = []
    
    rf_hyperparameter_grid = {
        "n_estimators": opt_trial.suggest_int(name="RF_N_ESTIMATORS", low=100, high=1000, step=100),
        "criterion": opt_trial.suggest_categorical(name="RF_CRITERION", choices=["gini", "entropy"]),
        "max_depth": opt_trial.suggest_int(name="RF_MAX_DEPTH", low=10, high=70, step=10),
        "min_samples_split": opt_trial.suggest_int(name="RF_MIN_SAMPLES_SPLIT", low=20, high=80, step=20),
        "min_samples_leaf": opt_trial.suggest_int(name="RF_MIN_SAMPLES_SPLIT", low=20, high=80, step=20),
        "min_impurity_decrease": opt_trial.suggest_float(name="RF_MIN_IMPURITY_DECREASE", low=0.1, high=1.0, log=True),
        "n_jobs": 10,
        "random_state": 42
    }
    
    for tr_idx, val_idx in strat_kfold.split(tr_x, tr_y):
        iterated_random_forest = RandomForestClassifier(**rf_hyperparameter_grid)
        tr_x_subset, tr_y_subset = tr_x.iloc[tr_idx], tr_y[tr_idx]
        ts_x_subset, ts_y_subset = tr_x.iloc[val_idx], tr_y[val_idx]
        
        iterated_random_forest.fit(tr_x_subset, tr_y_subset)
        predictions = iterated_random_forest.predict(ts_x_subset)
        
        current_iteration_precision_scores.append(precision_score(ts_y_subset, predictions, average="weighted", zero_division=0))
        
    return np.mean(np.asarray(current_iteration_precision_scores))

def main ():
    tr_x = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/tr_x.csv")
    tr_y = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/tr_y.csv")
    ts_x = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/ts_x.csv")
    ts_y = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/ts_y.csv")
    datasets_list = [tr_x, tr_y, ts_x, ts_y]
    
    strat_kfold = StratifiedKFold(shuffle=True, random_state=42)
    
    opt.logging.set_verbosity(opt.logging.INFO)
    rf_study = opt.create_study(study_name="RF_HYPERPARAMETER_TUNING", direction="maximize")
    rf_study.optimize(
        lambda trial: _optuna_training(
            trial,
            tr_x,
            tr_y.to_numpy().ravel(),
            ts_x,
            ts_y.to_numpy().ravel(),
            strat_kfold
        ),
        n_trials=2000,
        n_jobs=75
    )
    
    print(f"Best trial: {rf_study.best_trial} \nBest parameters: {rf_study.best_params}")
    
main()
