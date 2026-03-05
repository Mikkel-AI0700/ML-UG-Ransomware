import optuna as opt
import optuna.trial
import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier

def _optuna_training (
    opt_trial: opt.Trial,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    scoring_dictionary: dict[str, Callable],
    strat_kfold: StratifiedKFold,
):
    precision_iteration_score = []
    
    hyperparameter_grid = {
        "criterion": opt_trial.suggest_categorical(name="DT_criterion", choices=["gini", "entropy"]),
        "max_depth": opt_trial.suggest_int(name="DT_max_depth", low=15, high=55, step=5),
        "min_samples_split": opt_trial.suggest_int(name="DT_min_samples_split", low=10, high=70, step=10),
        "min_samples_leaf": opt_trial.suggest_int(name="DT_min_samples_leaf", low=6, high=30, step=6),
        "random_state": 42
    }

    for index, (tr_idx, val_idx) in enumerate(strat_kfold.split(train_x, train_y)):
        temp_tree_instance = DecisionTreeClassifier(**hyperparameter_grid)
        tr_x_subset, tr_y_subset = train_x.iloc[tr_idx], train_y[tr_idx]
        val_x_subset, val_y_subset = train_x.iloc[val_idx], train_y[val_idx]
        
        temp_tree_instance.fit(tr_x_subset, tr_y_subset)
        predictions = temp_tree_instance.predict(val_x_subset)
        
        precision_iteration_score.append(precision_score(val_y_subset, predictions, average="weighted"))
        
    return np.mean(np.asarray(precision_iteration_score))

def main ():
    tr_x = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/tr_x.csv")
    tr_y = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/tr_y.csv").to_numpy()
    ts_x = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/ts_x.csv")
    ts_y = pd.read_csv("/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/ts_y.csv").to_numpy()

    scoring_dictionary = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "f1": f1_score,
        "recall": recall_score
    }

    strat_fold = StratifiedKFold(shuffle=True, random_state=42)

    dt_study = optuna.create_study(study_name="DecisionTreeClassifier study", direction="maximize")
    dt_study.optimize(
        lambda trial: _optuna_training(
            opt_trial=trial,
            train_x=tr_x,
            train_y=tr_y,
            test_x=ts_x,
            test_y=ts_y,
            scoring_dictionary=scoring_dictionary,
            strat_kfold=strat_fold
        ),
        n_trials=2500,
        n_jobs=100
    )
    
    print(f"Best score: {dt_study.best_value} | Best params: {dt_study.best_params}")

main()
