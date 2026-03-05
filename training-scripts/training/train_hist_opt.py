import optuna as opt
from optuna.pruners import HyperbandPruner
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score

def _optuna_training (
    optuna_trial,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    strat_kfold: StratifiedKFold,
):
    iteration_predictions_array = []
    
    hist_tree_grid = {
        "learning_rate": optuna_trial.suggest_float(name="HIST_LEARNING_RATE", low=0.00001, high=0.1, log=True),
        "max_iter": optuna_trial.suggest_int(name="HIST_MAX_ITER", low=100, high=500, step=100),
        "min_samples_leaf": optuna_trial.suggest_int(name="HIST_MIN_SAMPLES_LEAF", low=20, high=120, step=20),
        "n_iter_no_change": optuna_trial.suggest_int(name="HIST_ITER_NO_CHANGE", low=5, high=10, step=1),
        "tol": optuna_trial.suggest_float(name="HIST_TOL", low=0.000001, high=0.1, log=True),
        "l2_regularization": optuna_trial.suggest_float(name="HIST_L2", low=0.0001, high=0.1, log=True),
        "random_state": 42
    }
    
    for tr_idx, val_idx in strat_kfold.split(train_x, train_y):
        iterated_hist_tree = HistGradientBoostingClassifier(**hist_tree_grid)
        tr_x_subset, tr_y_subset = train_x.iloc[tr_idx], train_y[tr_idx]
        val_x_subset, val_y_subset = train_x.iloc[val_idx], train_y[val_idx]
        
        iterated_hist_tree.fit(tr_x_subset, tr_y_subset)
        iteration_predictions = iterated_hist_tree.predict(val_x_subset)
        
        iteration_predictions_array.append(precision_score(
            val_y_subset,
            iteration_predictions,
            average="weighted",
            zero_division=0
        ))
        
    return np.mean(np.asarray(iteration_predictions_array))

def main ():
    tr_x = pd.read_csv("../../datasets/tr_x.csv")
    tr_y = pd.read_csv("../../datasets/tr_y.csv")
    ts_x = pd.read_csv("../../datasets/ts_x.csv")
    ts_y = pd.read_csv("../../datasets/ts_y.csv")
    
    strat_kfold = StratifiedKFold(shuffle=True, random_state=42)

    hist_tree_study = opt.create_study(
        study_name="HIST_TREE_TRAINING",
        direction="maximize",
        pruner=HyperbandPruner(
            min_resource=10,
            max_resource=20
        )
    )
    
    hist_tree_study.optimize(
        lambda trial: _optuna_training(
            trial,
            train_x=tr_x,
            train_y=tr_y.to_numpy().ravel(),
            strat_kfold=strat_kfold
        ),
        n_trials=2500,
        n_jobs=45
    )
    
    print(f"Best parameters: {hist_tree_study.best_params}")
    
    test_hist_tree = HistGradientBoostingClassifier(**hist_tree_study.best_params)
    test_hist_tree_predictions = test_hist_tree.predict(ts_x)
    final_precision_score = precision_score(test_hist_tree_predictions, ts_y)
    print(f"Precision score of final HistGradientBoostingClassifier: {final_precision_score}")
    
main()
