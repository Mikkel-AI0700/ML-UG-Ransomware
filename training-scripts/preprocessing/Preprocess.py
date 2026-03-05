import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    TargetEncoder,
    LabelEncoder,
)

def _removing_columns (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
    columns: list[str] = None
):
    train_x = train_x.drop(columns, axis=1)
    return train_x, train_y

def _standardize_columns (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
    columns: list[str] = None
):
    standardizer = StandardScaler()
    train_x[columns] = standardizer.fit_transform(train_x[columns])
    return train_x, train_y

def _one_hot_encoding_columns (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
    columns: list[str] = None
):
    ohe = OneHotEncoder()
    temporary_protcol_df = ohe.fit_transform(train_x[columns])
    temporary_protocol_df = pd.DataFrame(
        temporary_protcol_df.toarray(), 
        columns=ohe.get_feature_names_out()
    )

    train_x = pd.concat([
        train_x.drop(columns, axis=1),
        temporary_protocol_df
    ],
    axis=1
    )

    return train_x, train_y

def _target_encoding_columns (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
    columns: list[str] = None
):
    train_x.dropna(inplace=True)
    train_y.dropna(inplace=True)

    t_encoder = TargetEncoder()
    temp_flag_df = t_encoder.fit_transform(train_x[columns], train_y)
    temp_flag_df = pd.DataFrame(temp_flag_df, columns=t_encoder.get_feature_names_out())

    train_x = pd.concat([
        train_x.drop(columns, axis=1),
        temp_flag_df
    ],
    axis=1
    )

    return train_x, train_y

def _label_encode_target (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
    columns: list[str] = None
):
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)

    return train_x, train_y

def _oversample_training (
    train_x: np.ndarray = None,
    train_y: np.ndarray = None,
):
    oversampler = SMOTE()
    train_x, train_y = oversampler.fit_resample(train_x, train_y)
    return train_x, train_y

def main ():
    preprocessing_func_ref = [
        _removing_columns,
        _standardize_columns,
        _one_hot_encoding_columns,
        _target_encoding_columns,
        _label_encode_target,
    ]

    columns_to_preprocess = {
        "cols_remove": ["Family", "Clusters", "SeddAddress", "ExpAddress", "BTC", "USD", "IPaddress", "Threats", "Port"],
        "cols_standardize": ["Time", "Netflow_Bytes"],
        "cols_ohe": ["Protcol"],
        "cols_target": ["Flag"],
        "col_label": ["Prediction"]
    }

    dataset_path = "/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/final(2).csv"
    main_dataset = pd.read_csv(dataset_path)

    X = main_dataset.iloc[:, :-1]
    Y = main_dataset.iloc[:, -1]

    for pre_ref, col_key in zip(preprocessing_func_ref, columns_to_preprocess.keys()):
        X, Y = pre_ref(X, Y, columns_to_preprocess.get(col_key))
        print(f"Started preprocessing step: {pre_ref}")

        print(f"Train X columns: {X.columns}")

    tr_x, ts_x, tr_y, ts_y = train_test_split(
        X,
        Y,
        train_size=0.9,
        test_size=0.1,
        random_state=42
    )

    tr_x, tr_y = _oversample_training(tr_x, tr_y)

    tr_x, tr_y = pd.DataFrame(tr_x), pd.DataFrame(tr_y)
    ts_x, ts_y = pd.DataFrame(ts_x), pd.DataFrame(ts_y)

    datasets_list = [tr_x, tr_y, ts_x, ts_y]
    datasets_name = ["tr_x.csv", "tr_y.csv", "ts_x.csv", "ts_y.csv"]

    for csv_dataset, csv_name in zip(datasets_list, datasets_name):
        with open(f"/home/mikkel/Desktop/ai-projects/machine-learning/ug-ransomware/datasets/{csv_name}", "w") as csv_file:
            csv_dataset.to_csv(csv_file, index=False)

main()

