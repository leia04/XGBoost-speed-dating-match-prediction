import numpy as np
import pandas as pd

from preprocess import preprocess
from custom_xgb import XGBoostClassifier
from eval import print_accuracy


def load_processed(path_candidates):
    for p in path_candidates:
        try:
            return pd.read_csv(p)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Processed csv not found. Tried: {path_candidates}")


def main():
    np.random.seed(100)

    preprocess()

    data = load_processed([
        "data/processed/speed_dating_data_processed.csv",
        "data/processed/speed_dating_data1.csv",
        "speed_dating_data_1.csv",
        "speed_dating_data1.csv",
    ])

    columns_to_drop = data.columns[:31]
    data = data.drop(columns=columns_to_drop, errors="ignore")
    data = data.drop(columns=["expected_num_interested_in_me"], errors="ignore")

    data.replace(-99, np.nan, inplace=True)
    for col in data.columns:
        data[col].fillna(data[col].mean(), inplace=True)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * 0.2)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    X_train = data.drop("match", axis=1).iloc[train_indices]
    X_test = data.drop("match", axis=1).iloc[test_indices]
    y_train = data["match"].iloc[train_indices]
    y_test = data["match"].iloc[test_indices]

    columns_pm = [col for col in data.columns if (col.endswith("_p") or col.endswith("_m"))]
    for col in columns_pm:
        if col in X_train.columns:
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
            X_train[col] = (X_train[col] - min_val) / denom
            X_test[col] = (X_test[col] - min_val) / denom

    classifier = XGBoostClassifier()
    classifier.fit(
        X_train.values, y_train.values,
        subsample_cols=0.8, min_child_weight=1, depth=5, min_leaf=5,
        learning_rate=0.1, boosting_rounds=50, lambda_=1.0,
        gamma=1, eps=0.1, early_stopping_rounds=10,
        eval_set=(X_test.values, y_test.values),
    )

    y_pred_train = classifier.predict(X_train.values)
    y_pred_test = classifier.predict(X_test.values)

    print_accuracy(y_train.values, y_pred_train, y_test.values, y_pred_test)


if __name__ == "__main__":
    main()
