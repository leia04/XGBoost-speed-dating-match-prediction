import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from config import Config
from data_prep import preprocess, save_processed
from split import split_and_scale
from models.custom_xgb import XGBoostClassifier

def main():
    cfg = Config()

    df = preprocess(cfg.raw_path)
    save_processed(df, cfg.processed_path)

    X_train, X_test, y_train, y_test = split_and_scale(
        pd.read_csv(cfg.processed_path),
        seed=cfg.seed,
        test_size=cfg.test_size,
    )

    model = XGBoostClassifier().fit(
        X_train.values, y_train.values,
        learning_rate=0.1, boosting_rounds=50,
        depth=5, min_leaf=5,
        early_stopping_rounds=10,
        eval_set=(X_test.values, y_test.values),
    )

    pred = model.predict(X_test.values)
    proba = model.predict_proba(X_test.values)

    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")

if __name__ == "__main__":
    main()
