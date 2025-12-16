# Speed Dating Match Prediction (Custom XGBoost)

## Abstract
This project predicts whether a pair will form a match using the **Speed Dating** dataset.  
It focuses on feature engineering and a **custom implementation of an XGBoost-style classifier**, emphasizing modeling fundamentals rather than library usage.


## Problem
- Predict a binary outcome (`match / no match`) from structured behavioral data.
- Handle missing values and heterogeneous feature types.
- Maintain a clear and reproducible modeling pipeline.


## Approach
- Engineered interaction-based features such as age gaps, same-race indicators, and importance × rating scores.
- Standardized missing values using a sentinel value and applied consistent imputation.
- Implemented a custom gradient-boosted decision tree classifier with logistic loss.
- Evaluated generalization using a held-out test set.


## Key Findings
- Interaction features significantly improve predictive signal in behavioral datasets.
- Clear separation between preprocessing, modeling, and evaluation improves maintainability without over-engineering.
- Implementing the model from scratch clarifies trade-offs in boosting-based methods.

## Code
- `preprocess.py`: Converts raw speed-dating data into model-ready features through systematic feature engineering and encoding.
- `custom_xgb.py`: Custom implementation of a gradient-boosted decision tree classifier, including split evaluation and logistic loss optimization.
- `train.py`: Orchestrates data preparation, model training, and prediction.
- `evaluate.py`: Contains evaluation utilities to assess model performance on held-out data.

## Tools and Libraries
- **Python**
- **pandas** – Data manipulation and preprocessing
- **numpy** – Numerical computation

## Contribution
- Designed and implemented the full modeling pipeline.
- Engineered domain-specific interaction features.
- Implemented a custom gradient boosting classifier to demonstrate algorithmic understanding beyond library usage.
