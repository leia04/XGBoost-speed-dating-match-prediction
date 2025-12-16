import os
import pandas as pd
import numpy as np

def age_gap(x):
    if x["age"] == -99:
        return -99
    elif x["age_o"] == -99:
        return -99
    elif x["gender"] == "female":
        return x["age_o"] - x["age"]
    else:
        return x["age"] - x["age_o"]

def same_race(x):
    if x["race"] == -99:
        return -99
    elif x["race_o"] == -99:
        return -99
    elif x["race"] == x["race_o"]:
        return 1
    else:
        return -1

def same_race_point(x):
    if x["same_race"] == -99:
        return -99
    else:
        return x["same_race"] * x["importance_same_race"]

def rating(row, importance, score):
    if row[importance] == -99:
        return -99
    elif row[score] == -99:
        return -99
    else:
        return row[importance] * row[score]

def preprocess(
    input_csv: str = "data/raw/speed_dating.csv",
    output_csv: str = "data/processed/speed_dating_processed.csv",
):
    data = pd.read_csv(input_csv)

    data = data.fillna(-99)

    data["age_gap"] = data.apply(age_gap, axis=1)
    data["age_gap_abs"] = abs(data["age_gap"])

    data["same_race"] = data.apply(same_race, axis=1)
    if "importance_same_race" in data.columns:
        data["same_race_point"] = data.apply(same_race_point, axis=1)

    partner_imp = data.columns[9:15]
    partner_rate_me = data.columns[15:21]
    my_imp = data.columns[21:27]
    my_rate_partner = data.columns[27:33]

    new_label_partner = [
        "attractive_p", "sincere_partner_p", "intelligence_p",
        "funny_p", "ambition_p", "shared_interests_p"
    ]
    new_label_me = [
        "attractive_m", "sincere_partner_m", "intelligence_m",
        "funny_m", "ambition_m", "shared_interests_m"
    ]

    for i, j, k in zip(new_label_partner, partner_imp, partner_rate_me):
        data[i] = data.apply(lambda x: rating(x, j, k), axis=1)

    for i, j, k in zip(new_label_me, my_imp, my_rate_partner):
        data[i] = data.apply(lambda x: rating(x, j, k), axis=1)  

    data = pd.get_dummies(data, columns=["gender", "race", "race_o"], drop_first=True)
    bool_columns = data.select_dtypes(include="bool").columns
    data[bool_columns] = data[bool_columns].astype(int)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    data.to_csv(output_csv, index=False)
    return output_csv

if __name__ == "__main__":
    out = preprocess()
    print(f"Saved processed dataset: {out}")
