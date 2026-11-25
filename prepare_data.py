"""Convert the raw UCI Cleveland heart-disease file to data/heart_disease.csv.

Run only if you re-downloaded data/processed.cleveland.data (see README).
"""
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

df = pd.read_csv("data/processed.cleveland.data", header=None, names=COLS, na_values="?")
# UCI 'num': 0 = no disease, 1..4 = disease present -> binary risk target
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])
df.to_csv("data/heart_disease.csv", index=False)
print("Wrote data/heart_disease.csv:", df.shape)