
# train.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = "train.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop columns with 100% missing
    df = df.drop(columns=["metastatic_first_novel_treatment_type",
                          "metastatic_first_novel_treatment"], errors='ignore')

    # Impute BMI and categorical missing values
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["patient_race"] = df["patient_race"].fillna("Unknown")
    df["payer_type"] = df["payer_type"].fillna("Unknown")

    return df

def build_pipeline(categorical_cols, numeric_cols):
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline

def train():
    df = load_data()

    target = "metastatic_diagnosis_period"
    y = df[target]
    X = df.drop(columns=[target, "patient_id"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)



    preds = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, preds)  # works in all versions
    rmse = np.sqrt(mse)
    print("Validation RMSE:", rmse)

    with open("model.pkl", "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "categorical_cols": categorical_cols,
            "numeric_cols": numeric_cols
        }, f)

    print("Model saved → model.pkl")

if __name__ == "__main__":
    train()

    