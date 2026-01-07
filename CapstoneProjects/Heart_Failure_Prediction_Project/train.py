# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle


# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score,classification_report,confusion_matrix,roc_curve,root_mean_squared_error
# from sklearn.metrics import classification_report
# from sklearn.tree import export_text,plot_tree
# from sklearn.feature_extraction import DictVectorizer

# def load_data():
#     df = pd.read_csv('././heart_failure_clinical_records_dataset.csv')
#     df.rename(columns={'DEATH_EVENT': 'death_event'}, inplace=True)
#     # print(df.columns)
#     # x = df.drop(columns=['death_event'],axis=1)
#     # y = df['death_event']
#     return df


# def train_model(df):
#     x = df.drop(columns=['death_event'])
#     y = df['death_event'].astype(int)
#     X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)
#     dv = DictVectorizer(sparse=False)
#     train_dicts = X_train.to_dict(orient='records')
#     test_dicts = X_test.to_dict(orient='records')

#     X_train = dv.fit_transform(train_dicts)
#     X_test = dv.transform(test_dicts)

#     features = dv.get_feature_names_out().tolist()

#     LR_Model = LogisticRegression(max_iter=1000, solver='liblinear',penalty='l1')
#     LR_Model.fit(X_train,y_train)
#     return LR_Model



# def evaluate(LR_Model, X_train, X_test, y_train, y_test):
#     y_test_pred = model.predict(X_test)
#     y_train_pred = model.predict(X_train)

#     print("TRAINIG RESULTS: \n===============================")
#     clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
#     print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
#     print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
#     print(f"CLASSIFICATION REPORT:\n{clf_report}")

#     print("TESTING RESULTS: \n===============================")
#     clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
#     print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
#     print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
#     print(f"CLASSIFICATION REPORT:\n{clf_report}")

#     evaluate(LR_Model, X_train, X_test, y_train, y_test)

#     # final_result = {
#     #     'Logistic Regression': {
#     #         'Train': roc_auc_score(y_train, LR_Model.predict(X_train)),
#     #         'Test': roc_auc_score(y_test, LR_Model.predict(X_test)),
#     #     },
#     # }
#     # final_result


# def save_model(filename, model):
#     with open (filename, 'wb') as f_out:
#         pickle.dump(model, f_out)

#     print(f'model saved to {filename}')

# df = load_data()
# pipeline = train_model(df)
# save_model('model.bin', pipeline)



# train.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

MODEL_PATH = "model.bin"
RANDOM_STATE = 42

def load_data(csv_path: str = '././heart_failure_clinical_records_dataset.csv') -> pd.DataFrame:
    """Load dataset and standardize target column name."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"DEATH_EVENT": "death_event"})
    # Ensure target is integer 0/1
    df["death_event"] = df["death_event"].astype(int)
    return df

def make_pipeline() -> Pipeline:
    """Create a pipeline: DictVectorizer -> LogisticRegression."""
    return Pipeline(steps=[
        ("dv", DictVectorizer(sparse=False)),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear",  # supports L1 penalty & predict_proba
            penalty="l1",
            class_weight="balanced",  # helpful for moderate imbalance
            random_state=RANDOM_STATE
        ))
    ])

def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train the pipeline and print evaluation metrics on a held-out test set."""
    X = df.drop(columns=["death_event"])
    y = df["death_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # List-of-dicts for DictVectorizer
    train_dicts = X_train.to_dict(orient="records")
    test_dicts  = X_test.to_dict(orient="records")

    pipeline = make_pipeline()
    pipeline.fit(train_dicts, y_train)

    # Evaluation
    y_pred_test = pipeline.predict(test_dicts)
    y_proba_test = pipeline.predict_proba(test_dicts)[:, 1]

    print("\n=== TEST RESULTS ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print("Accuracy:", f"{accuracy_score(y_test, y_pred_test):.4f}")
    print("ROC-AUC:", f"{roc_auc_score(y_test, y_proba_test):.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred_test, digits=3))

    return pipeline

def save_pipeline(pipeline: Pipeline, filename: str = MODEL_PATH) -> None:
    with open(filename, "wb") as f_out:
        pickle.dump(pipeline, f_out)
    print(f"\nModel pipeline saved to {filename}")

if __name__ == "__main__":
    df = load_data()
    pipeline = train_and_evaluate(df)
    save_pipeline(pipeline, MODEL_PATH)
