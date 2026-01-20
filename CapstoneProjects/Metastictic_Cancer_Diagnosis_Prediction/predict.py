
# predict.py
import pickle
import pandas as pd

def predict_single(patient_id):
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]

    df_test = pd.read_csv("data/test.csv")
    row = df_test[df_test["patient_id"] == patient_id]

    if row.empty:
        return {"error": "Patient ID not found"}

    X = row.drop(columns=["patient_id"])
    y_pred = pipeline.predict(X)[0]

    return {
        "patient_id": int(patient_id),
        "metastatic_diagnosis_period": float(y_pred)
    }

if __name__ == "__main__":
    print(predict_single(730681))


# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_single

class PatientRequest(BaseModel):
    patient_id: int

app = FastAPI()

@app.post("/predict")
def predict(req: PatientRequest):
    return predict_single(req.patient_id)
