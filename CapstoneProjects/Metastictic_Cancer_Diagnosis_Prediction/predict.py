
import pickle
import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
#from predict import predict_single

def predict_single(patient_id):
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]

    df_test = pd.read_csv("test.csv")
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


class PatientRequest(BaseModel):
    patient_id: int

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Metastatic Cancer Prediction API is running"}

@app.post("/predict")
def predict(req: PatientRequest):
    return predict_single(req.patient_id)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)