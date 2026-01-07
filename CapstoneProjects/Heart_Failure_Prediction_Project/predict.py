
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Load model pipeline
with open("model.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

# Define request schema
class Patient(BaseModel):
    age: float
    anaemia: int
    creatinine_phosphokinase: int
    diabetes: int
    ejection_fraction: int
    high_blood_pressure: int
    platelets: float
    serum_creatinine: float
    serum_sodium: int
    sex: int
    smoking: int
    time: int

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Heart Failure Prediction API is running"}

@app.post("/predict")
def predict(patient: Patient):
    # Convert Pydantic model to dict
    features = patient.dict()

    # Predict using pipeline
    y_pred = pipeline.predict([features])[0]
    y_proba = pipeline.predict_proba([features])[0, 1]

    return {
        "predicted_class": int(y_pred),
        "risk_probability": round(y_proba, 3),
        "risk_label": "At Risk" if y_pred == 1 else "Not at Risk"
    }


###----------------------------------------
# import pickle
# from fastapi import FastAPI
# import uvicorn

# MODEL_PATH = "model.bin"

# with open(MODEL_PATH, "rb") as f_in:
#     pipeline = pickle.load(f_in)

# patient = {
#     "age": 75.0,
#     "anaemia": 0,
#     "creatinine_phosphokinase": 582,
#     "diabetes": 0,
#     "ejection_fraction": 20,
#     "high_blood_pressure": 1,
#     "platelets": 265000.0,
#     "serum_creatinine": 1.9,
#     "serum_sodium": 130,
#     "sex": 1,
#     "smoking": 0,
#     "time": 4
# }

# app = FastAPI()
# @app.get("/")
# def home():
#     return {"Hello"}

# @app.post("/predict")
# def predict():
    
#     # The pipeline expects a list-of-dicts because DictVectorizer is the first step
#     y_pred = pipeline.predict([patient])[0]           # 0 or 1
#     y_proba = pipeline.predict_proba([patient])[0, 1] # probability of class 1

#     print(f"Predicted class: {y_pred}  |  Probability of risk: {y_proba:.3f}")
#     print("At Risk" if y_pred == 1 else "Not at Risk")
# --------------------------------------------------------------------------------
    # return {
    #     "prediction": label,
    #     "probability_of_leaving": prob,
    #     "message": message,
    # }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# import pickle

# with open('model.bin', 'rb') as f_in:
#     pipeline = pickle.load(f_in)

# patient = {
#     'age': 75.0,
#     'anaemia': 0,
#     'creatinine_phosphokinase': 582,
#     'diabetes': 0,
#     'ejection_fraction': 20,
#     'high_blood_pressure': 1,
#     'platelets': 265000.0,
#     'serum_creatinine': 1.9,
#     'serum_sodium': 130,
#     'sex': 1,
#     'smoking': 0,
#     'time': 4
# }

# # If your pipeline uses DictVectorizer, pass a list of dicts
# y_pred = pipeline.predict([patient])[0]          # 0 or 1
# y_proba = pipeline.predict_proba([patient])[0,1] # probability of class 1

# print(f"Predicted class: {y_pred}  |  Probability of risk: {y_proba:.3f}")

# if y_pred == 1:
#     print("At Risk")
# else:
#     print("Not at Risk")



# predict.py