import pickle

MODEL_PATH = "model.bin"

with open(MODEL_PATH, "rb") as f_in:
    pipeline = pickle.load(f_in)

patient = {
    "age": 75.0,
    "anaemia": 0,
    "creatinine_phosphokinase": 582,
    "diabetes": 0,
    "ejection_fraction": 20,
    "high_blood_pressure": 1,
    "platelets": 265000.0,
    "serum_creatinine": 1.9,
    "serum_sodium": 130,
    "sex": 1,
    "smoking": 0,
    "time": 4
}

# The pipeline expects a list-of-dicts because DictVectorizer is the first step
y_pred = pipeline.predict([patient])[0]           # 0 or 1
y_proba = pipeline.predict_proba([patient])[0, 1] # probability of class 1

print(f"Predicted class: {y_pred}  |  Probability of risk: {y_proba:.3f}")
print("At Risk" if y_pred == 1 else "Not at Risk")



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