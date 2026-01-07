
import requests

# FastAPI endpoint URL
url = "http://127.0.0.1:9696/predict"

# Patient data payload
payload = {
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

# Send POST request
response = requests.post(url, json=payload)

# Print response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
