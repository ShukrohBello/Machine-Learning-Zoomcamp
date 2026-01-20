
# request.py
import requests

url = "http://127.0.0.1:8000/predict"

payload = {"patient_id": 730681}

response = requests.post(url, json=payload)
print(response.json())
