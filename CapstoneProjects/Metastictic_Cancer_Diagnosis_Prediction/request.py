
# request.py
import requests

url = "http://127.0.0.1:9696/predict"

payload = {"patient_id": 730681}

response = requests.post(url, json=payload)
print(response.json())
