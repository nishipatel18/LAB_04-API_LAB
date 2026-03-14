import requests
import json

url = 'http://127.0.0.1:8080/predict'

# Sample wine data (Class 0 - Barolo)
payload = {
    "alcohol": 13.2,
    "malic_acid": 1.78,
    "ash": 2.14,
    "alcalinity_of_ash": 11.2,
    "magnesium": 100.0,
    "total_phenols": 2.65,
    "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26,
    "proanthocyanins": 1.28,
    "color_intensity": 4.38,
    "hue": 1.05,
    "od280_od315": 3.40,
    "proline": 1050.0
}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print("Status:", response.status_code)
print("Body:", response.text)

if response.status_code == 200:
    try:
        prediction = response.json()['prediction']
        print('Predicted wine class:', prediction)
    except Exception as e:
        print("Could not parse JSON:", e)
else:
    print('Error:', response.status_code)
