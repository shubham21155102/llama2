import requests

url = "http://127.0.0.1:8000/generate"
prompt_text = "Hey, are you conscious? Can you talk to me?"

response = requests.post(url, json={"prompt": prompt_text})
print(response.json()["generated_text"])