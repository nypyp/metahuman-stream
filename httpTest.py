import requests

response = requests.get('http://8.134.150.174:8000')
print(response.text)