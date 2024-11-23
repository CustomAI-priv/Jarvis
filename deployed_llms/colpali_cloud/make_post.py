import requests
import json

# Define the URL
url = 'http://127.0.0.1:8000/process_images'

# Load the JSON data from a file
with open('sample.json', 'r') as file:
    json_data = json.load(file)

# Convert JSON data to string and then to bytes
json_bytes = json.dumps(json_data).encode('utf-8')

# Create a dictionary for files
files = {
    'file': ('sample.json', json_bytes, 'application/json')
}

# Send the POST request with the file
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    print("Request successful!")
    print("Response JSON:", response.json())
else:
    print("Request failed with status code:", response.status_code)
    print("Response text:", response.text)