import requests
import json
import base64

url = 'https://pale-sari-zeromeout-732e5b98.koyeb.app/predict'

try:
    # Read image file
    with open('cat.jpg', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Send the base64 string in the request body
    data = {'image': encoded_image}
    response = requests.post(url, json=data)
    
    # Force raise for bad status codes
    response.raise_for_status()
    
    # Parse the response
    result = response.json()
    print(f"Status Code: {response.status_code}")
    print(f"Prediction Result: {result}")

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
    print(f"Response Text: {response.text if 'response' in locals() else 'No response'}")