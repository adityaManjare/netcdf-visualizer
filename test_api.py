import os
import requests
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# --- 1. Check if the token is being loaded ---
api_token = os.getenv('HF_API_TOKEN')

print("--- API Token Test ---")
if api_token:
    print("✅ Token found in .env file.")
else:
    print("❌ ERROR: Token NOT found. Check your .env file name and content.")
    exit() # Stop the script if no token is found

# --- 2. Test the API connection ---
api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}
payload = {"inputs": "This is a test."}

print("\n--- API Connection Test ---")
print("📞 Calling the Hugging Face API...")

try:
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        print("✅ SUCCESS! Connected to the API and got a valid response.")
    elif response.status_code == 403:
        print("❌ ERROR 403: Forbidden. Your API token is invalid or does not have the correct permissions.")
    elif response.status_code == 401:
         print("❌ ERROR 401: Unauthorized. Your API token is likely incorrect or missing.")
    else:
        print(f"❌ FAILED with status code: {response.status_code}")
        print(f"   Response: {response.text}")

except Exception as e:
    print(f"❌ An error occurred: {e}")