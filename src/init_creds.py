import os
import requests

# Set Doppler token
doppler_token = os.getenv("DOPPLER_AIGUILD_TOKEN")
headers = {
    "Authorization": f"Bearer {doppler_token}"
}

def get_api_key():
    response = requests.get(
        "https://api.doppler.com/v3/configs/config/secrets/download",
        headers=headers,
        params={
            "project": "dep-training",
            "config": "prod_ai_guild_genai_practicum",
            "name": "AZURE_OPENAI_API_KEY"
        }
    )
    response.raise_for_status()
    api_key = response.json()["secrets"]["AZURE_OPENAI_API_KEY"]["raw"]
    return api_key

def get_endpoint():
    response = requests.get(
        "https://api.doppler.com/v3/configs/config/secrets/download",
        headers=headers,
        params={
            "project": "dep-training",
            "config": "prod_ai_guild_genai_practicum",
            "name": "AZURE_OPENAI_API_BASE"
        }
    )
    response.raise_for_status()
    endpoint = response.json()["secrets"]["AZURE_OPENAI_API_BASE"]["raw"]
    return endpoint