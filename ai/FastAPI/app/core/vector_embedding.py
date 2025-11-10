from dotenv import load_dotenv
import os
import httpx
import requests
load_dotenv()

VECTOR_DB = os.getenv("WEAVIATE_URL")
API_KEY = os.getenv("GMS_KEY")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_URL = os.getenv("EMBEDDING_GMS_URL")

async def embed(text:str) -> list:
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": EMB_MODEL,
        "input": text,
    }
    
    async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
        try:
            response = await requests.post(EMB_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            output = result["data"][0]["embedding"]
            
            return output

        except Exception as e:
            print("에러 발생, 에러 코드 : ", e)
            