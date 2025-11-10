import os
import httpx
import asyncio
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

API_KEY = os.getenv("GMS_KEY")
SHORT_SUMMARY_MODEL = os.getenv("SHORT_SUMMARY_MODEL")
LONG_SUMMARY_MODEL = os.getenv("LONG_SUMMARY_MODEL")
SUMMARY_URL = os.getenv("SUMMARY_GMS_URL")

async def shortSummarize(text:str):
    prompt = f"""
    아래의 내용이 입력받은 다이어리의 내용입니다.
    해당 내용을 최대 20글자 내로 중요한 내용만 뽑아서 요약해주세요.
    특별한 사건이 일어났다면 해당 사건과 관련해서 감정적인 요약을 해줘도 좋습니다.
    예시를 참고하여 결과를 받아 주세요.
    답변의 마지막은 반드시 "~한 날"로 끝나야만 합니다.
    
    예시 1: 
    Input : 오늘 버스를 타고 집에 가다가 어떤 사람이 내 발을 밟아서 너무 짜증났어.
    Response : 발을 밟혀 기분이 나쁜 날.
    
    예시 2: 
    Input : 출근하는 길에 엄청나게 귀여운 고양이를 봤는데 나한테 와서 부비적 거려서 기분 좋았어.
    Response : 고양이가 다가와서 기분 좋은 날.
    
    Input : {text}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    messages = [
            {"role": "system", "content": "당신은 입력받은 텍스트를 요약하는 고성능 AI입니다. 입력 받은 텍스트를 요약해주세요. 한국어로 대답해주세요."},
            {"role": "user", "content": prompt},
        ]

    payload = {
        "model": SHORT_SUMMARY_MODEL,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.5,
    }
    
    try:
        async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
            response = await client.post(SUMMARY_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        reason = result["choices"][0]["message"]["content"].strip()
        return reason
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 코드는 {e}")



async def longSummarize(text):
    prompt = f"""
    아래의 내용이 입력받은 다이어리의 내용입니다.
    해당 내용을 중요한 내용을 뽑아서 요약해주세요.
    특별한 사건이 일어났다면 해당 사건과 관련해서 감정적인 요약을 해줘도 좋습니다.
    해당 요약본은 우리 어플의 이용자에게는 보이지 않지만 추후 내부적으로 사용할 것이기 때문에 불필요한 내용은 없애되, 중요한 내용은 모두 남겨주세요.
    요약본의 길이는 상관 없습니다. Input text의 길이와 차이가 나지 않아도 괜찮습니다.
    아래의 예시를 참고하여 요약해주세요.
    
    예시 1: 
    Input : 오늘 버스를 타고 집에 가다가 어떤 사람이 내 발을 밟아서 너무 짜증났어.
    Response : 버스를 타고 집에 가다가 발을 밟혀 짜증이 남.
    
    예시 2: 
    Input : 출근하는 길에 엄청나게 귀여운 고양이를 봤는데 나한테 와서 부비적 거려서 기분 좋았어.
    Response : 출근길에 고양이가 부비적 거려서 기분이 좋았음.
    
    Input : {text}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    messages = [
            {"role": "system", "content": "당신은 입력받은 텍스트를 요약하는 고성능 AI입니다. 입력 받은 텍스트를 요약해주세요. 한국어로 대답해주세요."},
            {"role": "user", "content": prompt},
        ]

    payload = {
        "model": LONG_SUMMARY_MODEL,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    
    try:
        async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
            response = await client.post(SUMMARY_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        reason = result["choices"][0]["message"]["content"].strip()
        return reason
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 코드는 {e}")