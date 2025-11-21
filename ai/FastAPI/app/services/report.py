import os
import httpx
import asyncio
from dotenv import load_dotenv
load_dotenv()

GMS_API_KEY = os.getenv("GMS_KEY")
SUMMARIZE_API_URL = os.getenv("GMS_URL")
MODEL = os.getenv("REPORT_MODEL")


async def report(text: str) -> str:
    prompt = f"""
    사용자의 상사에게 어떠한 상담을 해주면 좋을지 간단하게 설명해줘야 합니다. 
    사용자는 소방관이며, 상사는 사용자의 팀장이라고 가정합니다. 
    팀장이 사용자에게 어떤 조언을 해주면 좋을지 알려주세요.
    아래가 사용자의 하루 분석 내용입니다.

    {text}
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_API_KEY}",
    }
    
    messages = [
            {
                "role": "system",
                "content": "당신은 친절한 상담사입니다. 프롬프트에 맞게 상담사의 입장에서 답변해주세요.",
            },
            {   
                "role": "user", 
                "content": prompt,
            },
        ]

    payload = {
        "model": "gpt-5",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
    }
    try:
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            response = await client.post(SUMMARIZE_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

        reason = result["choices"][0]["message"]["content"].strip()
        return reason

    except httpx.HTTPError as e:
        print(f"❌ HTTP 요청 실패: {e}")
        return "서버 요청 중 오류가 발생했습니다."
    except Exception as e:
        print(f"❌ 예기치 못한 오류: {e}")
        return "예기치 못한 오류가 발생했습니다."
