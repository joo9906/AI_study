import os
import asyncio
import weaviate
import httpx
from fastapi import HTTPException
from dotenv import load_dotenv
from langchain_community.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.vector_embedding import embed  # <- 당신이 만든 비동기 임베딩 함수

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
ADVICE_URL = os.getenv("COUNSELING_GMS_URL")
ADVICE_MODEL = os.getenv("COUNSELING_MODEL")
GMS_KEY = os.getenv("GMS_KEY")

# Weaviate 연결
client = weaviate.connect_to_local()

# 유사 상담내용 검색
async def retrieve_similar_cases(query: str, top_k: int = 2):
    """
    외부 async embed()로 벡터 생성 후, weaviate에서 직접 검색 수행
    """
    try:
        # ① 쿼리 임베딩 생성 (HTTP 비동기)
        query_vector = await embed(query)

        # ② 단일 상담 검색
        single_coll = client.collections.get("SingleCounsel")
        single_res = single_coll.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["content"],
        )

        # ③ 다중 상담 검색
        multi_coll = client.collections.get("MultiCounsel")
        multi_res = multi_coll.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_properties=["dialogue"],
        )

        # 결과만 텍스트로 추출
        single_texts = [o.properties.get("content", "") for o in single_res.objects]
        multi_texts = [o.properties.get("dialogue", "") for o in multi_res.objects]

        return single_texts, multi_texts

    except Exception as e:
        print("검색 중 오류:", e)
        return [], []

# 조언 생성 함수
async def gms_generate_advice(text: str):
    single, multi = await retrieve_similar_cases(text)
    
    single_text = "\n".join([f"- {s[:300]}..." for s in single])
    multi_text = "\n".join([f"- {m[:300]}..." for m in multi])
    
    prompt = f"""
        당신은 팀장으로서 팀원의 상태를 보고 조언을 제시하는 역할입니다.

        [사용자 요약]
        {text}

        [팀원의 상태와 유사한 상담 사례]
        단일 상담의 경우 : 
        {single_text}
        
        멀티턴 상담의 경우 : 
        {multi_text}

        - 존댓말로 조언 작성
        - 불필요한 감정 표현은 피하고, 현실적이고 따뜻하게 조언할 것
        - 팀장은 상담 전문가가 아니므로 보다 안전하고 조심스러운 접근 방법을 제시할 것.
        """
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GMS_KEY}",
    }
    
    messages = [
        {
            "role": "system",
            "content": "당신은 팀장의 입장에서 팀원에게 조언을 주는 상담 코치입니다. 한국어로 대답해 주세요. 전문 상담사가 아닌 조언을 하는 입장이므로 전문적인 용어보다는 쉽게, 보다 인간적으로 접근할 수 있는 방법을 알려주세요.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    payload = {
        "model": ADVICE_MODEL,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.6,
    }

    try:
        async with httpx.AsyncClient(verify=False, timeout=15.0) as cli:
            response = await cli.post(ADVICE_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        advice = result["choices"][0]["message"]["content"].strip()
        return advice

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GMS 요청 중 오류 발생: {e}")


# ---------------------------
# 테스트용 진입점
# ---------------------------
if __name__ == "__main__":
    test_summary = "최근 화재 출동이 많아 스트레스가 누적되고 있습니다. 수면 부족으로 집중력이 떨어집니다."
    advice = asyncio.run(gms_generate_advice(test_summary))
    print("\n=== 생성된 조언 ===\n")
    print(advice)
