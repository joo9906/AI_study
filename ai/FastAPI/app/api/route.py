
from fastapi import APIRouter, HTTPException
from app.models.schemas import EmotionOutput, EmotionInput, DailyInput, DailyOutput, CouncelInput, CouncelOutput, WeeklyInput, WeeklyOutput
from app.services.emotion_classify import emotionClassifying
from app.services.report import report
from app.services.summary import longSummarize, shortSummarize
import numpy as np
from .weav import embed_sentence, search_similar
import asyncio

router = APIRouter()

@router.get("/health", response_model=str)
async def health():
    """서버의 상태를 확인합니다."""
    return "OK"

# 사용자의 다이어리 문장들을 받아와 오늘의 감정 점수 + 일간 요약(짧은 요약, 긴 요약)을 반환
@router.post("/diary/report", response_model=EmotionOutput)
async def diary_classification(input_data: EmotionInput):
    try:
        user_id = input_data.user_id
        texts = input_data.texts

        # CPU/GPU 연산을 별도 스레드로 실행 (이벤트 루프 블로킹 방지)
        classify = await asyncio.to_thread(emotionClassifying, texts)

        # 감정 점수 받아오기
        sentiment = classify["sentiment"]
        score = classify["score"]

        # 짧고 긴 요약들 받아오기
        short_summary = await shortSummarize(" ".join(texts))
        long_summary = await longSummarize(" ".join(texts))

        result = {
            "user_id": user_id,
            "result": {
                "score": score,
                "sentiment": sentiment,
                "short_summary": short_summary,
                "long_summary": long_summary,
            },
        }
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 코드는 {e}")







# # 사용자 다이어리와 점수를 받아와 보고서 작성.
# @router.post("/gms/weekly/report", response_model=WeeklyOutput)
# async def gms_request(data: WeeklyInput):
#     try:
#         user_id = data.user_id
#         reply = await report(data.text)
        
#         return DailyOutput(result = reply)
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"오류 코드는 {e}")