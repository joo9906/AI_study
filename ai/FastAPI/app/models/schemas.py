from pydantic import BaseModel
from typing import List, Dict, Any

# 감정 분석
class EmotionInput(BaseModel):
    user_id : int
    texts : List[str]

class EmotionOutput(BaseModel):
    user_id : int
    result: Dict[str, Any]

# 일간 다이어리 전용
class DailyInput(BaseModel):
    user_id : int
    diary : str
    
class DailyOutput(BaseModel):
    user_id : int
    result: Dict[str, float]

# 주간(월간) 보고서 전용
class WeeklyInput(BaseModel):
    user_id : int
    text : str
    
class WeeklyOutput(BaseModel):
    user_id : int
    result : Dict[str, Any]

# 관리자 상담 전용
class CouncelInput(BaseModel):
    user_id : int
    summarized_list : List

class CouncelOutput(BaseModel):
    user_id : int
    suggest : str