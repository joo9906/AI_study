import S13P31A106.ai.FastAPI.app.api.weav as weav
from weaviate.classes.config import Property, DataType, Configure

# ✅ v4 클라이언트 연결
try:
    weaviate_client = weav.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051,
    )
    print("✅ Weaviate 연결 성공")
except Exception as e:
    print(f"⚠️ Weaviate 연결 실패: {e}")
    weaviate_client = None


# ✅ 클래스(컬렉션) 존재 여부 확인 후 생성
def init_schema():
    if weaviate_client is None:
        return

    existing = weaviate_client.collections.list_all()
    if "Diary" not in existing:
        print("📘 Diary 컬렉션 생성 중...")
        weaviate_client.collections.create(
            name="Diary",
            properties=[
                Property(name="user_id", data_type=DataType.INT),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="embedding", data_type=DataType.NUMBER, vectorize=False),
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # 외부 embedding 사용 시
        )
        print("✅ Diary 컬렉션 생성 완료")
    else:
        print("📘 Diary 컬렉션 이미 존재함")


# ✅ 문장 임베딩 (GPT 임베딩 or OpenAI Embedding 등 연결 가능)
def embed_sentence(sentence: str):
    if weaviate_client is None:
        return None
    try:
        # 예시: Weaviate 자체 embedding 사용 시
        collection = weaviate_client.collections.get("Diary")
        vector = collection.generate.vectorize(text=sentence)
        return vector
    except Exception as e:
        print(f"❌ 임베딩 오류: {e}")
        return None


# ✅ 유사 문장 검색
def search_similar(vector):
    if weaviate_client is None:
        return []
    try:
        collection = weaviate_client.collections.get("Diary")
        results = collection.query.near_vector(
            near_vector=vector,
            limit=5
        )
        return results.objects
    except Exception as e:
        print(f"❌ 유사 문장 검색 오류: {e}")
        return []


# 서버 시작 시 스키마 초기화
init_schema()
