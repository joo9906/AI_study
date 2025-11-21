import os
import json
from dotenv import load_dotenv
import requests
import weaviate
import weaviate.classes as wvc

load_dotenv()

# ENV
API_KEY = os.getenv("GMS_KEY")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_URL = os.getenv("EMBEDDING_GMS_URL")

# Connect to Weaviate
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    grpc_host="localhost",
    grpc_port=50051,
    http_secure=False,
    grpc_secure=False,
)

try:
    existing = client.collections.list_all()

    if "SingleCounsel" not in existing:
        client.collections.create(
            name="SingleCounsel",
            description="단일 상담 데이터",
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=[
                wvc.config.Property(name="input", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="output", data_type=wvc.config.DataType.TEXT),
            ],
        )
        print("✅ SingleCounsel 컬렉션 생성 완료")
    else:
        print("✅ SingleCounsel 이미 존재. 생성 생략.")

    if "MultiCounsel" not in existing:
        client.collections.create(
            name="MultiCounsel",
            description="멀티턴 상담 데이터",
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=[
                wvc.config.Property(name="dialogue", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="summary", data_type=wvc.config.DataType.TEXT),
            ],
        )
        print("✅ MultiCounsel 컬렉션 생성 완료")
    else:
        print("✅ MultiCounsel 이미 존재. 생성 생략.")

    def embed(text: str) -> list:
        headers = {
            "Content-type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        payload = {"model": EMB_MODEL, "input": text}

        res = requests.post(EMB_URL, headers=headers, json=payload)
        res.raise_for_status()
        
        result = res.json()["data"][0]["embedding"]

        return result  # 1536 or 3072 dim

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # -------------------------------
    # Upload SingleCounsel
    # -------------------------------
    single_collection = client.collections.get("SingleCounsel")
    single_data = load_jsonl("./total_kor_counsel_bot.jsonl")[:50]

    single_objs = []
    for item in single_data:
        vec = embed(item["input"])
        single_objs.append(
            wvc.data.DataObject(
                properties={
                    "input": item["input"].strip(),
                    "output": item["output"].strip(),
                },
                vector=vec,
            )
        )

    single_collection.data.insert_many(single_objs)
    print("✅ SingleCounsel 업로드 완료")

    # -------------------------------
    # ✅ Upload MultiCounsel
    # -------------------------------
    multi_collection = client.collections.get("MultiCounsel")
    multi_data = load_jsonl("./total_kor_multiturn_counsel_bot.jsonl")[:50]

    multi_objs = []
    for turns in multi_data:
        dialogue = "\n".join(f"{t['speaker']}: {t['utterance']}" for t in turns)
        print("현재 업로드 하는 내용은 : ", dialogue)
        vec = embed(dialogue)

        multi_objs.append(
            wvc.data.DataObject(
                properties={"dialogue": dialogue, "summary": ""},
                vector=vec,
            )
        )

    multi_collection.data.insert_many(multi_objs)
    print("✅ MultiCounsel 업로드 완료")

    # -------------------------------
    # ✅ Vector 확인 테스트
    # -------------------------------
    test_result = single_collection.query.fetch_objects(limit=1, return_vector=True)

    if test_result.objects and test_result.objects[0].vector:
        print("🎉 Vector length:", len(test_result.objects[0].vector))
    else:
        print("❌ Vector가 저장되지 않았습니다.")

finally:
    client.close()
    print("🔌 Weaviate 연결 종료")
