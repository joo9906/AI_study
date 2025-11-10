from dotenv import load_dotenv
import requests
import weaviate
import weaviate.classes as wvc

# Connect to Weaviate
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    grpc_host="localhost",
    grpc_port=50051,
    http_secure=False,
    grpc_secure=False,
)

single = client.collections.get("SingleCounsel")

result = single.query.fetch_objects(limit=1, include_vector=True)


for obj in result.objects:
    print("Input:", obj.properties["input"])
    print("Vector length:", len(obj.vector["default"]))

client.close()