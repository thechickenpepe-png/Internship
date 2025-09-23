from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

result = client.query_points(
    collection_name="project_air",
    query=[0.0]*384,   # you need a query vector here, otherwise Qdrant won’t know what to search for
    limit=1,
    with_payload=True
)

for hit in result.points:
    payload = hit.payload or {}   # <-- define payload here
    print("Keys:", payload.keys())
    print("Chunk text preview:", payload.get("chunk_text", "")[:200])
