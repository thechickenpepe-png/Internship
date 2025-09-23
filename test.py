import os
import fitz  # PyMuPDF
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- Setup ---
pdf_path = r"Project/737-345_AMM/PDF/05___041.PDF"
collection_name = "project_air_test"   # separate collection for testing

client = QdrantClient(url="http://localhost:6333")

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=300)

# --- Reset test collection ---
if client.collection_exists(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# --- Process PDF ---
doc = fitz.open(pdf_path)
points = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text")  # or "blocks"/"raw"
    if not text.strip():
        continue  # skip empty pages

    # Split into chunks
    chunks = text_splitter.split_text(text)

    for chunk in chunks:
        points.append(PointStruct(
            id=str(uuid4()),
            vector=embedding_function.embed_documents([chunk])[0],
            payload={
                "file_name": os.path.basename(pdf_path),
                "page_number": page_num,
                "chunk_text": chunk
            }
        ))

# --- Upload to Qdrant ---
if points:
    client.upsert(collection_name=collection_name, wait=True, points=points)
    print(f"✅ Inserted {len(points)} chunks into collection '{collection_name}'")
else:
    print("⚠️ No text chunks extracted from this PDF.")
