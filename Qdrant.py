import json
import os
import re
from uuid import uuid4
from langchain_text_splitters import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from langchain_huggingface import HuggingFaceEmbeddings
import fitz  

client = QdrantClient(url="http://localhost:6333")

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=300)

pdf_files = [r"Project/737-345_AMM/PDF/05___041.PDF"]

client.delete_collection(collection_name="project_air")
client.create_collection(
    collection_name="project_air",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

def extract_header_and_subheaders(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header, sub_header = None, None

    def is_header_line(line):
        letters = [c for c in line if c.isalpha()]
        return letters and sum(c.isupper() for c in letters) / len(letters) > 0.8

    # Ignore boilerplate lines like "MAINTENANCE MANUAL"
    ignore_phrases = ["MAINTENANCE MANUAL"]

    # Find first procedural step
    first_step_index = next((i for i, l in enumerate(lines) if re.match(r'^\d+\.\s', l)), len(lines))
    candidates = lines[:min(first_step_index, 5)]

    for i, line in enumerate(candidates):
        if any(phrase in line.upper() for phrase in ignore_phrases):
            continue  # skip this line

        if is_header_line(line) and len(line) > 15:
            header = line
            if i + 1 < len(candidates) and not is_header_line(candidates[i+1]):
                sub_header = candidates[i+1]
            break

    return header, sub_header

def extract_metadata(text, filepath, fallback_page_num):    
    # File name
    file_name = os.path.basename(filepath)

    header, sub_headers = extract_header_and_subheaders(text)

    # Function & Sequential Numbers from 12-digit codes
    task_codes = re.findall(r'\d{2}-\d{2}-\d{2}-\d{3}-\d{3}', text)

    if task_codes:
        ata_code = "-".join(task_codes[0].split('-')[:3])
        chapter, section, subject = task_codes[0].split('-')[:3]
    else:
        ata_match = re.search(r'\b(\d{2}-\d{2}-\d{2})\b', text)
        ata_code = ata_match.group(1) if ata_match else None
        if ata_code:
            chapter, section, subject = ata_code.split('-')
        else:
            chapter = section = subject = None

    functions = [code.split('-')[3] for code in task_codes]
    sequential_numbers = [code.split('-')[4] for code in task_codes]
    tasks = [f"{code.split('-')[3]}-{code.split('-')[4]}" for code in task_codes]

    # Page number and date
    page_match = re.search(r'Page\s+(\d+)', text)
    date_match = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}/\d{2}\b', text)

    page_num = int(page_match.group(1)) if page_match else fallback_page_num
    date = date_match.group(0) if date_match else None

    return {
    "file_name": file_name,
    "header": header,
    "sub_headers": " | ".join(sub_headers) if sub_headers else None,
    "functions": json.dumps(functions) if functions else None,
    "sequential_numbers": json.dumps(sequential_numbers) if sequential_numbers else None,
    "tasks": json.dumps(tasks) if tasks else None,
    "task_codes": json.dumps(task_codes) if task_codes else None,
    "chapter": chapter,
    "section": section,
    "subject": subject,
    "ata_code": ata_code,
    "pagenumber": page_num,
    "date": date,
    "chunk_text": text  # placeholder, will be filled later
    }

for filepath in pdf_files:
    doc = fitz.open(filepath)
    file_name = os.path.basename(filepath)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        header, sub_headers = extract_header_and_subheaders(text)
        metadata = extract_metadata(text, filepath, page_num)
        chunks = text_splitter.split_text(text)

        points = []
        if header:
            points.append(PointStruct(
                id=str(uuid4()),
                vector=embedding_function.embed_query(header),
                payload={**metadata, "header": header}   # key is "header"
            ))

        if sub_headers:
            points.append(PointStruct(
                id=str(uuid4()),
                vector=embedding_function.embed_query(sub_headers),
                payload={**metadata, "sub_headers": sub_headers}  # key is "sub_headers"
            ))

        for chunk in chunks:
            points.append(PointStruct(
                id=str(uuid4()),
                vector=embedding_function.embed_documents([chunk])[0],  # better for docs
                payload={**metadata, "chunk_text": chunk}  # key is "chunk_text"
            ))

        operation_info = client.upsert(collection_name="project_air", wait=True, points=points)
        print(operation_info)