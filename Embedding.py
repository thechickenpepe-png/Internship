import json
import os
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import fitz
import re

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma("Project_AIR", persist_directory="./chroma_air", embedding_function=embedding_function)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=300)

vectorstore.reset_collection()

pdf_files = [r"Project/737-345_AMM/PDF/05___041.PDF"]

def extract_header_and_subheaders(text):
    lines = text.strip().split('\n')

    # Find line with "1." — start of procedural steps
    first_step_index = next((i for i, line in enumerate(lines) if re.match(r'^1\.\s', line)), None)

    if first_step_index is None:
        return None, None  # or fallback values like "", ""

    # Header: lines above "1."
    header = lines[first_step_index - 1] if first_step_index and first_step_index > 0 else None

    # Subheading: line immediately after "1." or after task code
    sub_headers = None
    for i in range(first_step_index, len(lines)):
        if not re.match(r'^1\.\s', lines[i]) and lines[i].strip():
            sub_headers = lines[i]
            break

    return header, sub_headers


def extract_metadata(text, filepath, fallback_page_num):    
    # File name
    file_name = os.path.basename(filepath)

    header, sub_headers = extract_header_and_subheaders(text)

    # Function & Sequential Numbers from 12-digit codes
    task_codes = re.findall(r'\d{2}-\d{2}-\d{2}-\d{3}-\d{3}', text)
    functions = [code.split('-')[3] for code in task_codes]
    sequential_numbers = [code.split('-')[4] for code in task_codes]
    tasks = [f"{code.split('-')[3]}-{code.split('-')[4]}" for code in task_codes]

    # ATA code (6-digit) and its breakdown
    ata_match = re.search(r'\b(\d{2})-(\d{2})-(\d{2})\b', text)
    chapter = ata_match.group(1) if ata_match else None
    section = ata_match.group(2) if ata_match else None
    subject = ata_match.group(3) if ata_match else None
    ata_code = ata_match.group(0) if ata_match else None

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
    }


for filepath in pdf_files:
    doc = fitz.open(filepath)
    file_name = os.path.basename(filepath)
    image_dir = os.path.splitext(file_name)[0]

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        metadata = extract_metadata(text, filepath, page_num)
        chunks = text_splitter.split_text(text)

        documents = [
            Document(
                page_content=chunk,
                metadata=metadata
            )
            for chunk in chunks
        ]

        for doc in documents:
            print(doc.metadata)

        vectorstore.add_documents(documents)

print("Embedded successfully")
