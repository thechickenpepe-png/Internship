import os
import sys
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)

sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables if needed
load_dotenv()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"   

# Initialize embeddings + Qdrant client
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")  # adjust if remote

COLLECTION_NAME = "project_air"

def handle_userinput_getdocuments(user_question):
    query_vector = embedding_function.embed_query(user_question)

    # Step 1: initial semantic search
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1,
        with_payload=True
    )

    main_ata_code = None
    responsetoquestion = ""
    context_text = ""

    if search_result.points:
        first_payload = search_result.points[0].payload or {}

        # Extract all metadata from the top hit
        newfilename = first_payload.get('file_name', 'Unknown')
        newpageNumber = first_payload.get('pagenumber', 'Unknown')
        ata_code = first_payload.get('ata_code', 'Unknown')
        header = first_payload.get('header', 'Unknown')
        subheader = first_payload.get('subheader', 'Unknown')
        tasks = first_payload.get('task_codes', 'Unknown')
        chunk_text = first_payload.get("chunk_text", '')

        # Derive ATA code from task code if available
        task_code = first_payload.get("task_codes", None)
        if task_code:
            parts = task_code.split("-")                        
            if len(parts) >= 3:
                main_ata_code = "-".join(parts[:3])
                main_ata_code = str(main_ata_code).strip("[]'\" ")
        if not main_ata_code and ata_code != "Unknown":
            main_ata_code = ata_code
            main_ata_code = str(main_ata_code).strip("[]'\" ")

        # Build display for the first page
        responsetoquestion += (
            f"<div style='margin-bottom:15px;'>"
            f"<p><b>File:</b> {newfilename}</p>"
            f"<p><b>Header:</b> {header}</p>"
            f"<p><b>Subheader:</b> {subheader}</p>"
            f"<p><b>ATA Code:</b> {ata_code}</p>"
            f"<p><b>Page Number:</b> {newpageNumber}</p>"
            f"<p><b>Tasks:</b> {tasks}</p>"
            f"</div><hr>"
        )

        # Always add the first page’s chunk_text to context
        context_text += f"\n{chunk_text}\n"

    print("Main ATA Code:", main_ata_code)

    # Step 2: fetch ALL pages with this ATA code
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="ata_code",
                    match=MatchValue(value=main_ata_code)
                )
            ]
        ),
        limit=50,
        with_payload=True
    )
    
    seen = set()
    for hit in points:
        payload = hit.payload or {}
        print("DEBUG:", payload.get("file_name"), payload.get("pagenumber"), payload.get("ata_code"))
        filename = payload.get('file_name', 'Unknown')
        raw_page_number = payload.get('pagenumber', 'Unknown')
        chunk_text = payload.get("chunk_text", '')
        pagenumber = int(raw_page_number)
        key = (filename, pagenumber)
        if key in seen:
            continue
        seen.add(key)

        # Append chunk_text for every page (including first, but deduped by seen)
        if chunk_text:
            context_text += f"\n{chunk_text}\n"

    return main_ata_code, context_text, responsetoquestion

def extract_sub_ata_codes(context_text, main_ata_code):
    print(context_text)
    prompt = f"""
    You are an assistant that extracts ATA sub-codes from aircraft maintenance manuals.

    Context:
    {context_text}

    ATA Codes can appear in two formats:
    1. Codes formatted as DD-DD-DD, where D is a digit.
    2. Codes that include a manual prefix (e.g., AMM or SRM), followed by a space, 
    then a code in the format DD-DD-DD/DDD.

    From the context given, retrieve ALL ATA codes according to the formats above with their uses of it (Keep it to less than 10 words) EXCEPT {main_ata_code} and NOTHING ELSE.
    Respond with a plain JSON array of strings. Do NOT include markdown, code blocks, or any extra formatting.
    """
    
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        raw = response.json().get("response", "").strip()
        # Remove Markdown code block formatting
        if raw.startswith("```json"):
            raw = raw[len("```json"):].strip()
        if raw.endswith("```"):
            raw = raw[:-len("```")].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print("Failed to parse JSON:", raw)
            cleaned = raw.strip('"').strip("'")
            if cleaned:
                return [cleaned]
            return []
    else:
        print("Request failed:", response.status_code, response.text)
        return []


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    # Step 1: Retrieve chunks, ATA code, and metadata HTML
    main_ata_code, context_text, metadata = handle_userinput_getdocuments(user_question)

    if not main_ata_code:
        return jsonify({"error": "No ATA code found in retrieved chunks"}), "Unknown"

    # Step 2: Ask Ollama to extract sub ATA codes
    sub_codes = extract_sub_ata_codes(context_text, main_ata_code)

    # Step 3: Return both metadata and extracted codes
    return jsonify({
        "main_ata_code": main_ata_code,
        "sub_ata_codes": sub_codes,
        "sources": metadata
    })


@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running. POST to /chat with {'question': '...'}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
