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
OLLAMA_MODEL = "gemma3"   

# Initialize embeddings + Qdrant client
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(url="http://localhost:6333")  # adjust if remote

COLLECTION_NAME = "project_air"

def handle_userinput_getdocuments(user_question):
    query_vector = embedding_function.embed_query(user_question)

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=1,
        with_payload=True
    )   

    responsetoquestion = ""
    context_text = ""
    main_ata_code = None
    seen = set()

    for hit in search_result.points:
        payload = hit.payload or {}
        newfilename = payload.get('file_name', 'Unknown')
        newpageNumber = payload.get('pagenumber', 'Unknown')
        ata_code = payload.get('ata_code', 'Unknown')
        header = payload.get('header', 'Unknown')
        tasks = payload.get('task_codes', 'Unknown')
        chunk_text = payload.get("chunk_text", '')
    
        # Capture the main ATA code from the first hit
        if main_ata_code is None and ata_code != "Unknown":
            main_ata_code = ata_code

        key = (newfilename, newpageNumber)
        if key in seen:
            continue
        seen.add(key)

        # Build HTML metadata block
        responsetoquestion += (
            f"<div style='margin-bottom:15px;'>"
            f"<p><b>File:</b> {newfilename}</p>"
            f"<p><b>Header:</b> {header}</p>"
            f"<p><b>ATA Code:</b> {ata_code}</p>"
            f"<p><b>Page Number:</b> {newpageNumber}</p>"
            f"<p><b>Tasks:</b> {tasks}</p>"
            f"<p><b>Score:</b> {hit.score:.4f}</p>"
            f"</div><hr>"
        )

        # Build context for Ollama
        context_text += f"\n[ATA Code: {ata_code}]\n{chunk_text}\n"


    return main_ata_code, context_text, responsetoquestion

def extract_sub_ata_codes(context_text, main_ata_code):
    prompt = f"""
    You are an assistant that extracts ATA sub-codes from aircraft maintenance manuals.

    Main ATA code: {main_ata_code}

    From the following text, list ONLY the sub ATA codes that belong to this main ATA code.
    Output them as a JSON array of strings, nothing else.

    Context:
    {context_text}
    """
    
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )

    if response.status_code == 200:
        raw = response.json().get("response", "").strip()
        try:
            return json.loads(raw)   
        except json.JSONDecodeError:
            return []                
    else:
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

    print(sub_codes)
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
