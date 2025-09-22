import os
import json
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

app = Flask(__name__)
CORS(app)

sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables if needed
load_dotenv()

# Embeddings + Vectorstore
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma("Project_AIR", persist_directory="./chroma_air", embedding_function=embedding_function)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=300)

def handle_userinput_getdocuments(user_question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(user_question)

    print(f"Found {len(docs)} relevant documents")

    responsetoquestion = ""
    seen = set()

    for text in docs:
        newfilename = text.metadata.get('file_name', 'Unknown')
        newpageNumber = text.metadata.get('pagenumber', 'Unknown')
        ata_code = text.metadata.get('ata_code', 'Unknown')
        header = text.metadata.get('header', 'Unknown')
        tasks = text.metadata.get('task_codes', 'Unknown')

        key = (newfilename, newpageNumber)
        if key in seen:
            continue
        seen.add(key)

        # Build HTML response with only metadata
        responsetoquestion += (
            f"<div style='margin-bottom:15px;'>"
            f"<p><b>File:</b> {newfilename}</p>"
            f"<p><b>Header:</b> {header}</p>"
            f"<p><b>ATA Code:</b> {ata_code}</p>"
            f"<p><b>Page Number:</b> {newpageNumber}</p>"
            f"<p><b>Tasks:</b> {tasks}</p>"
            f"</div><hr>"
        )

    return responsetoquestion

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    print("User question received:", repr(user_question))

    response = handle_userinput_getdocuments(user_question)
    return jsonify({"answer": response})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running. POST to /chat with {'question': '...'}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
