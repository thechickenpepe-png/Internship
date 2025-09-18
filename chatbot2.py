import google.generativeai as genai
import os
import re
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from transformers import pipeline
import sys
from flask_cors import CORS
sys.stdout.reconfigure(encoding='utf-8')

genai.configure(api_key="AIzaSyBofwV83QUqXk42PejXvr4ZdCYBDpP5w6U")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

def parse_word_limit(user_question):
    match = re.search(r'(\d+)\s*words?', user_question.lower())
    if match:
        return int(match.group(1))
    return None

def gemini_summarize(text, style="short", word_limit=None):
    if style == "word_limit" and word_limit:
        prompt = f"Summarize the following text in about {word_limit} words:\n\n{text}"
    elif style == "short answer":
        prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{text}"
    else:
        prompt = f"Summarize the following text clearly and informatively:\n\n{text}"
    #r
    response = gemini_model.generate_content(prompt)
    return response.text


def handle_userinput_getdocuments(user_question, intent):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        "Project_MCPTT",
        persist_directory="./chroma_mcptt",
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(user_question)

    responsetoquestion = ""
    filename = ""
    pageNumber = ""

    print(f"Found {len(docs)} relevant documents")

    seen = set()
    for text in docs:
        newfilename = text.metadata.get('file_name', 'Unknown')
        newpageNumber = text.metadata.get('pagenumber', 'Unknown')
        key = (newfilename, newpageNumber)
        if key in seen:
            continue
        seen.add(key)

        
        combined_text = " ".join([d.page_content for d in docs])

        if intent == "detailed answer":
            # Just return the raw text from the file
            responsetoquestion += combined_text + "\n <br>"
        else:
            # Use Gemini for summarization/short answers
            summary_text = gemini_summarize(combined_text, intent)
            responsetoquestion += summary_text + "\n <br>"

        print(f"\nRetrieved doc from page {text.metadata.get('pagenumber')}")
        print("Text length:", len(text.page_content))

        if filename != newfilename:
            filename = newfilename
            print("Filename:", filename)
            file_url = f"http://localhost:8000/{filename}"
            responsetoquestion += f"From: <a href='{file_url}' download>{filename}</a>"

        if pageNumber != newpageNumber:
            pageNumber = newpageNumber
            responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

    return responsetoquestion

def get_intent(user_question):
    word_limit = parse_word_limit(user_question)
    if word_limit:
        return "word_limit", word_limit
    
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["short answer", "summary", "detailed answer", "less than 10 words"]
    result = classifier(user_question, candidate_labels=labels)
    return result["labels"][0]  # Top predicted intent

app = Flask(__name__)
load_dotenv()
CORS(app)
@app.route('/chat', methods=['POST'])

def chat():
    data = request.get_json()
    user_question = data.get("question", "")

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    print("User question received:", repr(user_question))
    intent = get_intent(user_question)
    print("Detected intent:", intent)

    response = handle_userinput_getdocuments(user_question, intent)
    return jsonify({"answer": response, "intent": intent})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running. POST to /chat with {'question': '...'}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)