from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from PyPDF2 import PdfReader
from attr import validators
from dotenv import load_dotenv
from transformers import pipeline
import sys
sys.stdout.reconfigure(encoding='utf-8')

# define the function to retrieve the answer for question posted by the user
def handle_userinput_getdocuments(user_question, intent):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        "Project_MCPTT",
        persist_directory="./chroma_mcptt",
        embedding_function=embedding_function,
        collection_metadata={"hnsw:space": "cosine"}
    )

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
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

        if filename != newfilename:
            filename = newfilename
            print("Filename:", filename)
            file_url = f"http://localhost:8000/{filename}"
            responsetoquestion += f"From: <a href='{file_url}' download>{filename}</a> | "

        if pageNumber != newpageNumber:
            pageNumber = newpageNumber
            responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

        #Branching logic
        if intent == "less than 10 words":
            summary = summarizer(text.page_content, min_length=5, max_length=10)
            summary_text = ' '.join(summary[0]["summary_text"].split()[:10])
            responsetoquestion += summary_text + "\n <br>"

        elif intent == "short answer":
            summary = summarizer(text.page_content, min_length=20, max_length=60)
            responsetoquestion += summary[0]["summary_text"] + "\n <br>"

        elif intent == "detailed answer":
            # Return full text instead of summarizing
            responsetoquestion += text.page_content + "\n <br>"

        else:
            # Fallback: return summary
            summary = summarizer(text.page_content, min_length=30, max_length=150)
            responsetoquestion += summary[0]["summary_text"] + "\n <br>"

        print(f"\nRetrieved doc from page {text.metadata.get('pagenumber')}")
        print("Text length:", len(text.page_content))

    return responsetoquestion


def get_intent(user_question):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["short answer", "summary", "detailed answer", "less than 10 words"]
    result = classifier(user_question, candidate_labels=labels)
    return result["labels"][0]  # Top predicted intent

def main():
    # Load environment variables
    load_dotenv()

    print("Starting chatbot...")

    user_question = "Give me a short answer on aircraft wings"
    print("User question received:", repr(user_question))

    if user_question:
        print("Analyzing intent...")
        intent = get_intent(user_question)
        print("Detected intent:", intent)
        if intent in ["short answer", "summary", "less than 10 words"]:
            response = handle_userinput_getdocuments(user_question, intent)
            print(response)
        else:
            print("Default action for other intents.")
            response = handle_userinput_getdocuments(user_question, intent)
            print(response)
    else:
        print("No question entered.")

if __name__ == "__main__":
    main()



