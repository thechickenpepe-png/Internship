import base64
import glob
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import requests
from PyPDF2 import PdfReader
from attr import validators
from dotenv import load_dotenv
from transformers import pipeline
import sys
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')

# define the function to retrieve the answer for question posted by the user
def handle_userinput_getdocuments(user_question):
    # define the embedding function, this should be same as the 
    # one defined while creating the embedding
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # define the verctore store. this should be also be the same in which 
    # the embedding was stored.
    vectorstore = Chroma("Project_MCPTT", persist_directory="./chroma_mcptt", embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    # create a pipelin with summarization effect on each of the retrieved 
    # document  
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Retrieve and generate using the relevant snippets of the documents
    retriever = vectorstore.as_retriever()

    # retrieve the documets using the retriever of the vectore store.
    docs = retriever.get_relevant_documents(user_question)

    responsetoquestion = ""
    filename = ""
    pageNumber = ""

    print(f"Found {len(docs)} relevant documents")

    # traverse through each document retrieved by the retriever.

    seen = set()
    for text in docs:
        newfilename = text.metadata.get('file_name', 'Unknown')
        newpageNumber = text.metadata.get('pagenumber', 'Unknown')
        key = (newfilename, newpageNumber)
        if key in seen:
            continue
        seen.add(key)
        
        # if filename is not asame as the previous document, treat it as new 
        # filename and append a link to it to the answer response
        if filename != newfilename:
           filename = newfilename
           print("Filename:",filename)
           file_url = f"http://localhost:8000/{filename}"
           responsetoquestion += f"From: <a href='{file_url}' download>{filename}</a> | "
        
        # if page number is not asame as the previous pagenumber, 
        # treat it as new pagenumber
        # and append a link to it to the answer response
        if pageNumber != newpageNumber:
           pageNumber = newpageNumber
           responsetoquestion += f"<p style='color:Tomato;'>Page Number: {pageNumber} </p>"

        # create summary of response text retrieved from the page
        input_length = len(text.page_content.split())
        max_length = min(50, input_length)  # Use 50 or less than input length
        summary = summarizer(text.page_content, min_length=5, max_length=max_length)
        # append summary of text to the response
        responsetoquestion = responsetoquestion + summary[0]["summary_text"] + "\n <br>"

    print(f"\nRetrieved doc from page {text.metadata.get('pagenumber')}")
    print("Text length:", len(text.page_content))
    print("Text preview:", text.page_content[:100])
        
    return responsetoquestion

def main():
    # Load environment variables
    load_dotenv()

    print("Starting chatbot...")

    user_question = "Give me a summary of the airplane wings"
    print("User question received:", repr(user_question))

    if user_question:
        print("Calling document handler...")
        response = handle_userinput_getdocuments(user_question)
        print(response)
    else:
        print("No question entered.")

if __name__ == "__main__":
    main()



