import gc
import glob
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from spire.pdf import *
import pymupdf4llm
import fitz
import pandas as pd

def main():
    load_dotenv()

    # create the open-source embedding function
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Define your vectore store with the name "Project_NAME"
    vectorstore = Chroma("Project_NAME", embedding_function, "./chroma_name",collection_metadata={"hnsw:space": "cosine"})
    
    # defining a counter for images
    image_count = 0

    # pick all the pdfs from the local folder to be eembedded
    pdf_names = list(glob.glob("pdf\\*.pdf"))

    for filepath in pdf_names:
        # each pdf name will create a file path with its own name to store a
        # tables and images 
        print(filepath) 

        # clear the garbage collection
        gc.collect() 

        # Create a PdfDocument object
        doc = PdfDocument()

        # Load a PDF document
        # the below object will be used in below code to extract page 
        # level information
        doc.LoadFromFile(filepath) 

if __name__ == '__main__':
    main()

# function defined to split the text in chunk size of 500 chars with 
# overlap of 200 chars
def get_text_chunks(text):
    
    # define the textsplitter function
    text_splitter = CharacterTextSplitter(
        chunk_size=400,  #This means thousand characters
        chunk_overlap=300,
        length_function=len,
        separator="\n"
    )
    
    # create chunks based on the above text splitter function    
    chunks = text_splitter.split_text(text)

    # return the collection of chunks
    return chunks

#funtion to store the text embedding of collection of chunks
def storetextembedding(text_chunks, vectorstore, file, pagenumber, hasImage, hasTable):

    loadingdocuments =  []

    # create a collection of documents to be loaded in vectore store
    for text in text_chunks:
        # define the doucment with text, and metadata with filename information
        # page number to which this chunk belongs
        # whether this page has image and table information in the form of flags
        # "imagePresent": hasImage, "tablePresent":hasTable        
        document_1 = Document(page_content=text, metadata={"file_name": file, "pagenumber": pagenumber, "imagePresent": hasImage, "tablePresent":hasTable})
        loadingdocuments.append(document_1)

    # load the entire collection of documents of collection of chunks of text 
    # on a page in the vectore store
    if len(loadingdocuments) != 0:
        vectorstore.add_documents(documents=loadingdocuments)

    vectorstore.persist()

    # retur the file name
    return file