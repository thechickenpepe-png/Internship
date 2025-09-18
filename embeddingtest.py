import os
import glob
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import fitz

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma("Project_MCPTT", persist_directory="./chroma_mcptt", embedding_function=embedding_function)
text_splitter = CharacterTextSplitter(chunk_size=650, chunk_overlap=350)

pdf_files = glob.glob("./*.pdf")  # or any folder you want

for filepath in pdf_files:
    doc = fitz.open(filepath)
    file_name = os.path.basename(filepath)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        chunks = text_splitter.split_text(text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "pagenumber": page_num,
                    "file_name": file_name
                }
            )
            for chunk in chunks
        ]
        for doc in documents:
            print(doc.metadata)

        vectorstore.add_documents(documents)

print("Embedded successfully")
