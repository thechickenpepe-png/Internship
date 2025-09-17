from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
import fitz

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma("Project_MCPTT", persist_directory="./chroma_mcptt", embedding_function=embedding_function)

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)

doc = fitz.open("Dataset.pdf")
for page_num, page in enumerate(doc, start=1):
    text = page.get_text()
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"page": page_num}) for chunk in chunks]
    vectorstore.add_documents(documents)

vectorstore.persist()
print("✅ Embedded successfully")
