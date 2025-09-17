import gc
import glob
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
SentenceTransformerEmbeddings,
)
from langchain_core.documents import Document
from spire.pdf import *
import pymupdf4llm
import fitz
import pandas as pd

def main():
    load_dotenv()

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

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
        
        #loading as this will plugin will help extarct the tables from pdf
        tableDoc = fitz.open(filepath) 

        # laoding as this will help extract the images from the pdf
        imageDir = os.path.splitext(os.path.basename(filepath))[0]
        image_helper = PdfImageHelper()

        # extarct the text from the pdf for each page with page_chunks true
        output = pymupdf4llm.to_markdown(filepath, page_chunks=True)

        # Loop through the pages in the document
        # I am starting from page number 4 as initial page is just contents/tables
        # I want to ingnore those as it messes the embeddings and gives wrong info
        for i in range(doc.Pages.Count):
            # this is print is just for debugging and tracking the page number
            print(i)
                
            # setting a flag to just confirm if page has table
            hasTable = 0

            # this tableDoc object was defined in previous block using fitz 
            # plugin
            tablePage = tableDoc[i]

            # check if the page has any tabular information
            tables = tablePage.find_tables()

            # define the path where this table will be stored
            pathToStoreTable = f"table\\" + imageDir + "\\" + str(i)
            j = 1

            # extracting and store each table availabe on the page
            for table in tables:
                # Process each table here
                hasTable = 1

                # create the path if it does not exist already
                if not os.path.exists(pathToStoreTable):
                    os.makedirs(pathToStoreTable)

                # create a pickle file name under which the table information
                # will be stored
                pickleTableName = pathToStoreTable+"\\table"+str(j)+".pkl"

                # extract the table in dataframe format from the table.
                df = pd.DataFrame(table.extract())

                # convert the dataframe to pickle with pickleTableName created 
                # two steps above
                df.to_pickle(pickleTableName)
                    
                # This print is just for the debugging purpose to print 
                # the content of the table, you can turn if off if not required
                print(table.extract())

                # move the counter
                j = j + 1

                # extract image information using the image_helper object defined 
                # in Section 1 
                images_info = image_helper.GetImagesInfo(doc.Pages[i])

                # Save images to specified location with specified format extension
            
                # flag to check if the document page has image or not
                hasImage = 0

                for j in range(len(images_info)):
                    hasImage = 1

                    # define the path to store each image on the page
                    # Format for directory is pdf\<PDF_FILE_NAME>\<PAGE_NUMBER>
                    pathToStoreImage = f"pdf\\" + imageDir+"\\"+str(i)

                    # create the path to store the image if does not exist already
                    if not os.path.exists(pathToStoreImage):
                        os.makedirs(f"pdf\\" + imageDir+"\\"+str(i))

                    # extract the image information from the page
                    image_info = images_info[j]

                    # covert the image information in file format
                    output_file = pathToStoreImage+"\\image"+str(j)+".png"
                    
                    # Save the final image file in the image output directory  
                    image_info.Image.Save(output_file)

                    # increment the image counter by 1
                    image_count += 1

                    # get the markdown text for given page number i from the ouput 
                    # variable defined in section 1
                    text = output[i].get("text")

                    # divide the text on each page further into chunks of 500 characters
                    text_chunks = get_text_chunks(text)

                    # store the text chunks in vectore store database with the filepath
                    # of where the image and tables are stored and also flat whether 
                    # page has image or table information or not
                    storetextembedding(text_chunks, vectorstore, filepath, i, hasImage, hasTable)

if __name__ == '__main__':
    main()

# function defined to split the text in chunk size of 500 chars with 
# overlap of 200 chars
def get_text_chunks(text):
    
    # define the textsplitter function
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  #This means thousand characters
        chunk_overlap=200,
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