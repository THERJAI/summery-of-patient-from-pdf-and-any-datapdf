from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel
import fitz  # PyMuPDF
import os
import uvicorn
from llama_index.core.node_parser import SentenceSplitter
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origin(s) you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to the PDFs
pdf_paths = {
    1: "Patient_1.pdf",
    2: "Patient_2.pdf",
    3: "Patient_3.pdf",
    4: "Patient_4.pdf"
}

predefined_text = """
give a 7 line of summary and combine this line one paragraph and always end with complete sentence
"""
predefined_text1 = """
recent 5 chief complaint name 
"""
predefined_text2 = """
Give me 5 medication name 
"""

predefined_text3 = """
Give me 5 Diagnosis name with their ICD Code
"""

predefined_text4 = """
Give me 5 Diagnosis test name with their CPT  like CBC (CPT :- 85025)
"""

predefined_text5 = """
Give me Family history if available 
"""

predefined_text6 = """
Give me allergy's name if patient have 
"""

# Initialize global variables
query_engine = None


# Pre-initialize LLM and embeddings
@app.on_event("startup")
async def startup_event():
    global query_engine  # Declare it as global to use it in the endpoint
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI

    # Initialize the LLM model
    llm = HuggingFaceInferenceAPI(model_name='mistralai/Mistral-7B-Instruct-v0.3',
                                  token="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")

    # Set the Hugging Face API token as an environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Initialize the embeddings model
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Load the PDF document
    pdf_path = pdf_paths.get(1)  # Default to the first PDF
    if pdf_path:
        pdf_document = fitz.open(pdf_path)
        pdf_text = "".join([pdf_document.load_page(page_num).get_text() for page_num in range(pdf_document.page_count)])
        document = Document(text=pdf_text)

        # Create an index from the document
        index = VectorStoreIndex.from_documents([document], embed_model=embed_model)
        query_engine = index.as_query_engine()



# Define the Pydantic model for the request body
class PDFRequest(BaseModel):
    pdf_id: int


# Define the API endpoint
@app.post("/print_text/")
async def print_text(request: PDFRequest):
    global query_engine

    # Select the PDF based on the user input
    pdf_path = pdf_paths.get(request.pdf_id)
    if not pdf_path:
        return {"error": "Invalid PDF ID"}

    # Load the PDF document
    pdf_document = fitz.open(pdf_path)
    pdf_text = "".join([pdf_document.load_page(page_num).get_text() for page_num in range(pdf_document.page_count)])
    document = Document(text=pdf_text)

    # Create an index from the document
    index = VectorStoreIndex.from_documents([document], embed_model=Settings.embed_model)
    query_engine = index.as_query_engine()

    # Pass the predefined text to the query engine
    response = query_engine.query(predefined_text)
    response1 = query_engine.query(predefined_text1)
    response2 = query_engine.query(predefined_text2)
    response3 = query_engine.query(predefined_text3)
    response4 = query_engine.query(predefined_text4)
    response5 = query_engine.query(predefined_text5)
    response6 = query_engine.query(predefined_text6)








    # Return the raw response
    return {"response": response.response,
            "response1":response1.response,
            "response2":response2.response,
            "response3":response3.response,
            "response4":response4.response,
            "response5":response5.response,
            "response6": response6.response,

            }


# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
