# uploader.py (fixed)

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client, Client

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Huggingface Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Supabase Vectorstore
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Folder where your PDFs are stored
pdf_folder = "Your pdf folder path"

# Loop over PDFs and upload
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        pages = loader.load()

        # Remove NULL characters from all page contents
        cleaned_pages = []
        for page in pages:
            page.page_content = page.page_content.replace('\u0000', '')
            cleaned_pages.append(page)

        # Split cleaned text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.split_documents(cleaned_pages)

        # Store embeddings into Supabase
        vectorstore.add_documents(documents)

print("✅ All PDF documents processed and uploaded to Supabase!")
