# main_app.py

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from supabase import create_client, Client

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Load .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embeddings (used for retrieval)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vectorstore (retriever)
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# LLM initialization
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B-FP8",   # <-- use this!
    task="text-generation",      # <-- different task
    temperature=0.2,
    max_new_tokens=500,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Streamlit Interface
st.set_page_config(page_title="Supabase RAG App", page_icon="🔍")
st.title("🔍 Supabase RAG (Retrieval Augmented Generation) App")

st.header("Ask your Questions")
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(query)
    st.subheader("Answer:")
    st.write(result)
