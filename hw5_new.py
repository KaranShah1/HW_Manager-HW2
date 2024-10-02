# HW5.py

import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import tiktoken
import zipfile
from io import BytesIO

# Setup OpenAI API
openai_api_key = st.secrets['openai']

# Vector DB Setup
def setup_vectordb():
    db_path = "HW5_VectorDB"
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name="HW5Collection", metadata={"hnsw:space": "cosine", "hnsw:M": 32})
        st.session_state.HW5_vectorDB = collection
        st.success("VectorDB setup complete!")
    else:
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.HW5_vectorDB = client.get_collection(name="HW5Collection")

# Add file content to vector storage
def add_to_vector_storage(collection, text, filename):
    openai_client = OpenAI(api_key=openai_api_key)
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding
    collection.add(documents=[text], ids=[filename], embeddings=[embedding])

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from zip containing HTML
def extract_html_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        html_files = {name: z.read(name).decode("utf-8") for name in z.namelist() if name.endswith('.html')}
    return html_files

# Query the vector DB
def query_vectordb(query, k=3):
    if 'HW5_vectorDB' in st.session_state:
        collection = st.session_state.HW5_vectorDB
        openai_client = OpenAI(api_key=openai_api_key)
        response = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
        query_embedding = response.data[0].embedding
        results = collection.query(query_embeddings=[query_embedding], include=['documents', 'distances'], n_results=k)
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

# Function to interact with LLM based on vector search results
def get_llm_response(query, context):
    openai_client = OpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": context}, {"role": "user", "content": query}]
    response = openai_client.chat.completions.create(model="gpt-4", messages=messages, max_tokens=500)
    return response.choices[0].message.content

# Upload and process files
def handle_file_upload(uploaded_file):
    collection = st.session_state.HW5_vectorDB

    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/zip":
        html_files = extract_html_from_zip(uploaded_file)
        for filename, text in html_files.items():
            add_to_vector_storage(collection, text, filename)
        st.success(f"Added {len(html_files)} HTML files to VectorDB!")
        return
    else:
        st.error("Unsupported file format. Only PDF and ZIP (containing HTML files) are supported.")
        return

    add_to_vector_storage(collection, text, uploaded_file.name)
    st.success(f"File '{uploaded_file.name}' added to VectorDB!")

# Initialize session state for chat history and memory type
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

setup_vectordb()

# Sidebar: Memory selection
st.sidebar.header("Memory Type")
memory_type = st.sidebar.radio("Choose memory type:", ["Short-term memory (5 responses)", "Token-based memory (5,000 tokens)"])

# File Upload
st.header("Upload Files to VectorDB")
uploaded_file = st.file_uploader("Upload PDF or ZIP containing HTML files", type=["pdf", "zip"])
if uploaded_file is not None:
    handle_file_upload(uploaded_file)

# Chat interaction
st.header("Chat with the Assistant")
prompt = st.chat_input("Ask a question based on the uploaded files...")

if prompt:
    # Display user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Perform a vector search
    results = query_vectordb(prompt)
    context = " ".join([doc for doc in results['documents'][0]]) if results else "No specific context found."

    # Prepare messages for LLM
    messages_for_llm = [{"role": "system", "content": context}] + st.session_state.messages

    # Memory management
    if memory_type == "Short-term memory (5 responses)":
        messages_for_llm = messages_for_llm[-11:]  # Keep the last 5 exchanges
    else:
        total_tokens = sum(len(tiktoken.encoding_for_model('gpt-4').encode(msg['content'])) for msg in messages_for_llm)
        while total_tokens > 5000 and len(messages_for_llm) > 1:
            messages_for_llm.pop(0)
            total_tokens = sum(len(tiktoken.encoding_for_model('gpt-4').encode(msg['content'])) for msg in messages_for_llm)

    # Get LLM response
    response = get_llm_response(prompt, context)

    # Display assistant's message
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add system message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
