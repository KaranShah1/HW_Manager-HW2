import sys
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import os
import zipfile
import tempfile
from collections import deque
import numpy as np
import json

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Ensure the OpenAI client is initialized
def initialize_openai():
    if 'ai_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.ai_client = OpenAI(api_key=api_key)

# Extract HTML files from the zip archive
def unzip_html_files(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        html_data = {}
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_data[file] = f.read()
    return html_data

# Create a ChromaDB collection for storing embeddings
def generate_collection():
    if 'url_collection' not in st.session_state:
        persist_directory = os.path.join(os.getcwd(), "chroma_storage")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("URL_Collection")

        zip_file_path = os.path.join(os.getcwd(), "university_orgs.zip")
        if not os.path.exists(zip_file_path):
            st.error(f"Zip file not found: {zip_file_path}")
            return None

        html_files = unzip_html_files(zip_file_path)

        if collection.count() == 0:
            with st.spinner("Extracting content and preparing embeddings..."):
                initialize_openai()

                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = soup.get_text(separator=' ', strip=True)

                        response = st.session_state.ai_client.embeddings.create(
                            input=text_content, model="text-embedding-ada-002"
                        )
                        embedding = response.data[0].embedding

                        collection.add(
                            documents=[text_content],
                            metadatas=[{"filename": filename}],
                            ids=[filename],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")
        else:
            st.info("Using existing vector collection.")

        st.session_state.url_collection = collection

    return st.session_state.url_collection

# Fetch relevant club information from the collection
def fetch_club_info(query):
    collection = st.session_state.url_collection

    initialize_openai()
    try:
        response = st.session_state.ai_client.embeddings.create(
            input=query, model="text-embedding-ada-002"
        )
        query_vector = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating AI embedding: {str(e)}")
        return "", []

    # Normalize the embedding vector
    query_vector = np.array(query_vector) / np.linalg.norm(query_vector)

    try:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )
        relevant_texts = results['documents'][0]
        doc_files = [result['filename'] for result in results['metadatas'][0]]
        return "\n".join(relevant_texts), doc_files
    except Exception as e:
        st.error(f"Error querying collection: {str(e)}")
        return "", []

# Call the LLM model for chat response generation
def generate_llm_response(model, messages, temp, query, tools=None):
    initialize_openai()
    try:
        response = st.session_state.ai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            tools=tools,
            tool_choice="auto" if tools else None,
            stream=True
        )
    except Exception as e:
        st.error(f"Error in API call: {str(e)}")
        return "", "Error occurred."

    full_text = ""
    tool_called = None
    tool_info = ""

    try:
        while True:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            tool_called = tool_call.function.name
                            if tool_called == "fetch_club_info":
                                extra_info = fetch_club_info(query)
                                tool_info = f"Tool used: {tool_called}"
                                update_prompt(messages, extra_info)
                                recursive_response, recursive_tool_info = generate_llm_response(
                                    model, messages, temp, tools)
                                full_text += recursive_response
                                tool_info += "\n" + recursive_tool_info
                                return full_text, tool_info
                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
            break
    except Exception as e:
        st.error(f"Error streaming response: {str(e)}")

    if tool_called:
        tool_info = f"Tool used: {tool_called}"
    else:
        tool_info = "No tools used."

    return full_text, tool_info

# Update the system message prompt with additional information
def update_prompt(messages, extra_info):
    for message in messages:
        if message["role"] == "system":
            message["content"] += f"\n\nAdditional data: {extra_info}"
            break

# Main chat system for querying documents
def main_chat_system():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory_queue' not in st.session_state:
        st.session_state.memory_queue = deque(maxlen=5)
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

    st.title("University Org Chatbot")

    if not st.session_state.system_initialized:
        with st.spinner("Processing and setting up..."):
            st.session_state.url_collection = generate_collection()
            if st.session_state.url_collection:
                st.session_state.system_initialized = True
                st.success("Chatbot is ready!")
            else:
                st.error("Failed to initialize system. Check the zip file.")

    if st.session_state.system_initialized and st.session_state.url_collection:
        st.subheader("Chat with the AI Assistant (GPT-4 Powered)")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask a question:")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            relevant_texts, doc_files = fetch_club_info(user_input)
            st.write(f"Debug: {len(relevant_texts)} characters found.")

            response, tool_usage_info = generate_llm_response(
                "gpt-4", user_input, 0.7, relevant_texts)

            if response is None:
                st.error("Failed to get a response from the AI.")
                return

            with st.chat_message("assistant"):
                st.markdown(response)
                st.info(tool_usage_info)

            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            st.session_state.memory_queue.append({
                "question": user_input,
                "answer": response
            })

            with st.expander("Documents referenced"):
                for doc in doc_files:
                    st.write(f"- {doc}")

    elif not st.session_state.system_initialized:
        st.info("The system is setting up. Please wait...")

if __name__ == "__main__":
    main_chat_system()
