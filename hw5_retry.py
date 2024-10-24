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

# Function to ensure the OpenAI client is initialized
def initialize_openai_client():
    if 'openai_api_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.openai_api_client = OpenAI(api_key=api_key)

# Function to extract HTML files from a zip archive
def unpack_html_from_zip(zip_archive_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_archive_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        html_data = {}
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_data[file] = f.read()
    return html_data

# Function to create the ChromaDB collection
def generate_chroma_collection():
    if 'document_vector_collection' not in st.session_state:
        storage_directory = os.path.join(os.getcwd(), "chroma_db_storage")
        client = chromadb.PersistentClient(path=storage_directory)
        vector_collection = client.get_or_create_collection("document_vector_collection")

        zip_archive_path = os.path.join(os.getcwd(), "su_orgs.zip")
        if not os.path.exists(zip_archive_path):
            st.error(f"Zip file not found: {zip_archive_path}")
            return None

        html_files = unpack_html_from_zip(zip_archive_path)

        if vector_collection.count() == 0:
            with st.spinner("Processing content..."):
                initialize_openai_client()

                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text_content = soup.get_text(separator=' ', strip=True)

                        response = st.session_state.openai_api_client.embeddings.create(
                            input=text_content, model="text-embedding-ada-002"
                        )
                        embedding = response.data[0].embedding

                        vector_collection.add(
                            documents=[text_content],
                            metadatas=[{"filename": filename}],
                            ids=[filename],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")
        else:
            st.info("Using the existing ChromaDB collection.")

        st.session_state.document_vector_collection = vector_collection

    return st.session_state.document_vector_collection

# Function to find relevant document information
def find_club_details(user_query):
    collection = st.session_state.document_vector_collection

    initialize_openai_client()
    try:
        response = st.session_state.openai_api_client.embeddings.create(
            input=user_query, model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return "", []

    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        relevant_text = results['documents'][0]
        relevant_docs = [result['filename'] for result in results['metadatas'][0]]
        return "\n".join(relevant_text), relevant_docs
    except Exception as e:
        st.error(f"Error querying the collection: {str(e)}")
        return "", []

# Function to communicate with the LLM model
def query_large_language_model(model, chat_log, temperature, query_text, toolset=None):
    initialize_openai_client()
    try:
        response = st.session_state.openai_api_client.chat.completions.create(
            model=model,
            messages=chat_log,
            temperature=temperature,
            tools=toolset,
            tool_choice="auto" if toolset else None,
            stream=True
        )
    except Exception as e:
        st.error(f"Error using the OpenAI API: {str(e)}")
        return "", "Error generating response."

    invoked_tool = None
    full_response = ""
    tool_info = ""

    try:
        while True:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            invoked_tool = tool_call.function.name
                            if invoked_tool == "fetch_club_info":
                                extra_details = find_club_details(query_text)
                                tool_info = f"Tool used: {invoked_tool}"
                                revise_system_prompt(chat_log, extra_details)
                                recursive_response, recursive_tool_usage = query_large_language_model(
                                    model, chat_log, temperature, toolset)
                                full_response += recursive_response
                                tool_info += "\n" + recursive_tool_usage
                                return full_response, tool_info
                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            break
    except Exception as e:
        st.error(f"Error in streaming response: {str(e)}")

    if invoked_tool:
        tool_info = f"Tool used: {invoked_tool}"
    else:
        tool_info = "No tools were invoked during this interaction."

    return full_response, tool_info

# Function to handle chatbot responses
def handle_chatbot_response(user_query, context_data, memory_log):
    base_system_message = """You are a virtual assistant designed to assist with information about student clubs and organizations at Syracuse University. 
    Your main information sources include:
    1. Contextual knowledge from vector embeddings of club data
    2. Content from newly uploaded documents
    3. Prior conversation history to help with follow-up queries

    When addressing questions, please follow these guidelines:
    1. Use the 'fetch_club_info' tool only if:
        a) A specific club name is mentioned, OR
        b) A follow-up inquiry references a previously mentioned club from the chat history.
    2. For general club inquiries:
        a) Rely on context from previous exchanges.
        b) Combine insights from documents and chat history.
    3. Always provide clarification when queries are unclear.
    Provide club lists in bullet points."""

    conversation_history = "\n".join(
        [f"User: {entry['question']}\nAssistant: {entry['answer']}" for entry in memory_log]
    )

    messages = [
        {"role": "system", "content": base_system_message},
        {"role": "user", "content": f"Context: {context_data}\n\nConversation log:\n{conversation_history}\n\nQuery: {user_query}"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_club_info",
                "description": "Retrieve details of a specific club or organization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "club_name": {
                            "type": "string",
                            "description": "The club or organization to search for"
                        }
                    },
                    "required": ["club_name"]
                }
            }
        }
    ]

    try:
        response, tool_usage = query_large_language_model(
            "gpt-4o", messages, 0.7, user_query, tools)
        return response, tool_usage
    except Exception as e:
        st.error(f"Error processing chatbot response: {str(e)}")
        return None, "Error generating chatbot response."

# Function to update the system prompt with additional info
def revise_system_prompt(chat_log, additional_info):
    for message in chat_log:
        if message["role"] == "system":
            message["content"] += f"\n\nExtra details: {additional_info}"
            break

# Streamlit app main function
def app_main():
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []
    if 'memory_log' not in st.session_state:
        st.session_state.memory_log = deque(maxlen=5)
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = False
    if 'vector_collection' not in st.session_state:
        st.session_state.vector_collection = None

    st.title("iSchool AI Assistant")

    if not st.session_state.app_ready:
        with st.spinner("Preparing documents and system..."):
            st.session_state.vector_collection = generate_chroma_collection()
            if st.session_state.vector_collection:
                st.session_state.app_ready = True
                st.success("System is ready!")
            else:
                st.error("Could not initialize the document collection. Check the zip file and try again.")

    if st.session_state.app_ready and st.session_state.vector_collection:
        st.subheader("Ask a Question")

        for message in st.session_state.chat_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_message = st.chat_input("Ask about clubs at Syracuse University:")

        if user_message:
            with st.chat_message("user"):
                st.markdown(user_message)

            relevant_text, relevant_docs = find_club_details(user_message)
            ai_response, tool_info = handle_chatbot_response(
                user_message, relevant_text, st.session_state.memory_log)

            if ai_response:
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                st.session_state.chat_log.append({"role": "user", "content": user_message})
                st.session_state.chat_log.append({"role": "assistant", "content": ai_response})
                st.session_state.memory_log.append({"question": user_message, "answer": ai_response})

            if tool_info:
                st.info(tool_info)

if __name__ == '__main__':
    app_main()
