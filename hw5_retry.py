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

# Workaround for sqlite3 issue in Streamlit Cloud (pysqlite3 is used here to avoid compatibility issues with sqlite3)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to initialize the OpenAI client with the API key
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to extract HTML files from a zip file
# The zip file contains HTML pages of student clubs and organizations
def extract_html_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract all files in the zip into a temporary directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        html_files = {}
        # Walk through the directory and find all HTML files
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.html'):
                    # Read the HTML file content and store it in the html_files dictionary
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_files[file] = f.read()
    return html_files

# Function to create or load the ChromaDB collection for embedding documents
def create_hw4_collection():
    if 'HW_URL_Collection' not in st.session_state:
        # Define the directory where the vector database (ChromaDB) will be stored
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        # Create or get the collection for storing club-related embeddings
        collection = client.get_or_create_collection("HW_URL_Collection")

        # Path to the zip file containing the HTML documents of clubs and organizations
        zip_path = os.path.join(os.getcwd(), "su_orgs.zip")
        if not os.path.exists(zip_path):
            st.error(f"Zip file not found: {zip_path}")
            return None

        # Extract the HTML files from the zip file
        html_files = extract_html_from_zip(zip_path)

        # If the collection is empty, process the content and add it to the collection
        if collection.count() == 0:
            with st.spinner("Processing content and preparing the system..."):
                ensure_openai_client()

                # Process each HTML file
                for filename, content in html_files.items():
                    try:
                        # Use BeautifulSoup to extract the text from the HTML
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)

                        # Generate the embedding using OpenAI's embedding model
                        response = st.session_state.openai_client.embeddings.create(
                            input=text, model="text-embedding-3-small"
                        )
                        embedding = response.data[0].embedding

                        # Add the text, metadata, and embedding to the ChromaDB collection
                        collection.add(
                            documents=[text],
                            metadatas=[{"filename": filename}],
                            ids=[filename],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")
        else:
            st.info("Using existing vector database.")

        # Store the collection in session state
        st.session_state.HW_URL_Collection = collection

    return st.session_state.HW_URL_Collection

# Function to retrieve relevant club information based on a user's query
# It returns matching documents and filenames from the ChromaDB collection
def get_relevant_info(query):
    collection = st.session_state.HW_URL_Collection
    ensure_openai_client()

    try:
        # Create an embedding for the user's query using OpenAI
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

    # Normalize the query embedding (convert it into a unit vector)
    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
        # Query the ChromaDB collection using the query embedding
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3  # Retrieve top 3 relevant documents
        )
        # Extract the text and filenames of relevant documents
        relevant_texts = results['documents'][0]
        relevant_docs = [result['filename'] for result in results['metadatas'][0]]
        return "\n".join(relevant_texts), relevant_docs
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return "", []

# Function to call the language model (LLM) with user messages and handle tool invocation
def call_llm(model, messages, temp, query, tools=None):
    ensure_openai_client()
    try:
        # Make a request to OpenAI to generate a response, specifying the model and messages
        response = st.session_state.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            tools=tools,
            tool_choice="auto" if tools else None,
            stream=True
        )
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return "", "Error occurred while generating response."

    tool_called = None
    full_response = ""
    tool_usage_info = ""

    # Process the streaming response and detect tool invocations
    try:
        while True:
            for chunk in response:
                # Check if the LLM called a tool (function)
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            tool_called = tool_call.function.name
                            # If the tool `get_club_info` is called, invoke the relevant function
                            if tool_called == "get_club_info":
                                extra_info = get_relevant_info(query)
                                tool_usage_info = f"Tool used: {tool_called}"
                                update_system_prompt(messages, extra_info)
                                # Recursively call LLM after getting relevant information
                                recursive_response, recursive_tool_info = call_llm(
                                    model, messages, temp, tools)
                                full_response += recursive_response
                                tool_usage_info += "\n" + recursive_tool_info
                                return full_response, tool_usage_info
                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            break
    except Exception as e:
        st.error(f"Error in streaming response: {str(e)}")

    if tool_called:
        tool_usage_info = f"Tool used: {tool_called}"
    else:
        tool_usage_info = "No tools were used in generating this response."

    return full_response, tool_usage_info

# Function to generate a chatbot response based on user input
def get_chatbot_response(query, context, conversation_memory):
    system_message = """You are an AI assistant specialized in providing information about student organizations and clubs at Syracuse University. 
    Your primary source of information is the context provided, which contains relevant data extracted from embeddings of club descriptions and details.

    Only use the get_club_info tool when:

    a) A specific club name is mentioned in the user's query, OR
    b) If the user asks a follow-up question about a specific club mentioned in a previous response and this could be at any point in the chat, then find the club name from the previous response and pass it as an argument.

    Always prioritize using the context for general inquiries about clubs or types of clubs."""

    # Combine recent conversation history to maintain continuity in the conversation
    condensed_history = "\n".join(
        [f"Human: {exchange['question']}\nAI: {exchange['answer']}" for exchange in conversation_memory]
    )

    # Prepare the messages to send to the LLM, including system message, user input, and context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context}\n\nConversation history:\n{condensed_history}\n\nQuestion: {query}"}
    ]

    # Define available tools (functions) for the LLM to call, such as `get_club_info`
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_club_info",
                "description": "Get information about a specific club or organization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "club_name": {
                            "type": "string",
                            "description": "The name of the club or organization to look up"
                        }
                    },
                    "required": ["club_name"]
                }
            }
        }
    ]

    # Call the LLM to get a response based on the user input
    return call_llm(model="gpt-4", messages=messages, temp=0.8, query=query, tools=tools)

