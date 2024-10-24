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

# Workaround to ensure sqlite3 library works properly in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to initialize the OpenAI client if it's not already initialized
def initialize_openai_client():
    if 'openai_client' not in st.session_state:  # Check if the client is not in session state
        api_key = st.secrets["openai"]  # Fetch API key from Streamlit secrets
        st.session_state.openai_client = OpenAI(api_key=api_key)  # Store the OpenAI client in session state

# Function to extract HTML files from a zip archive
def extract_html_files_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:  # Create a temporary directory to store the extracted files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:  # Open the zip file
            zip_ref.extractall(temp_dir)  # Extract the zip content into the temp directory

        html_files = {}  # Dictionary to hold extracted HTML file contents
        for root, dirs, files in os.walk(temp_dir):  # Traverse the extracted directory
            for file in files:
                if file.endswith('.html'):  # Only process files with .html extension
                    file_path = os.path.join(root, file)  # Get the full path of the HTML file
                    with open(file_path, 'r', encoding='utf-8') as f:  # Open the HTML file in read mode
                        html_files[file] = f.read()  # Read the HTML content and store in the dictionary
    return html_files  # Return the dictionary with HTML filenames and contents

# Function to create a new vector database collection
def create_new_collection():
    if 'Vector_Club_Collection' not in st.session_state:  # Check if collection does not already exist in session state
        # Create the directory for persistent storage of the vector database
        persist_directory = os.path.join(os.getcwd(), "vector_database_storage")
        # Initialize a PersistentClient using the ChromaDB library to persist the collection
        client = chromadb.PersistentClient(path=persist_directory)
        # Get or create a collection for student club URLs in the vector database
        collection = client.get_or_create_collection("Vector_Club_Collection")

        # Path to the zip file containing the HTML documents
        zip_path = os.path.join(os.getcwd(), "student_orgs.zip")
        if not os.path.exists(zip_path):  # Check if the zip file exists
            st.error(f"Zip file not found: {zip_path}")  # Display an error if the file is missing
            return None  # Exit if the file is not found

        html_files = extract_html_files_from_zip(zip_path)  # Extract the HTML files from the zip archive

        if collection.count() == 0:  # Check if the collection is empty
            with st.spinner("Processing documents and creating embeddings..."):
                initialize_openai_client()  # Ensure OpenAI client is initialized

                # Iterate through the extracted HTML files
                for filename, content in html_files.items():
                    try:
                        # Parse the HTML content to extract text using BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)  # Get clean text from the HTML

                        # Generate embeddings using OpenAI API
                        response = st.session_state.openai_client.embeddings.create(
                            input=text, model="text-embedding-ada-002"
                        )
                        embedding = response.data[0].embedding  # Extract embedding from response

                        # Add the document, metadata, and embeddings to the collection
                        collection.add(
                            documents=[text],
                            metadatas=[{"filename": filename}],
                            ids=[filename],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing {filename}: {str(e)}")  # Handle errors gracefully
        else:
            st.info("Using existing vector collection from the database.")  # Notify that the database already exists

        st.session_state.Vector_Club_Collection = collection  # Store the collection in the session state

    return st.session_state.Vector_Club_Collection  # Return the vector collection

# Function to get relevant club information based on a user query
def get_relevant_club_info(query):
    collection = st.session_state.Vector_Club_Collection  # Fetch the collection from session state

    initialize_openai_client()  # Ensure OpenAI client is initialized
    try:
        # Generate embeddings for the user's query using OpenAI API
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding  # Extract the query embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")  # Handle any errors during embedding creation
        return "", []  # Return empty results on error

    # Normalize the embedding vector
    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
        # Query the vector database collection for the most relevant results
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3  # Fetch the top 3 relevant results
        )
        relevant_texts = results['documents'][0]  # Extract relevant document text
        relevant_docs = [result['filename'] for result in results['metadatas'][0]]  # Get the filenames of the results
        return "\n".join(relevant_texts), relevant_docs  # Return the relevant text and document names
    except Exception as e:
        st.error(f"Error querying the vector database: {str(e)}")  # Handle any errors during querying
        return "", []  # Return empty results on error

# Function to interact with the large language model (LLM)
def interact_with_llm(model, messages, temp, query, tools=None):
    initialize_openai_client()  # Ensure OpenAI client is initialized
    try:
        # Call the OpenAI chat completion API with streaming enabled
        response = st.session_state.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            tools=tools,
            tool_choice="auto" if tools else None,  # Automatically choose tools if any are available
            stream=True  # Stream the response as it is generated
        )
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")  # Handle API errors
        return "", "Error occurred while generating response."  # Return error message

    tool_called = None  # Track if any tool was used during the conversation
    full_response = ""  # To accumulate the entire response
    tool_usage_info = ""  # To track any tool usage info

    try:
        while True:
            for chunk in response:  # Loop over the streamed response
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    # If a tool call is detected in the response
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            tool_called = tool_call.function.name  # Capture the tool name
                            if tool_called == "get_club_info":
                                extra_info = get_relevant_club_info(query)  # Fetch extra information
                                tool_usage_info = f"Tool used: {tool_called}"  # Log the tool usage
                                # Update the system prompt with the extra info
                                update_system_prompt(messages, extra_info)
                                # Make a recursive call with the updated system prompt
                                recursive_response, recursive_tool_info = interact_with_llm(
                                    model, messages, temp, tools
                                )
                                full_response += recursive_response  # Add recursive response
                                tool_usage_info += "\n" + recursive_tool_info  # Add tool usage info
                                return full_response, tool_usage_info  # Return full response and tool info
                elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    # Add the content from the response to the full response
                    full_response += chunk.choices[0].delta.content
            break
    except Exception as e:
        st.error(f"Error in streaming response: {str(e)}")  # Handle errors during streaming

    if tool_called:
        tool_usage_info = f"Tool used: {tool_called}"  # Log tool usage
    else:
        tool_usage_info = "No tools were used in generating this response."  # Log if no tools were used

    return full_response, tool_usage_info  # Return the final response and tool usage info

# Function to get a chatbot response for the user's query
def get_chatbot_reply(query, context, conversation_memory):
    # The system message defines the behavior and rules for the chatbot
    system_message = """You are an AI assistant specialized in providing information about student organizations and clubs at Syracuse University. 
    Your primary source of information combines:
    1. Context from vector embeddings of club descriptions and details
    2. Information from newly updated club websites.
    3. Prior chat memory.

    If you don't have enough information, call the external API 'get_club_info' and include relevant club information.
    Be polite and concise in your responses.
    """
    conversation_history = [{"role": "system", "content": system_message}]  # Initialize conversation history

    # Add past user and assistant messages to conversation memory
    for i in range(0, len(conversation_memory), 2):
        conversation_history.append({"role": "user", "content": conversation_memory[i]})
        conversation_history.append({"role": "assistant", "content": conversation_memory[i+1]})

    # Add the new user query
    conversation_history.append({"role": "user", "content": query})

    # Define the LLM model to be used (you can adjust this based on your setup)
    model_name = "gpt-4"

    # Call the interact_with_llm function to get the assistant's response
    response, tool_info = interact_with_llm(
        model=model_name, messages=conversation_history, temp=0.7, query=query, tools=[get_relevant_club_info]
    )

    return response, tool_info

# Streamlit app main logic
def main():
    st.title("Student Club Information Chatbot")  # App title

    # Initialize the vector collection and chatbot memory
    create_new_collection()
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = deque(maxlen=10)  # Store the last 10 conversation exchanges

    query = st.text_input("Ask about student clubs:")  # User input field

    if query:
        # Get chatbot reply and tool usage information
        chatbot_reply, tool_info = get_chatbot_reply(query, context={}, conversation_memory=st.session_state.chat_memory)

        # Display the chatbot reply
        st.write(f"Chatbot: {chatbot_reply}")

        # Display tool usage information (if any)
        st.write(f"{tool_info}")

        # Update the conversation memory with the new exchange
        st.session_state.chat_memory.append(query)
        st.session_state.chat_memory.append(chatbot_reply)

if __name__ == "__main__":
    main()  # Run the main function when the script is executed
