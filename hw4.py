import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader


# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


# Function to ensure the OpenAI client is initialized
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        # Get the API key from Streamlit secrets
        api_key = st.secrets["openai"]
        # Initialize the OpenAI client and store it in session state
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to create the ChromaDB collection
def create_lab4_collection():
    if 'Lab4_vectorDB' not in st.session_state:
        # Set up the ChromaDB client
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("Lab4Collection")

        ensure_openai_client()

        # Define the directory containing the PDF files
        pdf_dir = os.path.join(os.getcwd(), "Lab4_datafiles")
        if not os.path.exists(pdf_dir):
            st.error(f"Directory not found: {pdf_dir}")
            return None

        # Process each PDF file in the directory
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                filepath = os.path.join(pdf_dir, filename)
                try:
                    # Extract text from the PDF
                    with open(filepath, "rb") as file:
                        pdf_reader = PdfReader(file)
                        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])

                    # Generate embeddings for the extracted text
                    response = st.session_state.openai_client.embeddings.create(
                        input=text, model="text-embedding-3-small"
                    )
                    embedding = response.data[0].embedding

                    # Add the document to ChromaDB
                    collection.add(
                        documents=[text],
                        metadatas=[{"filename": filename}],
                        ids=[filename],
                        embeddings=[embedding]
                    )
                except Exception as e:
                    st.error(f"Error processing {filename}: {str(e)}")

        # Store the collection in session state
        st.session_state.Lab4_vectorDB = collection

    return st.session_state.Lab4_vectorDB

# Function to query the vector database
def query_vector_db(collection, query):
    ensure_openai_client()
    try:
        # Generate embedding for the query
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Query the ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return results['documents'][0], [result['filename'] for result in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return [], []

# Function to get chatbot response using OpenAI's GPT model
def get_chatbot_response(query, context):
    ensure_openai_client()
    # Construct the prompt for the GPT model
    prompt = f"""You are an AI assistant with knowledge from specific documents. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information.

Context:
{context}

User Question: {query}

Answer:"""

    try:
        # Generate streaming response using OpenAI's chat completion
        response_stream = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o",  # Using the latest GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True  # Enable streaming
        )
        return response_stream
    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return None

# Initialize session state for chat history, system readiness, and collection
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None

# Page content
st.title("Lab 4 - Document Chatbot")

# Check if the system is ready, if not, prepare it
if not st.session_state.system_ready:
    # Show a spinner while processing documents
    with st.spinner("Processing documents and preparing the system..."):
        st.session_state.collection = create_lab4_collection()
        if st.session_state.collection:
            # Set the system as ready and show a success message
            st.session_state.system_ready = True
            st.success("AI ChatBot is Ready!!!")
        else:
            st.error("Failed to create or load the document collection. Please check the file path and try again.")

# Only show the chat interface if the system is ready
if st.session_state.system_ready and st.session_state.collection:
    st.subheader("Chat with the AI Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, dict):
            # New format (dictionary with 'role' and 'content' keys)
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif isinstance(message, tuple):
            # Old format (tuple with role and content)
            role, content = message
            # Convert 'You' to 'user', and assume any other role is 'assistant'
            with st.chat_message("user" if role == "You" else "assistant"):
                st.markdown(content)

    # User input
    user_input = st.chat_input("Ask a question about the documents:")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Query the vector database
        relevant_texts, relevant_docs = query_vector_db(st.session_state.collection, user_input)
        context = "\n".join(relevant_texts)

        # Get streaming chatbot response
        response_stream = get_chatbot_response(user_input, context)

        # Display AI response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        # Add to chat history (new format)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        # Display relevant documents
        with st.expander("Relevant documents used"):
            for doc in relevant_docs:
                st.write(f"- {doc}")

elif not st.session_state.system_ready:
    st.info("The system is still preparing. Please wait...")
else:
    st.error("Failed to create or load the document collection. Please check the file path and try again.")
