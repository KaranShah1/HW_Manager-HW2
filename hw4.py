import sys
import streamlit as st
import os
import zipfile
import tempfile
from collections import deque
from bs4 import BeautifulSoup

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import libraries for AI clients
import openai
import google.generativeai as genai
import chromadb
import cohere  # Correct import for Cohere

# Helper Functions for Client Initialization
def initialize_clients():
    """Ensure all required AI clients are initialized."""
    if 'openai_client' not in st.session_state:
        openai.api_key = st.secrets["openai"]
        st.session_state.openai_client = openai
    if 'google_ai_client' not in st.session_state:
        genai.configure(api_key=st.secrets["gemini"])
        st.session_state.google_ai_client = genai
    if 'cohere_client' not in st.session_state:
        api_key = st.secrets["cohere"]
        st.session_state.cohere_client = cohere.Client(api_key)  # Corrected client initialization

# Function to Extract HTML Files from ZIP
def extract_html_from_zip(zip_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        html_files = {}
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.html'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_files[file] = f.read()
    return html_files

# Function to Create ChromaDB Collection
def create_hw4_collection():
    if 'HW_URL_Collection' not in st.session_state:
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("HW_URL_Collection")

        zip_path = os.path.join(os.getcwd(), "su_orgs.zip")
        if not os.path.exists(zip_path):
            st.error(f"Zip file not found: {zip_path}")
            return None

        html_files = extract_html_from_zip(zip_path)

        if collection.count() == 0:
            with st.spinner("Processing content and preparing the system..."):
                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)
                        response = st.session_state.openai_client.Embedding.create(input=text)
                        embedding = response['data'][0]['embedding']
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
        st.session_state.HW_URL_Collection = collection
    return st.session_state.HW_URL_Collection

# Function to Query Vector Database
def query_vector_db(collection, query):
    try:
        response = st.session_state.openai_client.Embedding.create(input=query)
        query_embedding = response['data'][0]['embedding']

        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        return results['documents'][0], [result['filename'] for result in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return [], []

# Function to Get Chatbot Response Based on Selected Model
def get_chatbot_response(query, context, conversation_memory, selected_model):
    condensed_history = "\n".join([f"Human: {item['question']}\nAI: {item['answer']}" for item in conversation_memory])

    prompt = f"Context: {context}\nConversation history:\n{condensed_history}\nHuman: {query}\nAI:"
    
    if selected_model == "OpenAI GPT-4":
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = st.session_state.openai_client.ChatCompletion.create(model="gpt-4", messages=messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            st.error(f"Error with GPT-4: {str(e)}")
    
    elif selected_model == "Cohere":
        try:
            response = st.session_state.cohere_client.generate(prompt=prompt, model="claude-3-opus-20240229", max_tokens=1024)
            return response.text
        except Exception as e:
            st.error(f"Error with Cohere: {str(e)}")
    
    elif selected_model == "Google Gemini":
        try:
            model = st.session_state.google_ai_client.GenerativeModel('gemini-1.0-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error with Gemini: {str(e)}")

# Main Streamlit App
def main():
    initialize_clients()

    # Initialize session state for chat and system
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = deque(maxlen=5)
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'collection' not in st.session_state:
        st.session_state.collection = None

    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.radio("Choose an LLM:", ("OpenAI GPT-4", "Cohere", "Google Gemini"))

    st.title("HW 4 - iSchool Chatbot")

    # System Preparation
    if not st.session_state.system_ready:
        with st.spinner("Preparing system..."):
            st.session_state.collection = create_hw4_collection()
            if st.session_state.collection:
                st.session_state.system_ready = True
                st.success("AI ChatBot is Ready!!!")
            else:
                st.error("Failed to create or load the collection.")

    # Chat Interface
    if st.session_state.system_ready and st.session_state.collection:
        st.subheader(f"Chat with the AI (Using {selected_model})")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask a question:")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            combined_query = f"{' '.join([exchange['question'] for exchange in st.session_state.conversation_memory])} {user_input}"
            relevant_texts, relevant_docs = query_vector_db(st.session_state.collection, combined_query)
            context = "\n".join(relevant_texts)

            response = get_chatbot_response(user_input, context, st.session_state.conversation_memory, selected_model)

            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.conversation_memory.append({"question": user_input, "answer": response})

            with st.expander("Relevant documents used"):
                for doc in relevant_docs:
                    st.write(f"- {doc}")
    else:
        st.error("System not ready. Please check the setup.")

if __name__ == "__main__":
    main()
