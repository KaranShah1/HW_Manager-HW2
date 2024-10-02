import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import cohere
import requests
from bs4 import BeautifulSoup
import tiktoken
import google.generativeai as genai
import zipfile

# Function to read webpage content from a URL
def read_webpage_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        document = " ".join([p.get_text() for p in soup.find_all("p")])
        return document
    except requests.RequestException as e:
        st.error(f"Error reading webpage from {url}: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing the webpage: {e}")
        return None

# Function to calculate tokens
def calculate_tokens(messages):
    """Calculate total tokens for a list of messages."""
    total_tokens = 0
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    for msg in messages:
        total_tokens += len(encoding.encode(msg['content']))
    return total_tokens

def truncate_messages_by_tokens(messages, max_tokens):
    """Truncate the message buffer to ensure it stays within max_tokens."""
    total_tokens = calculate_tokens(messages)
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        total_tokens = calculate_tokens(messages)
    return messages

# Function to verify OpenAI API key
def verify_openai_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to generate summary using OpenAI
def generate_openai_response(client, messages, model):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        return stream
    except Exception as e:
        st.error(f"Error generating response: {e}", icon="❌")
        return None

def verify_cohere_key(api_key):
    try:
        client = cohere.Client(api_key)
        client.generate(prompt="Hello", max_tokens=5)
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to generate response using Cohere
def generate_cohere_response(client, messages):
    try:
        stream = client.chat_stream(
            model='command-r',
            message=messages[-1]['content'],
            chat_history=[{"role": m['role'], "message": m['content']} for m in messages[:-1]],
            temperature=0,       
            max_tokens=1500
        )
        return stream
    except Exception as e:
        st.error(f"Error generating response: {e}", icon="❌")
        return None

# Function to verify Gemini API key
def verify_gemini_key(api_key):
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-pro')
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

def generate_gemini_response(client, messages, prompt):
    try:
        msgs = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            msgs.append({"role": role, "parts": msg["content"]})
        response = client.generate_content(
            contents=[*msgs, {"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1500,
            ),
            stream=True
        )
        return response
    except Exception as e:
        return None

def generate_conversation_summary(client, messages, llm_provider):
    if llm_provider == 'Gemini':
        msgs = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            msgs.append({"role": role, "parts": [{"text": msg["content"]}]})
        prompt = {"role": "user", "parts": [{"text": "Summarize the key points of this conversation concisely:"}]}
        response = client.generate_content(
            contents=[prompt, *msgs],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=150,
            ),
        )
        return response.text
    elif "OpenAI" in llm_provider:
        summary_prompt = "Summarize the key points of this conversation concisely:"
        for msg in messages:
            summary_prompt += f"\n{msg['role']}: {msg['content']}"
        response = client.chat.completions.create(
            model="gpt-4-mini" if llm_provider == "OpenAI GPT-4O-Mini" else "gpt-4",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    else:  # Cohere
        summary_prompt = "Summarize the key points of this conversation concisely:"
        chat_history = []
        for msg in messages:
            chat_history.append({"role": msg['role'], "message": msg['content']})
            summary_prompt += f"\n{msg['role']}: {msg['content']}"
        response = client.chat(
            model='command-r',
            message=summary_prompt,
            chat_history=chat_history,
            temperature=0,       
            max_tokens=150
        )
        return response.text

# Vector DB functions
def add_to_collection(collection, text, filename):
    openai_client = OpenAI(api_key = st.secrets['openai'])
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    collection.add(
        documents=[text],
        ids=[filename],
        embeddings=[embedding]
    )
    return collection

def setup_vectordb():
    db_path = "HW4_VectorDB"
    
    # Check if vector DB exists on disk
    if not os.path.exists(db_path):
        st.info("Setting up vector DB for the first time...")
        
        # Initialize the ChromaDB client with persistence
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get the collection
        collection = client.get_or_create_collection(
            name="HW4Collection",
            metadata={"hnsw:space": "cosine", "hnsw:M": 32}
        )
        
        # su_orgs_path = os.path.join(os.getcwd(), "HWs/su_orgs/")
        # html_files = [f for f in os.listdir(su_orgs_path) if f.endswith('.html')]
        
        # for html_file in html_files:
        #     file_path = os.path.join(su_orgs_path, html_file)
        #     with open(file_path, 'r', encoding='utf-8') as file:
        #         soup = BeautifulSoup(file, 'html.parser')
        #         text = soup.get_text(separator=' ', strip=True)
        #         collection = add_to_collection(collection, text, html_file)

        # Define the path to the zip file
        zip_path = os.path.join(os.getcwd(), "su_orgs.zip")
        if not os.path.exists(zip_path):
            st.error(f"Zip file not found: {zip_path}")
            return None

        # Extract HTML files from zip
        html_files = extract_html_from_zip(zip_path)

        
        st.success(f"VectorDB setup complete with {len(html_files)} HTML files!")
    else:
        # If it already exists, just load it
        st.info("VectorDB already exists. Loading from disk...")
        client = chromadb.PersistentClient(path=db_path)
        st.session_state.HW4_vectorDB = client.get_collection(name="HW4Collection")

def query_vectordb(query, k=3):
    if 'HW4_vectorDB' in st.session_state:
        collection = st.session_state.HW4_vectorDB
        openai_client = OpenAI(api_key = st.secrets['openai'])
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            include=['documents', 'distances', 'metadatas'],
            n_results=k
        )
        return results
    else:
        st.error("VectorDB not set up. Please set up the VectorDB first.")
        return None

st.title("iSchool Student Organizations Chatbot")

# Set up VectorDB
setup_vectordb()

# Sidebar: LLM provider selection
st.sidebar.header("LLM Provider")
llm_provider = st.sidebar.selectbox(
    "Choose your LLM provider:",
    options=["OpenAI GPT-4O-Mini", "OpenAI GPT-4O", "Cohere", "Gemini"]
)

# Sidebar: Conversation memory type
st.sidebar.header("Conversation Memory")
memory_type = st.sidebar.radio(
    "Choose conversation memory type:",
    options=["Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens"]
)

# API key verification
if "OpenAI" in llm_provider:
    openai_api_key = st.secrets['openai']
    client, is_valid, message = verify_openai_key(openai_api_key)
    model = "gpt-4o-mini" if llm_provider == "OpenAI GPT-4O-Mini" else "gpt-4o"
elif "Cohere" in llm_provider:
    cohere_api_key = st.secrets['cohere']
    client, is_valid, message = verify_cohere_key(cohere_api_key)
else:
    gemini_api_key = st.secrets['gemini']
    client, is_valid, message = verify_gemini_key(gemini_api_key)

if is_valid:
    st.sidebar.success(f"{llm_provider} API key is valid!", icon="✅")
else:
    st.sidebar.error(f"Invalid {llm_provider} API key: {message}", icon="❌")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "system"
    with st.chat_message(role):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about iSchool student organizations?"):
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Query VectorDB for relevant documents
    results = query_vectordb(prompt)
    if results:
        context = " ".join([doc for doc in results['documents'][0]])
        context_message = {"role": "system", "content": f"Here is relevant information: {context}"}
    else:
        context_message = {"role": "system", "content": "No specific context found. Please answer based on general knowledge about iSchool student organizations."}
    
    messages_for_llm = [context_message] + st.session_state.messages
    
    # Apply conversation memory type
    if memory_type == "Buffer of 5 questions":
        messages_for_llm = messages_for_llm[-11:]  # System message + last 5 Q&A pairs
    elif memory_type == "Conversation summary":
        if 'conversation_summary' not in st.session_state:
            st.session_state.conversation_summary = ""
        st.session_state.conversation_summary = generate_conversation_summary(client, messages_for_llm, llm_provider)
        messages_for_llm = [
            context_message,
            {"role": "system", "content": f"Conversation summary: {st.session_state.conversation_summary}"},
            st.session_state.messages[-1]
        ]
    else:
        messages_for_llm = truncate_messages_by_tokens(messages_for_llm, 5000)

    with st.chat_message("system"):
        message_placeholder = st.empty()
        full_response = ""
        if "OpenAI" in llm_provider:
            stream = generate_openai_response(client, messages_for_llm, model)
            if stream:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        elif "Cohere" in llm_provider:
            stream = generate_cohere_response(client, messages_for_llm)
            if stream:
                for event in stream:
                    if event.event_type == "text-generation":
                        full_response += event.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
        else:  # Gemini
            response = generate_gemini_response(client, messages_for_llm, prompt)
            if response:
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "system", "content": full_response})