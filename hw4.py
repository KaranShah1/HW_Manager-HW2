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
        pdf_dir = os.path.join(os.getcwd(), "su_orgs")
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


# Function to get chatbot response using the selected LLM
def get_chatbot_response(query, context, llm_provider, client):
    prompt = f"""You are an AI assistant with knowledge from specific documents. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information.

Context:
{context}

User Question: {query}

Answer:"""

    if llm_provider == 'OpenAI GPT-4O-Mini' or llm_provider == 'OpenAI GPT-4O':
        model = "gpt-4o-mini" if llm_provider == "OpenAI GPT-4O-Mini" else "gpt-4o"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content
    elif llm_provider == 'Gemini':
        response = client.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=150,
            ),
        )
        return response.text
    else:  # Cohere
        response = client.chat(
            model='command-r',
            message=prompt,
            temperature=0,       
            max_tokens=150
        )
        return response.text


# Function to summarize the conversation based on the selected LLM
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
            model="gpt-4o-mini" if llm_provider == "OpenAI GPT-4O-Mini" else "gpt-4o",
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


# API key verification
def verify_keys(llm_provider):
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
        return client
    else:
        st.sidebar.error(f"Invalid {llm_provider} API key: {message}", icon="❌")
        st.stop()
        return None


# Initialize session state for chat history, system readiness, and collection
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None

# Sidebar to select the LLM provider
llm_provider = st.sidebar.selectbox(
    "Select LLM provider",
    ["OpenAI GPT-4O", "OpenAI GPT-4O-Mini", "Gemini", "Cohere"]
)

# Verify API key and initialize the client
client = verify_keys(llm_provider)

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
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about the documents:")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Query the vector database and get the relevant context
        context, doc_names = query_vector_db(st.session_state.collection, user_input)

        # Get the chatbot's response
        response = get_chatbot_response(user_input, context, llm_provider, client)

        # Display chatbot's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add chatbot's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Summarize the conversation
    if st.button("Summarize Conversation"):
        summary = generate_conversation_summary(client, st.session_state.chat_history, llm_provider)
        st.subheader("Conversation Summary")
        st.write(summary)
