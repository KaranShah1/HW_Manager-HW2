import streamlit as st
from openai import OpenAI
import os
import pandas as pd

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to ensure the OpenAI client is initialized
def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to create the ChromaDB collection for news stories
def create_news_collection():
    if 'news_vectorDB' not in st.session_state:
        # Set up the ChromaDB client
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection("NewsCollection")

        ensure_openai_client()

        # Define the path to the CSV file
        csv_file = os.path.join(os.getcwd(), "news_data.csv")
        if not os.path.exists(csv_file):
            st.error(f"CSV file not found: {csv_file}")
            return None

        # Read the CSV file
        news_df = pd.read_csv(csv_file)

        # Process each news story in the CSV
        for _, row in news_df.iterrows():
            try:
                news_text = row['content']
                news_title = row['title']
                metadata = {"title": news_title, "category": row['category']}

                # Generate embeddings for the news story
                response = st.session_state.openai_client.embeddings.create(
                    input=news_text, model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding

                # Add the document to ChromaDB
                collection.add(
                    documents=[news_text],
                    metadatas=[metadata],
                    ids=[row['id']],  # Assuming 'id' is a column in your CSV
                    embeddings=[embedding]
                )
            except Exception as e:
                st.error(f"Error processing news story {news_title}: {str(e)}")

        # Store the collection in session state
        st.session_state.news_vectorDB = collection

    return st.session_state.news_vectorDB

# Function to query the vector database for news stories
def query_vector_db(collection, query):
    ensure_openai_client()
    try:
        # Generate embedding for the query
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding

        # Query the ChromaDB collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return results['documents'][0], [result['title'] for result in results['metadatas'][0]]
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return [], []

# Function to get chatbot response using OpenAI's GPT model
def get_chatbot_response(query, context):
    ensure_openai_client()
    # Construct the prompt for the GPT model
    prompt = f"""You are an AI assistant with access to news stories. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information.

Context:
{context}

User Question: {query}

Answer:"""

    try:
        response = st.session_state.openai_client.completions.create(
            model="gpt-4",  # Using GPT-4 model
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.error(f"Error getting chatbot response: {str(e)}")
        return None

# Initialize session state for chat history and system readiness
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False
if 'collection' not in st.session_state:
    st.session_state.collection = None

# Page content
st.title("News Chatbot")

# Check if the system is ready, if not, prepare it
if not st.session_state.system_ready:
    # Show a spinner while processing news stories
    with st.spinner("Processing news stories..."):
        st.session_state.collection = create_news_collection()
        if st.session_state.collection:
            # Set the system as ready
            st.session_state.system_ready = True
            st.success("News ChatBot is Ready!")
        else:
            st.error("Failed to create or load the news collection.")

# Only show the chat interface if the system is ready
if st.session_state.system_ready and st.session_state.collection:
    st.subheader("Chat with the News AI Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask about news stories:")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Query the vector database for relevant news
        relevant_texts, relevant_titles = query_vector_db(st.session_state.collection, user_input)
        context = "\n".join(relevant_texts)

        # Get chatbot response
        chatbot_response = get_chatbot_response(user_input, context)

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(chatbot_response)

        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": chatbot_response})

        # Display relevant news titles
        with st.expander("Relevant news stories used"):
            for title in relevant_titles:
                st.write(f"- {title}")

elif not st.session_state.system_ready:
    st.info("The system is still preparing. Please wait...")
else:
    st.error("Failed to create or load the news collection.")
