import sys
import streamlit as st
from openai import OpenAI
import os
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Solution for the SQLite issue in Streamlit Cloud.
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


def initialize_openai_client():
    # Check if the OpenAI client has been initialized, if not, initialize it using API key from secrets
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to create the ChromaDB collection
def create_news_collection():
    # If the News_Collection isn't in session state, create a new one
    if 'News_Collection' not in st.session_state:
        persist_directory = os.path.join(os.getcwd(), "chroma_db")
         
        # Create a persistent ChromaDB client to store vectors
        client = chromadb.PersistentClient(path=persist_directory)

         # Retrieve or create the news collection
        collection = client.get_or_create_collection("News_Collection")

        # Path to CSV file for news data
        csv_path = os.path.join(os.getcwd(), "Example_news_info_for_testing.csv")
          # If CSV file is not found, display error
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found: {csv_path}")
            return None

          # If collection is empty, process and add data
        if collection.count() == 0:
            with st.spinner("Processing content and preparing the system..."):
                initialize_openai_client()

                df = pd.read_csv(csv_path)
                for index, row in df.iterrows():
                    try:
                        # Convert days_since_2000 to a readable date
                        date = (datetime(2000, 1, 1) + timedelta(days=int(row['days_since_2000']))).strftime('%Y-%m-%d')

                        # Create a document combining company, date, document, and URL
                        text = f"Company: {row['company_name']}\nDate: {date}\nDocument: {row['Document']}\nURL: {row['URL']}"

                        # Create OpenAI embeddings for the document
                        response = st.session_state.openai_client.embeddings.create(
                            input=text, model="text-embedding-3-small"
                        )
                        embedding = response.data[0].embedding

                        # Add the document, metadata, ID, and embeddings to the collection
                        collection.add(
                            documents=[text],
                            metadatas=[{"company": row['company_name'], "date": date, "url": row['URL']}],
                            ids=[str(index)],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing row {index}: {str(e)}")
        else:
            # Inform that the existing vector database will be used
            st.info("Using existing vector database.")
           
        # Store the collection in session state
        st.session_state.News_Collection = collection

    return st.session_state.News_Collection

# Function to get relevant news info based on the query
def get_relevant_info(query):
    # Get the news collection from session state
    collection = st.session_state.News_Collection

    initialize_openai_client()
    try:
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

    # Normalize the embedding
    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
       # Query the collection for the most relevant results based on the query embedding
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )
         # Retrieve the relevant text and metadata
        relevant_texts = results['documents'][0]
        relevant_docs = [f"{result['company']} - {result['date']}" for result in results['metadatas'][0]]
        return "\n".join(relevant_texts), relevant_docs
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return "", []

# Function to call the OpenAI LLM (large language model) and manage its response
def call_llm(model, messages, temp, query, tools=None):
    initialize_openai_client()
    try:
        # Make a streaming call to OpenAI's chat model with given parameters
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

    # Create a Streamlit empty container to hold the streaming response
    response_container = st.empty()

    try:
        # Stream the response from OpenAI
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls: 
                #hasattr() is a built-in function in Python that checks if an object has a specified attribute
                 
                # Check if any tools were called in the response
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.function:
                        tool_called = tool_call.function.name
                        if tool_called == "get_news_info":
                            extra_info = get_relevant_info(query)
                            tool_usage_info = f"Tool used: {tool_called}"
                            update_system_prompt(messages, extra_info)
                            recursive_response, recursive_tool_info = call_llm(
                                model, messages, temp, query, tools)
                            full_response += recursive_response
                            tool_usage_info += "\n" + recursive_tool_info
                            response_container.markdown(full_response)
                            return full_response, tool_usage_info
            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                # Update the response container with the latest content
                response_container.markdown(full_response)
    except Exception as e:
        st.error(f"Error in streaming response: {str(e)}")

    if tool_called:
        tool_usage_info = f"Tool used: {tool_called}"
    else:
        tool_usage_info = "No tools were used in generating this response."

    return full_response, tool_usage_info

def generate_chatbot_response(query, context, conversation_memory):
    system_message = """You are an AI assistant specialized in providing information about news stories for a large global law firm. 
    Your primary source of information is the context provided, which contains relevant data extracted from embeddings of news articles.

    Only use the get_news_info tool when:

    a) A specific company or news topic related to a comapny is mentioned in the user's query, OR
    b) If the user asks a follow-up question about a specific news item mentioned in a previous response.

    Always prioritize using the context for general inquiries about news or types of news.

    When identifying the most compelling news, consider these factors:

    Recency: More recent news is generally more relevant.
    Global impact: The extent to which the news affects people or industries worldwide.
    Business implications: How the news could influence companies, markets, or the economy.
    Legal relevance: How closely the news ties to legal issues.
    Novelty: How new or unique the news is.

    Provide a ranked list of news items with brief explanations of why they are interesting for a global law firm. Include the company name and date for each news item."""

    # Create a condensed conversation history
    condensed_history = "\n".join(
        [f"Human: {exchange['question']}\nAI: {exchange['answer']}" for exchange in conversation_memory]
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context}\n\nConversation history:\n{condensed_history}\n\nQuestion: {query}"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_news_info",
                "description": "Get information about specific news stories or topics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The news topic, company name, or story to look up"
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
    ]

    try:
        response, tool_usage_info = call_llm(
            "gpt-4o", messages, 0.7, query, tools)
        return response, tool_usage_info
    except Exception as e:
        st.error(f"Error getting GPT-4 response: {str(e)}")
        return None, "Error occurred while generating response."

def update_system_prompt(messages, extra_info):
    for message in messages:
        if message["role"] == "system":
            message["content"] += f"\n\nAdditional information: {extra_info}"
            break

def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = deque(maxlen=5)
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'collection' not in st.session_state:
        st.session_state.collection = None

    st.title("My AI News Bot -  Karan Shah")
    st.write("""
    This AI assistant helps global law firms stay updated on the latest legal news.
    Ask about specific company news, legal disputes, or business trends worldwide.
    """)
    # Sidebar for displaying news topics
    with st.sidebar:
        st.header("Available News Topics")
        st.write("""
        - Corporate mergers
        - Intellectual property disputes
        - Regulatory changes
        - Legal cases
        """)

    if not st.session_state.system_ready:
        with st.spinner("Processing news articles and preparing the system..."):
            st.session_state.collection = create_news_collection()
            if st.session_state.collection:
                st.session_state.system_ready = True
                st.success("News Bot is Ready!")
            else:
                st.error(
                    "Failed to create or load the news collection. Please check the CSV file and try again.")

    if st.session_state.system_ready and st.session_state.collection:
        st.subheader("Chat with the News Bot (Using OpenAI GPT-4)")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask a question about the news:")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            relevant_texts, relevant_docs = get_relevant_info(user_input)
            st.write(
                f"Debug: Relevant texts found: {len(relevant_texts)} characters")

            with st.chat_message("assistant"):
                response, tool_usage_info = generate_chatbot_response(
                    user_input, relevant_texts, st.session_state.conversation_memory)

                if response is None:
                    st.error("Failed to get a response from the AI. Please try again.")
                    return

                st.info(tool_usage_info)

            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            st.session_state.conversation_memory.append({
                "question": user_input,
                "answer": response
            })

            with st.expander("Relevant news articles used"):
                for doc in relevant_docs:
                    st.write(f"- {doc}")

    elif not st.session_state.system_ready:
        st.info("The system is still preparing. Please wait...")
    else:
        st.error(
            "Failed to create or load the news collection. Please check the CSV file and try again.")

if __name__ == "__main__":
    main()
