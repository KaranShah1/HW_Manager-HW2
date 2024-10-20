import sys
import streamlit as st
from openai import OpenAI
import os
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to ensure the OpenAI client is initialized
def ensure_openai_client():
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
                ensure_openai_client()

                # Read data from CSV
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

    # Ensure OpenAI client is initialized
    ensure_openai_client()
    try:
        # Generate OpenAI embeddings for the query
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

    # Normalize the query embedding
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
    ensure_openai_client()
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
                # Check if any tools were called in the response
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.function:
                        tool_called = tool_call.function.name
                        if tool_called == "get_news_info":
                            # If 'get_news_info' is called, get extra news info
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
                # Accumulate the content of the response
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

# Function to manage the chatbot response based on query and conversation history
def get_chatbot_response(query, context, conversation_memory):
    # System message for the chatbot, defining its role and behavior
    system_message = """You are an AI assistant designed to provide information about news stories for a global law firm. Your responses should be based on the context provided, which includes relevant data from embedded news articles.

Please only use the get_news_info tool in the following cases:

When the user’s query specifically mentions a company or a news topic related to a company, or
If the user asks a follow-up question regarding a specific news item discussed in a previous response.
For general inquiries about news or categories of news, prioritize using the context provided.

When asked to identify the most interesting news stories, please consider:

How relevant the news is to legal matters.
The global significance of the news.
The potential impact on businesses.
The novelty or uniqueness of the story.
The recency of the news (more recent stories tend to be more engaging).
Provide a ranked list of news stories, including brief explanations for why each one is relevant or interesting to a global law firm, along with the company name and the date of the news item."""

    # Condense the conversation history to keep track of previous exchanges
    condensed_history = "\n".join(
        [f"Human: {exchange['question']}\nAI: {exchange['answer']}" for exchange in conversation_memory]
    )

    # Prepare the message to be sent to the LLM, combining system and user context
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context}\n\nConversation history:\n{condensed_history}\n\nQuestion: {query}"}
    ]

    # Define the tool for news info retrieval
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
        # Call the LLM with the prepared messages and tools
        response, tool_usage_info = call_llm(
            "gpt-4o", messages, 0.7, query, tools)
    except Exception as e:
        st.error(f"Error generating chatbot response: {str(e)}")
        return "", "Error occurred while generating chatbot response."

    # Return the response and tool usage information
    return response, tool_usage_info

# Streamlit app's main function
def main():
    st.title("Legal News Chatbot")
    # Introduction section of the app
    st.write("""
    This chatbot is designed to help global law firms access important legal news.
    Ask about company news, legal stories, or trends affecting businesses worldwide.
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

    # User input section for asking the chatbot
    user_input = st.text_input("Ask about a company, news story, or legal topic:")

    # If the user submits a query
    if user_input:
        # Create the news collection if it doesn't exist yet
        create_news_collection()

        # Retrieve context and chatbot response
        context = "Global law firm news updates"
        conversation_memory = deque(maxlen=10)  # Keep track of last 10 exchanges
        answer, tool_info = get_chatbot_response(
            user_input, context, conversation_memory)

        # Update conversation memory with the latest interaction
        conversation_memory.append(
            {"question": user_input, "answer": answer})

        # Display the chatbot's answer and tool usage information
        st.write("### Response")
        st.write(answer)

        st.write("### Tool usage")
        st.write(tool_info)

if __name__ == "__main__":
    main()
