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

# Function to initialize the OpenAI client
def initialize_openai_client():
    # If OpenAI client is not yet initialized, initialize it using API key
    if 'openai_api_client' not in st.session_state:
        api_key = st.secrets["openai"]
        st.session_state.openai_api_client = OpenAI(api_key=api_key)

# Function to set up the ChromaDB collection
def setup_news_collection():
    # If the collection isn't available, create it
    if 'news_vector_collection' not in st.session_state:
        persist_directory = os.path.join(os.getcwd(), "persistent_chroma_db")
        # Instantiate a persistent ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)
        # Get or create the news collection
        collection = client.get_or_create_collection("news_vector_collection")

        # Specify path to CSV file containing news data
        csv_file = os.path.join(os.getcwd(), "Example_news_info_for_testing.csv")
        if not os.path.exists(csv_file):
            st.error(f"CSV file not found: {csv_file}")
            return None

        # Check if the collection is empty and populate if necessary
        if collection.count() == 0:
            with st.spinner("Initializing the system and processing content..."):
                initialize_openai_client()

                # Load CSV data into a DataFrame
                news_data = pd.read_csv(csv_file)
                for index, row in news_data.iterrows():
                    try:
                        # Convert 'days_since_2000' to a readable date format
                        news_date = (datetime(2000, 1, 1) + timedelta(days=int(row['days_since_2000']))).strftime('%Y-%m-%d')
                        
                        # Combine relevant fields into a single document text
                        news_text = f"Company: {row['company_name']}\nDate: {news_date}\nDocument: {row['Document']}\nURL: {row['URL']}"

                        # Generate embeddings for the news document
                        embedding_response = st.session_state.openai_api_client.embeddings.create(
                            input=news_text, model="text-embedding-3-small"
                        )
                        embedding = embedding_response.data[0].embedding

                        # Add the document, metadata, and embedding to the collection
                        collection.add(
                            documents=[news_text],
                            metadatas=[{"company": row['company_name'], "date": news_date, "url": row['URL']}],
                            ids=[str(index)],
                            embeddings=[embedding]
                        )
                    except Exception as e:
                        st.error(f"Error processing row {index}: {str(e)}")
        else:
            st.info("Using the existing vector database.")

        st.session_state.news_vector_collection = collection

    return st.session_state.news_vector_collection

# Function to find relevant news based on the query
def fetch_relevant_news(query):
    collection = st.session_state.news_vector_collection

    initialize_openai_client()
    try:
        # Generate embeddings for the query using OpenAI
        query_embedding_response = st.session_state.openai_api_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = query_embedding_response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding for query: {str(e)}")
        return "", []

    # Normalize the query embedding for similarity search
    query_embedding = np.array(query_embedding) / np.linalg.norm(query_embedding)

    try:
        # Search the collection for matching results
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )
        relevant_texts = results['documents'][0]
        relevant_metadata = [f"{meta['company']} - {meta['date']}" for meta in results['metadatas'][0]]
        return "\n".join(relevant_texts), relevant_metadata
    except Exception as e:
        st.error(f"Error querying the collection: {str(e)}")
        return "", []

# Function to invoke the OpenAI language model and handle responses
def invoke_language_model(model, message_log, temp, query, toolset=None):
    initialize_openai_client()
    try:
        # Make an API call to the OpenAI model for chat completion with possible tool use
        response = st.session_state.openai_api_client.chat.completions.create(
            model=model,
            messages=message_log,
            temperature=temp,
            tools=toolset,
            tool_choice="auto" if toolset else None,
            stream=True
        )
    except Exception as e:
        st.error(f"Error invoking OpenAI model: {str(e)}")
        return "", "Error occurred during language model response."

    tool_invoked = None
    complete_response = ""
    tool_usage_info = ""

    # Stream the response in real-time
    response_container = st.empty()

    try:
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.function:
                        tool_invoked = tool_call.function.name
                        if tool_invoked == "fetch_news_data":
                            additional_info = fetch_relevant_news(query)
                            tool_usage_info = f"Tool invoked: {tool_invoked}"
                            append_system_message(message_log, additional_info)
                            recursive_response, recursive_tool_info = invoke_language_model(
                                model, message_log, temp, query, toolset)
                            complete_response += recursive_response
                            tool_usage_info += "\n" + recursive_tool_info
                            response_container.markdown(complete_response)
                            return complete_response, tool_usage_info
            elif hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                complete_response += chunk.choices[0].delta.content
                response_container.markdown(complete_response)
    except Exception as e:
        st.error(f"Error streaming the response: {str(e)}")

    tool_usage_info = f"Tool used: {tool_invoked}" if tool_invoked else "No tools were used."
    return complete_response, tool_usage_info

# Function to process the chatbot response based on user input and context
def generate_chatbot_response(query, context_data, memory_log):
    system_instructions = """You are a specialized AI assistant providing insights into legal news for a global law firm. Base your responses on the provided context, which contains embedded data from news articles.

Use the fetch_news_data tool in the following cases:
- If the query mentions a company or a related legal topic.
- When a follow-up question is asked about a previously discussed news item.

For general inquiries, prioritize using the context.

For selecting the most important news, consider:
- Legal relevance.
- Global significance.
- Business impact.
- Uniqueness or novelty.
- Recency.

Rank news items with short explanations including company name and date."""

    summarized_history = "\n".join(
        [f"User: {entry['question']}\nAI: {entry['answer']}" for entry in memory_log]
    )

    # Construct message log for the conversation
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": f"Context: {context_data}\n\nHistory:\n{summarized_history}\n\nQuery: {query}"}
    ]

    # Define the tool for fetching news
    tool_list = [
        {
            "type": "function",
            "function": {
                "name": "fetch_news_data",
                "description": "Retrieve specific news stories or information based on a query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The name of the company or news topic to retrieve"
                        }
                    },
                    "required": ["topic"]
                }
            }
        }
    ]

    try:
        # Invoke the language model to generate a response
        final_response, tool_details = invoke_language_model(
            "gpt-4o", messages, 0.7, query, tool_list)
    except Exception as e:
        st.error(f"Error generating chatbot response: {str(e)}")
        return "", "Error occurred while generating chatbot response."

    return final_response, tool_details

# Main function for Streamlit app
def main():
    st.title("Legal News AI Assistant")
    st.write("""
    This AI assistant helps global law firms stay updated on the latest legal news.
    Ask about specific company news, legal disputes, or business trends worldwide.
    """)

    # Sidebar content with sample topics
    with st.sidebar:
        st.header("Sample News Topics")
        st.write("""
        - Corporate Mergers
        - Intellectual Property Cases
        - Regulatory Changes
        - Significant Legal Disputes
        """)

    # Input field for user to ask a question
    user_query = st.text_input("Enter your query regarding a company, legal case, or news story:")

    if user_query:
        # Set up news collection if it's not already initialized
        setup_news_collection()

        # Fetch context and generate response
        news_context = "Legal news database for global law firms"
        memory_log = deque(maxlen=10)  # Keep track of the last 10 interactions
        response, tool_usage = generate_chatbot_response(
            user_query, news_context, memory_log)

        # Update conversation history
        memory_log.append({"question": user_query, "answer": response})

        # Display chatbot response and tool usage
        st.write("### Response")
        st.write(response)

        st.write("### Tool Usage")
        st.write(tool_usage)

if __name__ == "__main__":
    main()
