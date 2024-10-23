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

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to ensure the OpenAI client is initialized

def ensure_openai_client():
    if 'openai_client' not in st.session_state:
        api_key = st.secrets["openai_api_key"]
        st.session_state.openai_client = OpenAI(api_key=api_key)

# Function to extract HTML files from zip


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

# Function to create the ChromaDB collection


def create_collection():
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
                ensure_openai_client()

                for filename, content in html_files.items():
                    try:
                        soup = BeautifulSoup(content, 'html.parser')
                        text = soup.get_text(separator=' ', strip=True)

                        response = st.session_state.openai_client.embeddings.create(
                            input=text, model="text-embedding-3-small"
                        )
                        embedding = response.data[0].embedding

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

# Function to get relevant club info based on the query


def get_relevant_info(query):
    collection = st.session_state.HW_URL_Collection

    ensure_openai_client()
    try:
        response = st.session_state.openai_client.embeddings.create(
            input=query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating OpenAI embedding: {str(e)}")
        return "", []

    # Normalize the embedding
    query_embedding = np.array(query_embedding) / \
        np.linalg.norm(query_embedding)

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )
        relevant_texts = results['documents'][0]
        relevant_docs = [result['filename']
                         for result in results['metadatas'][0]]
        return "\n".join(relevant_texts), relevant_docs
    except Exception as e:
        st.error(f"Error querying the database: {str(e)}")
        return "", []



def interact_with_llm(model, messages, temp, query,  tools=None):
    ensure_openai_client()
    try:
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

    try:
        while True:
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            tool_called = tool_call.function.name
                            if tool_called == "get_club_info":
                                extra_info = get_relevant_info(query)
                                tool_usage_info = f"Tool used: {tool_called}"
                                update_system_prompt(messages, extra_info)
                                recursive_response, recursive_tool_info = interact_with_llm(
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


def get_chatbot_response(query, context, conversation_memory):
    system_message = """You are an AI assistant specialized in providing information about student organizations and clubs at Syracuse University. 
    Your primary source of information combines:
    1. Context from vector embeddings of club descriptions and details
    2. Information from newly uploaded files
    3. Conversation history for handling follow-up questions

    When processing queries, follow these rules:

    1. Use the get_club_info tool ONLY when:
        a) A specific club name is mentioned in the user's query, OR
        b) A follow-up question references a specific club from any previous response in the chat history
           - Search the conversation history to find the referenced club name
           - Pass the found club name as an argument

    2. For general inquiries about clubs or types of clubs:
        a) Prioritize using the provided context
        b) Incorporate relevant information from uploaded files
        c) Reference previous conversation context when applicable

    3. When handling follow-up questions:
        a) Review conversation history to maintain context
        b) Resolve ambiguous references using previous exchanges
        c) Combine historical context with new search results

    4. For unclear or ambiguous queries:
        a) Ask for clarification
        b) Reference relevant parts of conversation history
        c) Suggest possible interpretations based on context

    Always maintain conversation context to provide coherent and relevant responses across multiple exchanges.
    Give response in list format when giving the list of clubs."""

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

    try:
        response, tool_usage_info = interact_with_llm(
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

    st.title("iSchool Chatbot")

    if not st.session_state.system_ready:
        with st.spinner("Processing documents and preparing the system..."):
            st.session_state.collection = create_collection()
            if st.session_state.collection:
                st.session_state.system_ready = True
                st.success("AI ChatBot is Ready!")
            else:
                st.error(
                    "Failed to create or load the document collection. Please check the zip file and try again.")

    if st.session_state.system_ready and st.session_state.collection:
        st.subheader("Chat with the Assistant")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask a question about information at Syracuse University:")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            relevant_texts, relevant_docs = get_relevant_info(user_input)
            st.write(
                f"Debug: Relevant texts found: {len(relevant_texts)} characters")

            response, tool_usage_info = get_chatbot_response(
                user_input, relevant_texts, st.session_state.conversation_memory)

            if response is None:
                st.error("Failed to get a response from the AI. Please try again.")
                return

            with st.chat_message("assistant"):
                st.markdown(response)
                st.info(tool_usage_info)

            st.session_state.chat_history.append(
                {"role": "user", "content": user_input})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            st.session_state.conversation_memory.append({
                "question": user_input,
                "answer": response
            })

            with st.expander("Relevant documents used"):
                for doc in relevant_docs:
                    st.write(f"- {doc}")

    elif not st.session_state.system_ready:
        st.info("The system is still preparing. Please wait...")
    else:
        st.error(
            "Failed to create or load the document collection. Please check the zip file and try again.")


# if __name__ == "__main__":
main()