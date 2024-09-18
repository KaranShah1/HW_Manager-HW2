import os
import openai
import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
import google.generativeai as genai
import tiktoken

# Sidebar Options
st.sidebar.title("LLM Interaction Settings")

# URL Input in Sidebar
url1 = st.sidebar.text_input("Enter first URL:")
url2 = st.sidebar.text_input("Enter second URL:")

# LLM Vendor Selection in Sidebar
llm_vendor = st.sidebar.selectbox(
    "Select LLM Vendor", 
    ("Cohere", "Gemini", "OpenAI 3.5", "OpenAI 4")
)

# Conversation Memory Type Selection in Sidebar
memory_type = st.sidebar.selectbox(
    "Select Conversation Memory Type", 
    ("Buffer of 5 questions", "Conversation Summary", "Buffer of 5,000 tokens")
)

# Session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to calculate tokens
def calculate_tokens(messages):
    """Calculate total tokens for a list of messages."""
    total_tokens = 0
    encoding = tiktoken.encoding_for_model('gpt-4')
    for msg in messages:
        total_tokens += len(encoding.encode(msg['content']))
    return total_tokens

# Function to truncate messages by token limit
def truncate_messages_by_tokens(messages, max_tokens):
    """Truncate the message buffer to ensure it stays within max tokens."""
    total_tokens = calculate_tokens(messages)
    while total_tokens > max_tokens and len(messages) > 1:
        messages.pop(0)
        total_tokens = calculate_tokens(messages)
    return messages

# Function to handle the conversation memory logic
def handle_memory(messages, memory_type):
    if memory_type == "Buffer of 5 questions":
        # Only keep the last 5 messages
        return messages[-5:]
    elif memory_type == "Conversation Summary":
        # Create a simple summary of the conversation
        summary = " ".join([msg['content'] for msg in messages])
        return [{"role": "system", "content": summary}]
    elif memory_type == "Buffer of 5,000 tokens":
        # Truncate the messages within a 5,000 token buffer
        max_tokens = 5000
        return truncate_messages_by_tokens(messages, max_tokens)

# Function to generate Cohere response
def generate_cohere_response(client, messages):
    try:
        response = client.chat(messages=[msg['content'] for msg in messages])
        return response.generations[0].text
    except Exception as e:
        st.error(f"Error generating Cohere response: {e}")
        return None

# Function to verify Gemini API key
def verify_gemini_key(api_key):
    try:
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel('gemini-pro')  # Use Gemini model
        return client, True, "API key is valid"
    except Exception as e:
        return None, False, str(e)

# Function to generate Gemini response
def generate_gemini_response(client, messages):
    try:
        msgs = handle_memory(messages, memory_type)
        formatted_msgs = [{"role": "user" if msg['role'] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in msgs]
        response = client.generate_content(
            contents=formatted_msgs,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1500,
            )
        )
        return response.generations[0].text
    except Exception as e:
        st.error(f"Error generating Gemini response: {e}")
        return None

# Function to generate OpenAI response
def generate_openai_response(client, messages, model):
    try:
        chat_history = handle_memory(messages, memory_type)
        formatted_messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
        response = openai.ChatCompletion.create(
            model=model,
            messages=formatted_messages,
            temperature=0,
            max_tokens=1500
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error generating OpenAI response: {e}")
        return None

# Reading Webpages and Combining Documents Logic
def read_webpage_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator="\n")
            return text
        else:
            st.warning(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error reading content from {url}: {e}")
        return None

# Fetch the webpage contents if URLs are provided
documents = []
if url1:
    document1 = read_webpage_from_url(url1)
    if document1:
        documents.append(document1)
if url2:
    document2 = read_webpage_from_url(url2)
    if document2:
        documents.append(document2)

combined_documents = "\n\n".join(documents)  # Combine the contents of both URLs

# Prepare the context message with the documents to refer to
context_message = {"role": "system", "content": f"Here are the documents to refer to:\n{combined_documents}"}
st.session_state.messages.append(context_message)

# LLM Vendor Selection Logic
response = None
if llm_vendor == "Cohere":
    client = cohere.Client(api_key="your_cohere_api_key")
    response = generate_cohere_response(client, st.session_state.messages)
elif llm_vendor == "Gemini":
    client, is_valid, message = verify_gemini_key(api_key="your_gemini_api_key")
    if is_valid:
        response = generate_gemini_response(client, st.session_state.messages)
    else:
        st.error(message)
elif llm_vendor == "OpenAI 3.5":
    openai.api_key = "your_openai_api_key"
    model = "gpt-3.5-turbo"  # Specify the OpenAI model
    response = generate_openai_response(openai, st.session_state.messages, model)
elif llm_vendor == "OpenAI 4":
    openai.api_key = "your_openai_api_key"
    model = "GPT-4o-mini"  # Specify the OpenAI model
    response = generate_openai_response(openai, st.session_state.messages, model)

# Display response
if response:
    st.write(response)

# After every chat, append the new user message to session state messages
new_user_message = st.text_input("Your message:")
if new_user_message:
    st.session_state.messages.append({"role": "user", "content": new_user_message})
