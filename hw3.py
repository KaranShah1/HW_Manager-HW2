import os
import openai
import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
import google.generativeai as genai

# Sidebar Options
st.sidebar.title("LLM Interaction Settings")

# URL Input in Sidebar
url1 = st.sidebar.text_input("Enter first URL:")
url2 = st.sidebar.text_input("Enter second URL:")

# LLM Vendor Selection in Sidebar
llm_vendor = st.sidebar.selectbox(
    "Select LLM Vendor", 
    ("Cohere", "Gemini", "OpenAI")
)

# Conversation Memory Type Selection in Sidebar
memory_type = st.sidebar.selectbox(
    "Select Conversation Memory Type", 
    ("Buffer of 5 questions", "Conversation Summary", "Buffer of 5,000 tokens")
)

# Session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to handle the conversation memory logic
def handle_memory(messages, memory_type):
    if memory_type == "Buffer of 5 questions":
        return messages[-5:]
    elif memory_type == "Conversation Summary":
        # Create a simple summary of the conversation
        summary = " ".join([msg['content'] for msg in messages])
        return [{"role": "system", "content": summary}]
    elif memory_type == "Buffer of 5,000 tokens":
        # Placeholder for token-count-based memory logic (OpenAI-specific logic may apply)
        return messages  # Assuming message length doesn't exceed 5,000 tokens for now

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
        # Apply memory handling based on the selected memory type
        msgs = handle_memory(messages, memory_type)

        # Prepare the message history with "user" and "model" roles
        formatted_msgs = [{"role": "user" if msg['role'] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in msgs]

        # Generate the response from Gemini API
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
        # Apply memory handling based on the selected memory type
        chat_history = handle_memory(messages, memory_type)

        # Format messages for OpenAI's API
        formatted_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history
        ]

        # Generate the response from OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=formatted_messages,
            temperature=0,
            max_tokens=1500
        )
        
        return response.choices[0].message['content']
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Reading Webpages and Combining Documents Logic
def read_webpage_from_url(url):
    try:
        # Send an HTTP request to the URL
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract the text content of the webpage
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
if llm_vendor == "Cohere":
    # Add logic to handle Cohere-specific API calls
    client = cohere.Client(api_key="your_cohere_api_key")
    messages = [{"role": "user", "content": "Hello"}]  # Example messages
    response = generate_cohere_response(client, st.session_state.messages)
elif llm_vendor == "Gemini":
    # Add logic to handle Gemini-specific API calls
    client, is_valid, message = verify_gemini_key(api_key="your_gemini_api_key")
    if is_valid:
        response = generate_gemini_response(client, st.session_state.messages)
    else:
        st.error(message)
elif llm_vendor == "OpenAI":
    # Handle OpenAI-specific API calls
    openai.api_key = "your_openai_api_key"
    model = "gpt-3.5-turbo"  # Specify the OpenAI model
    response = generate_openai_response(openai, st.session_state.messages, model)

# Display response
if response:
    st.write(response)

# After every chat, append the new user message to session state messages
new_user_message = st.text_input("Your message:")
if new_user_message:
    st.session_state.messages.append({"role": "user", "content": new_user_message})
