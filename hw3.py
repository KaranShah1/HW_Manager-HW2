import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai

# Function to fetch content from a URL
def fetch_url_content(url):
    if not url:
        return ""
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

# Sidebar for URL inputs
st.sidebar.title("User Input")
url1 = st.sidebar.text_input("Enter URL 1:")
url2 = st.sidebar.text_input("Enter URL 2 (optional):")

# Sidebar for LLM selection
st.sidebar.title("Choose LLM Vendor")
llm_vendor = st.sidebar.selectbox(
    "Choose LLM:",
    ("OpenAI GPT-4", "GPT-4o-mini", "Gemini", "Cohere Command")
)

# Sidebar for conversation memory selection
st.sidebar.title("Conversation Memory Type")
memory_type = st.sidebar.selectbox(
    "Choose Memory Type:",
    ("Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens")
)

# Fetch content from URLs
url1_content = fetch_url_content(url1)
url2_content = fetch_url_content(url2)

# Define a function to call the selected LLM
def call_llm(question, llm_vendor, context):
    # Use OpenAI API for GPT models
    if llm_vendor == "OpenAI GPT-4":
        openai.api_key = st.secrets["openai_api_key"]  # Add OpenAI API Key in Streamlit secrets
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": context}, {"role": "user", "content": question}],
            stream=True
        )
        return response

    elif llm_vendor == "GPT-4o-mini":
        # Call to GPT-4o-mini API (pseudo-code, implement as per your API documentation)
        # You'll need the API key and endpoint for GPT-4o-mini
        response = "This is a placeholder response from GPT-4o-mini"
        return response

    elif llm_vendor == "Gemini":
        # Call to Gemini API (pseudo-code, implement as per your API documentation)
        # Implement your Gemini API key and logic here
        response = "This is a placeholder response from Gemini"
        return response

    elif llm_vendor == "Cohere Command":
        # Call to Cohere API
        cohere_api_key = st.secrets["cohere_api_key"]
        headers = {
            "Authorization": f"Bearer {cohere_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": context + "\nUser question: " + question,
            "model": "command-xlarge-nightly",
            "max_tokens": 300,
            "temperature": 0.75
        }
        response = requests.post(
            "https://api.cohere.ai/generate",
            json=payload,
            headers=headers
        ).json()
        return response['generations'][0]['text']

    else:
        return "Invalid LLM Vendor"

# Combine URL content and conversation history as context
context = f"URL 1 content: {url1_content}\n\nURL 2 content: {url2_content}"

# Example question from user
question = st.text_input("Ask your question:")
if question:
    response = call_llm(question, llm_vendor, context)

    # Stream the response (for OpenAI models)
    if llm_vendor == "OpenAI GPT-4":
        for message in response:
            st.write(message["choices"][0]["delta"].get("content", ""), end="")
    else:
        st.write(response)
