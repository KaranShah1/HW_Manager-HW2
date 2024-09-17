import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
import google.generativeai as genai
from openai import OpenAI

# Function to read content from a URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Function to generate text using Cohere
def generate_text(prompt, api_key):
    co = cohere.Client(api_key)
    events = co.chat_stream(
        model="command-r",
        message=prompt,
        temperature=0,
        max_tokens=1500,
        prompt_truncation='AUTO',
        connectors=[],
        documents=[]
    )
    response_text = ""
    for event in events:
        if event.event_type == "text-generation":
            response_text += str(event.text)
    return response_text

# Function to generate a response using Google's Gemini
def google_dem(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    messages = [{'role': 'user', 'parts': [prompt]}]
    response = model.generate_content(messages)
    return response.text

# Streamlit app interface
st.title("Karan Shah📄 Document Question Answering")
st.write(
    "Enter a URL below and ask a question about it – GPT, Gemini, or Cohere will answer! "
    "To use this app, you need to provide an API key for OpenAI, Gemini, or Cohere."
)

# Fetch the API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai")
gemini_api_key = st.secrets.get("gemini")
cohere_api_key = st.secrets.get("cohere")

if not (openai_api_key or gemini_api_key or cohere_api_key):
    st.info("Please add your API keys for OpenAI, Gemini, or Cohere to continue.", icon="🗝")
else:
    # Input URL from the user
    url = st.text_input("Enter a URL to summarize and ask questions about:")

    # Sidebar options for selecting models and summaries
    st.sidebar.header("Summary Options")

    # Choose between GPT-4o-mini, Gemini, or Cohere
    model_option = st.sidebar.selectbox(
        "Choose the model:",
        ["GPT-4o-mini", "Gemini", "Cohere"]
    )

    # Choose summary option
    summary_option = st.sidebar.selectbox(
        "Choose a summary type:",
        ["Summarize in 100 words", "Summarize in 2 connecting paragraphs", "Summarize in 5 bullet points"]
    )

    # Dropdown menu for language selection
    language_option = st.selectbox(
        "Choose output language:",
        ["English", "French", "Spanish"]
    )

    # Ask the user for a question via st.text_area
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not url
    )

    if url and question:
        # Process the URL and extract its text content
        document = read_url_content(url)

        if document:
            # Adjust the prompt based on summary and language options
            if summary_option == "Summarize in 100 words":
                prompt = f"Summarize the following document in 100 words: {document}"
            elif summary_option == "Summarize in 2 connecting paragraphs":
                prompt = f"Summarize the following document in 2 connecting paragraphs: {document}"
            elif summary_option == "Summarize in 5 bullet points":
                prompt = f"Summarize the following document in 5 bullet points: {document}"

            # Add the language selection to the prompt
            prompt += f"\n\nOutput the summary in {language_option}."

            # If GPT-4o-mini is selected
            if model_option == "GPT-4o-mini":
                if openai_api_key:
                    client = OpenAI(api_key=openai_api_key)
                    model = "gpt-4o-mini"

                    messages = [
                        {
                            "role": "user",
                            "content": f"{prompt} \n\n---\n\n {question}",
                        }
                    ]

                    # Generate an answer using the OpenAI API
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                    )

                    # Stream the response to the app using st.write_stream
                    st.write_stream(stream)
                else:
                    st.error("Please add your OpenAI API key to use GPT-4o-mini.")

            # If Gemini is selected
            elif model_option == "Gemini":
                if gemini_api_key:
                    response_text = google_dem(prompt, gemini_api_key)
                    st.write(response_text)
                else:
                    st.error("Please add your Gemini API key to use Gemini.")

            # If Cohere is selected
            elif model_option == "Cohere":
                if cohere_api_key:
                    response_text = generate_text(prompt, cohere_api_key)
                    st.write(response_text)
                else:
                    st.error("Please add your Cohere API key to use Cohere.")
