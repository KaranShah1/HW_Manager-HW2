import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import cohere
import google.generativeai as genai

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

# Define the generate_text function using Cohere
def generate_text(prompt, api_key):
    co = cohere.Client(api_key)
    events = co.chat_stream(
        model="command-r",  # Use the correct model name
        message=prompt,
        temperature=0,  # Adjust temperature as needed
        max_tokens=1500,
        prompt_truncation='AUTO',
        connectors=[],
        documents=[]
    )
    response_text = ""
    for event in events:
        if event.event_type == "text-generation":
            response_text = response_text + str(event.text)

    return response_text

def google_dem(question_to_ask, api_key):
    
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-pro')

    messages = []

    gem_message = "\nPlease answer the following question: \n" + str(question_to_ask)

    messages.append({'role': 'user', 'parts': gem_message})
    response = model.generate_content(messages)

    return response.text


# Show title and description
st.title("Karan Shah📄 Document Question Answering")
st.write(
    "Enter a URL below and ask a question about it – GPT, Gemini, or Cohere will answer! "
    "To use this app, you need to provide an API key for OpenAI, Gemini, or Cohere."
)

# Fetch the API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai")
#claude_3_opus_key = st.secrets.get("claude_3_opus")
gemini_api_key = st.secrets.get("gemini")
cohere_api_key = st.secrets.get("cohere")

if not (openai_api_key or claude_3_opus_key or gemini_api_key or cohere_api_key):
    st.info("Please add your API keys for OpenAI, Gemini, or Cohere to continue.", icon="🗝")
else:
    # Input URL from the user
    url = st.text_input("Enter a URL to summarize and ask questions about:")

    # Sidebar options for selecting models and summaries
    st.sidebar.header("Summary Options")

    # Choose between GPT-4o-mini, Gemini or Cohere
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
                    from openai import OpenAI
                    client = OpenAI(api_key=openai_api_key)
                    model = "gpt-4o-mini"

                    messages = [
                        {
                            "role": "user",
                            "content": f"{prompt} \n\n---\n\n {question}",
                        }
                    ]

                    # Generate an answer using the OpenAI API.
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                    )

                    # Stream the response to the app using st.write_stream.
                    st.write_stream(stream)
                else:
                    st.error("Please add your OpenAI API key to use GPT-4o-mini.")

            # # If Claude 3 Opus is selected
            # elif model_option == "Claude 3 Opus":
            #     if claude_3_opus_key:
            #         headers = {
            #             "Authorization": f"Bearer {claude_3_opus_key}",
            #             "Content-Type": "application/json"
            #         }
            #         payload = {
            #             "model": "claude-v1",  # Assuming model name for Claude; adjust if needed
            #             "messages": [
            #                 {"role": "user", "content": f"{prompt} \n\n---\n\n {question}"}
            #             ]
            #         }

            #         # Send request to Claude 3 Opus API
            #         response = requests.post(
            #             "https://api.anthropic.com/v1/completions",  # Assuming endpoint; adjust if needed
            #             headers=headers,
            #             json=payload
            #         )

            #         if response.status_code == 200:
            #             # Display response content
            #             st.write(response.json().get('completion', 'No completion found'))
            #         else:
            #             st.error(f"Failed to get response from Claude 3 Opus: {response.status_code}")
            #     else:
            #         st.error("Please add your Claude 3 Opus API key to use Claude 3 Opus.")

            # If Gemini is selected
            elif model_option == "Gemini":
                if gemini_api_key:
                    # Generate text using Cohere
                    response_text = google_dem(prompt, gemini_api_key)
                    st.write(response_text)
                else:
                    st.error("Please add your Cohere API key to use Cohere.")
                
     
            # If Cohere is selected
            elif model_option == "Cohere":
                if cohere_api_key:
                    # Generate text using Cohere
                    response_text = generate_text(prompt, cohere_api_key)
                    st.write(response_text)
                else:
                    st.error("Please add your Cohere API key to use Cohere.")
