import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF for handling PDF files
import requests
from bs4 import BeautifulSoup
from googletrans import Translator  # For translation

# Function to read content from URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("Karan Shah📄 Document Question Answering")
st.write(
    "Upload a document or provide a URL, then ask a question – GPT or Claude 3 Opus will answer! "
    "You can choose the language of the response too."
)

# Fetch the OpenAI and Claude 3 Opus API keys from Streamlit secrets.
openai_api_key = st.secrets.get("somesection")
claude_3_opus_key = st.secrets.get("claude_3_opus")

if not openai_api_key and not claude_3_opus_key:
    st.info("Please add your OpenAI or Claude 3 Opus API key to continue.", icon="🗝")
else:
    # Sidebar options for selecting models and summaries
    st.sidebar.header("Summary Options")
    
    # Choose between GPT-4o-mini or Claude 3 Opus
    model_option = st.sidebar.selectbox(
        "Choose the model:",
        ["GPT-4o-mini", "Claude 3 Opus"]
    )
    
    # Choose summary option
    summary_option = st.sidebar.selectbox(
        "Choose a summary type:",
        ["Summarize in 100 words", "Summarize in 2 connecting paragraphs", "Summarize in 5 bullet points"]
    )

    # Dropdown menu to select the output language
    language_option = st.selectbox(
        "Select the output language:",
        ["English", "Hindi", "French"]
    )

    # Translator initialization (using googletrans library)
    translator = Translator()

    # URL input field
    url_input = st.text_input("Enter a URL to process:")

    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader(
        "Or upload a document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
    )

    # Ask the user for a question via st.text_area.
    question = st.text_area(
        "Now ask a question about the document or URL content!",
        placeholder="Can you give me a short summary?",
        disabled=not (uploaded_file or url_input),
    )

    # Process the URL or the uploaded file
    document = ""
    if url_input:
        document = read_url_content(url_input)
        if not document:
            st.error("Failed to read the content from the provided URL.")
    elif uploaded_file:
        if uploaded_file.type == "application/pdf":
            # If it's a PDF, extract the text using fitz
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    document += page.get_text()
        else:
            # Handle .txt or .md files
            document = uploaded_file.read().decode()

    if document and question:
        # Create the summary prompt based on user choice
        if summary_option == "Summarize in 100 words":
            prompt = f"Summarize the following document in 100 words: {document}"
        elif summary_option == "Summarize in 2 connecting paragraphs":
            prompt = f"Summarize the following document in 2 connecting paragraphs: {document}"
        elif summary_option == "Summarize in 5 bullet points":
            prompt = f"Summarize the following document in 5 bullet points: {document}"
        
        # Add the user's question to the prompt
        prompt += f"\n\n---\n\n {question}"
        
        # Handle GPT-4o-mini
        if model_option == "GPT-4o-mini":
            if openai_api_key:
                client = OpenAI(api_key=openai_api_key)
                model = "gpt-4o-mini"

                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

                # Generate an answer using the OpenAI API.
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                # Get the text response
                response_text = stream['choices'][0]['message']['content']

            else:
                st.error("Please add your OpenAI API key to use GPT-4o-mini.")
        
        # Handle Claude 3 Opus
        elif model_option == "Claude 3 Opus":
            if claude_3_opus_key:
                headers = {
                    "Authorization": f"Bearer {claude_3_opus_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "claude-v1",  # Assuming model name for Claude; adjust if needed
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }

                # Send request to Claude 3 Opus API
                response = requests.post(
                    "https://api.anthropic.com/v1/completions",  # Assuming endpoint; adjust if needed
                    headers=headers,
                    json=payload
                )

                if response.status_code == 200:
                    response_text = response.json()['completion']
                else:
                    st.error(f"Failed to get response from Claude 3 Opus: {response.status_code}")
            else:
                st.error("Please add your Claude 3 Opus API key to use Claude 3 Opus.")
        
        # Translate the response into the selected language
        if language_option != "English":
            translated_text = translator.translate(response_text, dest=language_option.lower()).text
        else:
            translated_text = response_text
        
        # Display the translated answer
        st.write(translated_text)
