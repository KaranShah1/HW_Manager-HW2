import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF for handling PDF files
import requests

# Show title and description.
st.title("Karan Shah📄 Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – GPT or Claude 3 Opus will answer! "
    "To use this app, you need to provide an OpenAI or Claude 3 Opus API key."
)

# Fetch the OpenAI and Claude 3 Opus API keys from Streamlit secrets.
openai_api_key = st.secrets.get("somesection")
claude_3_opus_key = st.secrets.get("claude_3_opus")

if not openai_api_key and not claude_3_opus_key:
    st.info("Please add your OpenAI or Claude 3 Opus API key to continue.", icon="🗝")
else:
    # Sidebar options for selecting models and summaries
    st.sidebar.header("Summary Options")
    
    # Choose between GPT-4o, GPT-4o-mini, or Claude 3 Opus
    model_option = st.sidebar.selectbox(
        "Choose the model:",
        ["GPT-4o", "GPT-4o-mini", "Claude 3 Opus"]
    )
    
    # Choose summary option
    summary_option = st.sidebar.selectbox(
        "Choose a summary type:",
        ["Summarize in 100 words", "Summarize in 2 connecting paragraphs", "Summarize in 5 bullet points"]
    )

    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, or .pdf)", type=("txt", "md", "pdf")
    )

    # Ask the user for a question via st.text_area.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question.
        if uploaded_file.type == "application/pdf":
            # If it's a PDF, extract the text using fitz
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                document = ""
                for page in doc:
                    document += page.get_text()

        else:
            # Handle .txt or .md files
            document = uploaded_file.read().decode()

        # Create the summary prompt
        if summary_option == "Summarize in 100 words":
            prompt = f"Summarize the following document in 100 words: {document}"
        elif summary_option == "Summarize in 2 connecting paragraphs":
            prompt = f"Summarize the following document in 2 connecting paragraphs: {document}"
        elif summary_option == "Summarize in 5 bullet points":
            prompt = f"Summarize the following document in 5 bullet points: {document}"
        
        # Handle GPT models (GPT-4o or GPT-4o-mini)
        if "GPT" in model_option:
            client = OpenAI(api_key=openai_api_key)
            model = "gpt-4o" if model_option == "GPT-4o" else "gpt-4o-mini"
            
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
        
        # Handle Claude 3 Opus model
        elif model_option == "Claude 3 Opus":
            # Set up Claude API request structure
            headers = {
                "Authorization": f"Bearer {claude_3_opus_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-v1",  # Assuming model name for Claude; adjust if needed
                "messages": [
                    {"role": "user", "content": f"{prompt} \n\n---\n\n {question}"}
                ]
            }

            # Send request to Claude 3 Opus API
            response = requests.post(
                "https://api.anthropic.com/v1/completions",  # Assuming endpoint; adjust if needed
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                # Display response content
                st.write(response.json()['completion'])
            else:
                st.error("Failed to get response from Claude 3 Opus.")
