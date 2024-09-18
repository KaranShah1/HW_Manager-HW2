import streamlit as st
import cohere
import genai
from openai import OpenAI

# Fetch the API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai")
cohere_api_key = st.secrets.get("cohere")
gemini_api_key = st.secrets.get("gemini")

# Show title and description.
st.title("HW-03-Karan Shah📄 Document question answering and Chatbot")
st.write(
    "Upload a document below and ask a question about it – GPT-3.5, GPT-4, Cohere, or Gemini will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an API key for OpenAI, Cohere, or Gemini."
)

# Dropdown to select the LLM model (OpenAI 3.5, OpenAI 4.0, Cohere, Gemini)
model_option = st.sidebar.selectbox("Choose the LLM Model", ["OpenAI GPT-3.5", "OpenAI GPT-4", "Cohere", "Gemini"])

# Dropdown for language selection
language_option = st.sidebar.selectbox("Choose output language:", ["English", "French", "Spanish"])

# File uploader for document input
uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

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

# Define the google_dem function using Gemini
def google_dem(question_to_ask, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    messages = []
    gem_message = "\nPlease answer the following question: \n" + str(question_to_ask)
    messages.append({'role': 'user', 'parts': gem_message})
    response = model.generate_content(messages)
    return response.text

# Function to handle OpenAI GPT models (3.5 and 4)
def openai_chat(prompt, api_key, model_name):
    client = OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1500,
        temperature=0,
    )
    return response['choices'][0]['message']['content']

# Function to translate text based on language selection (optional - can integrate a translation API if needed)
def translate_text(text, target_language):
    if target_language == "English":
        return text
    elif target_language == "French":
        # Add your translation logic here
        return "Translated to French: " + text
    elif target_language == "Spanish":
        # Add your translation logic here
        return "Translated to Spanish: " + text

# Ensure OpenAI, Cohere, and Gemini work alongside existing features
if openai_api_key or cohere_api_key or gemini_api_key:
    # Let the user input a prompt or interact with a chatbot
    prompt = st.text_input("Ask a question or interact with the chatbot:")

    if prompt:
        response_text = ""
        
        # If OpenAI GPT-3.5 is selected
        if model_option == "OpenAI GPT-3.5" and openai_api_key:
            response_text = openai_chat(prompt, openai_api_key, "gpt-3.5-turbo")
        
        # If OpenAI GPT-4 is selected
        elif model_option == "OpenAI GPT-4" and openai_api_key:
            response_text = openai_chat(prompt, openai_api_key, "gpt-4")
        
        # If Cohere is selected
        elif model_option == "Cohere" and cohere_api_key:
            response_text = generate_text(prompt, cohere_api_key)
        
        # If Gemini is selected
        elif model_option == "Gemini" and gemini_api_key:
            response_text = google_dem(prompt, gemini_api_key)
        
        # Translate based on language selection
        response_text = translate_text(response_text, language_option)
        
        # Display the response
        st.write(response_text)
    
    # Display the document summary (if a document is uploaded)
    if uploaded_file:
        document = uploaded_file.read().decode()
        st.write("### Document Content:")
        st.text(document)

        # Choose the summary style
        summary_style = st.radio(
            "Select a summary style:",
            ["Summarize the document in 100 words", "Summarize in 2 paragraphs", "Summarize in 5 bullet points"]
        )

        summary_prompt = f"Summarize the following document: {document} in style: {summary_style}"

        # Summarize using the selected model
        if model_option == "OpenAI GPT-3.5":
            summary = openai_chat(summary_prompt, openai_api_key, "gpt-3.5-turbo")
            st.write(summary)

        elif model_option == "OpenAI GPT-4":
            summary = openai_chat(summary_prompt, openai_api_key, "gpt-4")
            st.write(summary)

        elif model_option == "Cohere":
            summary = generate_text(summary_prompt, cohere_api_key)
            st.write(summary)

        elif model_option == "Gemini":
            summary = google_dem(summary_prompt, gemini_api_key)
            st.write(summary)
else:
    st.error("Please add at least one API key to use this app.")
