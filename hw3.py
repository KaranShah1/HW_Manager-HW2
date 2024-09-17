import streamlit as st
import requests
from openai import OpenAI
import cohere

# Show title and description
st.title("LAB-03-Karan Shah📄 Document Question Answering and Chatbot")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Fetch the API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai")
gemini_api_key = st.secrets.get("gemini")
cohere_api_key = st.secrets.get("cohere")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "pdf"))

    # Sidebar options for summarizing 
    st.sidebar.title("Options")

    # Model selection
    model_option = st.sidebar.selectbox("Choose the model:", ["GPT-4o-mini", "Gemini", "Cohere"])

    # Summary options
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ),
    )

    if uploaded_file:
        # Process the uploaded file, handling potential decoding issues
        try:
            document = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            # If utf-8 decoding fails, use a more lenient approach
            document = uploaded_file.read().decode('utf-8', errors='replace')

        # Instruction based on user selection on the sidebar menu
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Generate the summary using the selected model (OpenAI, Gemini, or Cohere)
        if model_option == "GPT-4o-mini":
            stream = client.chat_completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
            )

            # Display the streamed output
            for chunk in stream:
                st.write(chunk.choices[0].message['content'])

        elif model_option == "Gemini":
            if gemini_api_key:
                response_text = google_dem(prompt=document, api_key=gemini_api_key)
                st.write(response_text)
            else:
                st.error("Please add your Gemini API key to use Gemini.")

        elif model_option == "Cohere":
            if cohere_api_key:
                response_text = generate_text(prompt=document, api_key=cohere_api_key)
                st.write(response_text)
            else:
                st.error("Please add your Cohere API key to use Cohere.")

    # Set up session state to hold chatbot messages with a buffer limit
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [{"role": "assistant", "content": "How can I help you?"}]

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
    response = co.generate(
        model="command-r",  # Use the correct model name
        prompt=prompt,
        temperature=0.5,  # Adjust temperature as needed
        max_tokens=500
    )
    return response.generations[0].text

def google_dem(prompt, api_key):
    # This function assumes Gemini API has a similar call pattern
    # Adjust this according to the actual SDK or API specifications for Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    messages = [{"role": "user", "parts": prompt}]
    response = model.generate_content(messages)
    return response.text

# Chatbot section
st.write("## Chatbot Interaction")
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input for the chatbot
if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response
    if model_option == "GPT-4o-mini" and openai_api_key:
        stream = client.chat_completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.chat_history,
            stream=True,
        )
        for chunk in stream:
            response_text = chunk.choices[0].message['content']
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").write(response_text)

    elif model_option == "Gemini" and gemini_api_key:
        response_text = google_dem(prompt, gemini_api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)

    elif model_option == "Cohere" and cohere_api_key:
        response_text = generate_text(prompt, cohere_api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        st.chat_message("assistant").write(response_text)

    else:
        st.error("Please provide the correct API key to use the selected model.")
