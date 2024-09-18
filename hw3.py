import streamlit as st
from openai import OpenAI
import genai  # Gemini API
import cohere  # Cohere API

# Show title and description
st.title("HW-03-Karan Shah📄 Document question answering and Chatbot")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Fetch API keys from Streamlit secrets
openai_api_key = st.secrets.get("openai")
gemini_api_key = st.secrets.get("gemini")
cohere_api_key = st.secrets.get("cohere")

# Sidebar options for summarizing
st.sidebar.title("Options")

# Model selection
model_option = st.sidebar.selectbox(
    "Choose the LLM Model",
    ("OpenAI GPT-3.5", "OpenAI GPT-4o", "Gemini", "Cohere")
)

# Memory options for conversation
memory_option = st.sidebar.radio(
    "Choose memory type",
    ("Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens")
)

# Input for two URLs
url_1 = st.sidebar.text_input("Enter the first URL:")
url_2 = st.sidebar.text_input("Enter the second URL:")

# Define summary options
summary_options = st.sidebar.radio(
    "Select a format for summarizing the document:",
    ("Summarize the document in 100 words",
     "Summarize the document in 2 connecting paragraphs",
     "Summarize the document in 5 bullet points"),
)

# If a file is uploaded
uploaded_file = st.file_uploader("Upload a document (.txt or .pdf)", type=("txt", "md"))

if uploaded_file:
    document = uploaded_file.read().decode()

    # Instruction based on the selected summary format
    instruction = f"Summarize the document in {summary_options.lower()}."

    # Prepare the messages
    messages = [
        {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {instruction}"}
    ]

    # Handle OpenAI models (GPT-3.5, GPT-4o)
    if model_option.startswith("OpenAI"):
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.", icon="🗝")
        else:
            client = OpenAI(api_key=openai_api_key)
            stream = client.chat.completions.create(model=model_option.split(" ")[1], messages=messages, stream=True)
            st.write_stream(stream)

# Define the function for Gemini (Google)
def google_dem(question_to_ask, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    messages = [{'role': 'user', 'parts': "\nPlease answer the following question: \n" + str(question_to_ask)}]
    response = model.generate_content(messages)
    return response.text

# Define the generate_text function using Cohere
def generate_text(prompt, api_key):
    co = cohere.Client(api_key)
    events = co.chat_stream(model="command-r", message=prompt, temperature=0, max_tokens=1500)
    response_text = ""
    for event in events:
        if event.event_type == "text-generation":
            response_text += str(event.text)
    return response_text

# If a conversation prompt is given
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [{"role": "assistant", "content": "How can I help you?"}]

conversation_buffer_size = 4  # Set conversation buffer size

def manage_conversation_buffer():
    """Ensure the conversation buffer size does not exceed the limit."""
    if len(st.session_state.chat_history) > conversation_buffer_size:
        st.session_state.chat_history = st.session_state.chat_history[-conversation_buffer_size:]

# Display chatbot conversation
st.write("## Chatbot Interaction")
for msg in st.session_state.chat_history:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Get user input for chatbot
if prompt := st.chat_input("Ask the chatbot a question or interact:"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.markdown(prompt)

    manage_conversation_buffer()

    # Handle responses for different LLM models
    if model_option == "Gemini":
        if gemini_api_key:
            response_text = google_dem(prompt, gemini_api_key)
        else:
            st.error("Please add your Gemini API key to use Gemini.")
    elif model_option == "Cohere":
        if cohere_api_key:
            response_text = generate_text(prompt, cohere_api_key)
        else:
            st.error("Please add your Cohere API key to use Cohere.")
    else:  # Handle OpenAI models (GPT-3.5, GPT-4o)
        client = OpenAI(api_key=openai_api_key)
        stream = client.chat.completions.create(model=model_option.split(" ")[1], messages=st.session_state.chat_history, stream=True)
        response_text = st.write_stream(stream)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})

    # Now, implement the logic to ask, "Do you want more info?"
    follow_up_response = "Do you want more info?"
    st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
    st.chat_message("assistant").write(follow_up_response)

    manage_conversation_buffer()

# Use URLs in chatbot response
if url_1 and url_2:
    st.write(f"Using information from: {url_1} and {url_2}")
    # Implement URL processing and use it in conversation (e.g., scraping info and summarizing)
    # This can be done using a separate library like requests or BeautifulSoup if needed.
