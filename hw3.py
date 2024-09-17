import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from openai import OpenAI

# Show title and description
st.title("LAB-03-Karan Shah📄 Document Question Answering and Chatbot")
st.write(
    "Upload a document, fetch content from a URL, and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Sidebar for URL inputs
    st.sidebar.title("User Input")
    url1 = st.sidebar.text_input("Enter URL 1:")
    url2 = st.sidebar.text_input("Enter URL 2 (optional):")

    # Sidebar for LLM selection
    st.sidebar.title("Choose the model:")
    llm_vendor = st.sidebar.selectbox(
        "Choose LLM:",
        ("OpenAI GPT-4", "GPT-4o-mini", "Gemini", "Cohere")
    )

    # Sidebar for conversation memory selection
    st.sidebar.title("Conversation Memory Type")
    memory_type = st.sidebar.selectbox(
        "Choose Memory Type:",
        ("Buffer of 5 questions", "Conversation summary", "Buffer of 5,000 tokens")
    )

    # Sidebar for file upload
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # Fetch content from a URL
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

    # Fetch content from URLs
    url1_content = fetch_url_content(url1)
    url2_content = fetch_url_content(url2)

    # Combine URL content and uploaded document as context
    document_content = uploaded_file.read().decode() if uploaded_file else ""
    context = f"URL 1 content: {url1_content}\n\nURL 2 content: {url2_content}\n\nDocument content: {document_content}"

    # Sidebar options for summarizing
    st.sidebar.title("Options")

    # # Model selection
    # openAI_model = st.sidebar.selectbox("Choose the GPT Model", ("mini", "regular"))
    # model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

    # Summary options
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ),
    )

    # Function to call the selected LLM
    def call_llm(question, llm_vendor, context):
        if llm_vendor == "OpenAI GPT-4":
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": context}, {"role": "user", "content": question}],
                stream=True
            )
            return response
        elif llm_vendor == "GPT-4o-mini":
            # Placeholder for GPT-4o-mini (Implement API)
            return "This is a placeholder response from GPT-4o-mini"
        elif llm_vendor == "Gemini":
            # Placeholder for Gemini API (Implement API)
            return "This is a placeholder response from Gemini"
        elif llm_vendor == "Cohere Command":
            cohere_api_key = st.secrets["cohere"]
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

    # # Example question from user
    # question = st.text_input("Ask your question:")
    # if question:
    #     response = call_llm(question, llm_vendor, context)

    #     # Stream the response (for OpenAI models)
    #     if llm_vendor == "OpenAI GPT-4":
    #         for message in response:
    #             st.write(message["choices"][0]["delta"].get("content", ""), end="")
    #     else:
    #         st.write(response)

    # # Set up the session state to hold chatbot messages with a buffer limit
    # if "chat_history" not in st.session_state:
    #     st.session_state["chat_history"] = [
    #         {"role": "assistant", "content": "How can I help you?"}
    #     ]

    # Define the conversation buffer size (2 user messages and 2 responses)
    conversation_buffer_size = 4  # 2 user messages + 2 assistant responses

    def manage_conversation_buffer():
        """Ensure the conversation buffer size does not exceed the limit."""
        if len(st.session_state.chat_history) > conversation_buffer_size:
            st.session_state.chat_history = st.session_state.chat_history[-conversation_buffer_size:]

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.chat_history:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input for the chatbot
    if prompt := st.chat_input("Ask the chatbot a question or interact:"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ensure the conversation buffer size does not exceed the limit
        manage_conversation_buffer()

        # Generate a response from OpenAI using the same model
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.chat_history,
            stream=True,
        )

        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        # Append the assistant's response to the session state
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Now, implement logic to ask, "Do you want more info?"
        if "yes" in prompt.lower():
            follow_up_response = "Great! Here's more information: ..."
        elif "no" in prompt.lower():
            follow_up_response = "Okay! Feel free to ask anything else."
        else:
            follow_up_response = "Do you want more info?"

        # Append the follow-up response and display
        st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
        st.chat_message("assistant").write(follow_up_response)

        manage_conversation_buffer()
