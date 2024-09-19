import streamlit as st
import requests
from openai import OpenAI
from bs4 import BeautifulSoup
import tiktoken  # Tokenizer from OpenAI


# Function to read the content from a URL
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Dictionary to map LLM providers to their respective models
model_options = {
    "OpenAI": {"basic": "gpt-3.5-turbo", "advanced": "gpt-4o"},
    "Claude": {"basic": "claude-3-haiku-20240307", "advanced": "claude-3-5-sonnet-20240620"},
    "Mistral": {"basic": "open-mistral-7b", "advanced": "mistral-large-2407"},
}

# Set a maximum token limit for the buffer (you can adjust this based on your needs).
max_tokens = 4000
summary_threshold = 5  # Number of messages before we start summarizing

# Function to calculate tokens for a message using OpenAI tokenizer
def calculate_token_count(messages, model_name="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["content"]))
    return total_tokens

# Truncate conversation history to fit within max_tokens
def truncate_messages_by_tokens(messages, max_tokens, model_name="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model_name)
    total_tokens = 0
    truncated_messages = []

    # Always retain the last user-assistant pair
    recent_pair = messages[-2:] if len(messages) >= 2 else messages

    # Calculate the token count for the most recent pair
    for message in recent_pair:
        total_tokens += len(encoding.encode(message["content"]))

    # Traverse the older messages in reverse order (newest to oldest)
    for message in reversed(messages[:-2]):  # Exclude the most recent pair
        message_token_count = len(encoding.encode(message["content"]))

        # Add message if it doesn't exceed the max_tokens limit
        if total_tokens + message_token_count <= max_tokens:
            truncated_messages.insert(0, message)
            total_tokens += message_token_count
        else:
            break

    truncated_messages.extend(recent_pair)
    return truncated_messages, total_tokens

def summarize_conversation(messages, model_to_use, client):
    user_messages = [msg["content"]
                     for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg["content"]
                          for msg in messages if msg["role"] == "assistant"]
    conversation_summary_prompt = f"Summarize this conversation: \n\nUser: {user_messages} \nAssistant: {assistant_messages}"

    # Call LLM to summarize
    summary_response = client.chat.completions.create(
        model=model_to_use,
        messages=[{"role": "system", "content": conversation_summary_prompt}],
        stream=False,
    )

    # Extract the summary content from the response structure
    summary_content = summary_response.choices[0].message.content

    return summary_content

# Separate function for calling OpenAI models
def call_openai_model(prompt_with_urls, model_name, api_key, chat_history):
    # Create an OpenAI client
    client = OpenAI(api_key=api_key)
    messages_for_llm = chat_history.copy()
    messages_for_llm[-1]['content'] = prompt_with_urls

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages_for_llm,
        stream=True,
    )

    return stream

# Separate function for calling Claude models
def call_claude_model(prompt_with_urls, model_name, api_key, chat_history):
    anthropic_client = Anthropic(api_key=api_key)
    messages_for_llm = chat_history.copy()
    messages_for_llm[-1]['content'] = prompt_with_urls

    st.write(messages_for_llm)

    stream = anthropic_client.messages.create(
        model=model_name,
        max_tokens=256,
        messages=messages_for_llm,
        stream=True,
        temperature=0.5
    )

    return stream

# Separate function for calling Mistral models
def call_mistral_model(prompt_with_urls, model_name, client, chat_history):
    messages_for_llm = chat_history.copy()
    messages_for_llm[-1]['content'] = prompt_with_urls

    stream = client.chat.completions.create(
        model=model_name,
        messages=messages_for_llm,
        stream=True,
    )

    return stream

# Main Streamlit code
st.title(" -- Testing 📄 Chatbot Interaction")
st.write("Interact with the chatbot!")

# Sidebar options
st.sidebar.title("Options")

# Input fields for two URLs
url1 = st.sidebar.text_input("Enter URL 1", value="")
url2 = st.sidebar.text_input("Enter URL 2", value="")

# Add option to select LLM provider
llm_provider = st.sidebar.selectbox(
    "Choose LLM Provider",
    ("OpenAI", "Gemini", "Cohere")
)

# Checkboxes for advanced models
use_advanced = st.sidebar.checkbox("Use advanced model", value=False)

# Memory selection options
memory_type = st.sidebar.selectbox(
    "Select Conversation Memory Type",
    ("Buffer of 5 questions", "Conversation Summary", "Buffer of 5000 tokens")
)

client = OpenAI(api_key=st.secrets["openai"])

# Based on provider selection and use_advanced flag, update model options
model_to_use = model_options[llm_provider]["advanced" if use_advanced else "basic"]

# Condition to check if at least one URL is provided
if not url1 and not url2:
    st.sidebar.warning("Please provide at least one URL to interact with the chatbot.")
else:
    # Set up the session state to hold chatbot messages with a token-based buffer
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    if "conversation_summary" not in st.session_state:
        st.session_state["conversation_summary"] = ""  # Initialize summary

    # Fetch content from the provided URLs
    url1_content = read_url_content(url1) if url1 else ""
    url2_content = read_url_content(url2) if url2 else ""

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.chat_history:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input for the chatbot
    if prompt := st.chat_input("Ask the chatbot a question related to the URLs provided:"):
        # Add content from the URLs to the prompt
        prompt_with_urls = f"Refer to the content from the provided URLs in your response. \n\n{url1_content}\n\n{url2_content}\n\n{prompt}"

        # Append the user input to the session state
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display the user input in the chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Conversation memory logic based on memory type
        if memory_type == "Buffer of 5000 tokens":
            truncated_messages, total_tokens = truncate_messages_by_tokens(
                st.session_state.chat_history, max_tokens, model_name=model_to_use
            )
            st.session_state.chat_history = truncated_messages

        elif memory_type == "Conversation Summary" and len(st.session_state.chat_history) > summary_threshold:
            st.session_state["conversation_summary"] = summarize_conversation(
                st.session_state.chat_history, model_to_use, client
            )
            st.session_state.chat_history = [
                {"role": "system", "content": st.session_state["conversation_summary"]}
            ] + st.session_state.chat_history[-2:]

        elif memory_type == "Buffer of 5 questions" and len(st.session_state.chat_history) > 5:
            st.session_state["conversation_summary"] = summarize_conversation(
                st.session_state.chat_history[:5], model_to_use, client
            )
            st.session_state.chat_history = [
                {"role": "system", "content": st.session_state["conversation_summary"]}
            ] + st.session_state.chat_history[-5:]

        # Generate a response from the selected LLM provider using the appropriate model
        if llm_provider == "OpenAI":
            stream = call_openai_model(
                prompt_with_urls, model_to_use, st.secrets["openai_api_key"], st.session_state.chat_history
            )
        elif llm_provider == "Claude":
            stream = call_cohere_model(
                prompt_with_urls, model_to_use, st.secrets["cohere_api_key"], st.session_state.chat_history
            )
        elif llm_provider == "Mistral":
            stream = call_gemini_model(
                prompt_with_urls, model_to_use, st.secrets["gemini_api_key"], st.session_state.chat_history
            )

        # Stream the assistant's response
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        # Append the assistant's response to the session state
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Handle follow-up questions
        if "yes" in prompt.lower():
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Here's more information. Do you want more info?"}
            )
            with st.chat_message("assistant"):
                st.markdown("Here's more information. Do you want more info?")
        elif "no" in prompt.lower():
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "What question can I help with next?"}
            )
            with st.chat_message("assistant"):
                st.markdown("What question can I help with next?")
        else:
            follow_up_question = "Do you want more info?"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": follow_up_question})
            with st.chat_message("assistant"):
                st.markdown(follow_up_question)
