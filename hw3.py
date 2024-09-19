import streamlit as st
from openai import OpenAI
import tiktoken  # Tokenizer from OpenAI

# Set a maximum token limit for the buffer (you can adjust this based on your needs).
max_tokens = 2048
summary_threshold = 5  # Number of messages before we start summarizing

# # Function to generate Gemini response
# def generate_gemini_response(client, messages):
#     try:
#         msgs = handle_memory(messages, memory_type)
#         formatted_msgs = [{"role": "user" if msg['role'] == "user" else "model", "parts": [{"text": msg["content"]}]} for msg in msgs]
#         response = client.generate_content(
#             contents=formatted_msgs,
#             generation_config=genai.types.GenerationConfig(
#                 temperature=0,
#                 max_output_tokens=1500,
#             )
#         )
#         return response.generations[0].text
#     except Exception as e:
#         st.error(f"Error generating Gemini response: {e}")
#         return None
# def generate_cohere_response(client, messages):
#     try:
#         response = client.chat(messages=[msg['content'] for msg in messages])
#         return response.generations[0].text
#     except Exception as e:
#         st.error(f"Error generating Cohere response: {e}")
#         return None
# Function to generate OpenAI response
# def generate_openai_response(client, messages, model):
#     try:
#         chat_history = handle_memory(messages, memory_type)
#         formatted_messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
#         response = openai.ChatCompletion.create(
#             model=model,
#             messages=formatted_messages,
#             temperature=0,
#             max_tokens=1500
#         )
#         return response.choices[0].message['content']
#     except Exception as e:
#         st.error(f"Error generating OpenAI response: {e}")
#         return None
        
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
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    assistant_messages = [msg["content"] for msg in messages if msg["role"] == "assistant"]
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

# Show title and description.
st.title("Karan Shah 📄 Chatbot Interaction")
st.write(
    "Interact with the chatbot! "
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

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

    # Based on provider selection, update model options
    if llm_provider == "OpenAI":
        if use_advanced:
            model_to_use = "gpt-4o"
        else:
            model_to_use = "gpt-4o-mini"
    elif llm_provider == "Cohere":
        # if use_advanced:
        #     model_to_use = "command-r"
        # else:
            model_to_use = "command-r"
    elif llm_provider == "gemini":
        # if use_advanced:
        #     model_to_use = "mistral-advanced"
        # else:
            model_to_use = "Gemini"

    # Toggle the checkbox automatically
    if use_advanced and model_to_use.endswith("mini"):
        st.sidebar.warning("You've selected a basic model, 'Use advanced model' will be unchecked.")
        use_advanced = False
    elif not use_advanced and model_to_use.endswith("advanced"):
        st.sidebar.warning("Advanced model is selected.")

    # Condition to check if at least one URL is provided
    if not url1 and not url2:
        st.sidebar.warning(
            "Please provide at least one URL to interact with the chatbot.")
    else:
        # Set up the session state to hold chatbot messages with a token-based buffer
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {"role": "assistant", "content": "How can I help you?"}
            ]
        if "conversation_summary" not in st.session_state:
            st.session_state["conversation_summary"] = ""  # Initialize summary

        # Display the chatbot conversation
        st.write("## Chatbot Interaction")
        for msg in st.session_state.chat_history:
            chat_msg = st.chat_message(msg["role"])
            chat_msg.write(msg["content"])

        # Get user input for the chatbot
        if prompt := st.chat_input("Ask the chatbot a question related to the URLs provided:"):
            # Ensure that the question references the URLs
            if url1 and url2:
                prompt_with_urls = f"Refer to these URLs in your response: {url1} and {url2}. {prompt}"
            elif url1:
                prompt_with_urls = f"Refer to this URL in your response: {url1}. {prompt}"
            else:
                prompt_with_urls = f"Refer to this URL in your response: {url2}. {prompt}"

#combined_documents = "\n\n".join(documents)  # Combine the contents of both URLs
            # Append the user input to the session state
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt})

            # Display the user input in the chat
            with st.chat_message("user"):
                st.markdown(prompt)

            # Conversation memory logic based on memory type
            if memory_type == "Buffer of 5000 tokens":
                truncated_messages, total_tokens = truncate_messages_by_tokens(
                    st.session_state.chat_history, max_tokens, model_name=model_to_use
                )
                st.session_state.chat_history = truncated_messages

            elif memory_type == "Conversation Summary":
                if len(st.session_state.chat_history) > summary_threshold:
                    st.session_state["conversation_summary"] = summarize_conversation(
                        st.session_state.chat_history, model_to_use, client
                    )
                    st.session_state.chat_history = [
                        {"role": "system", "content": st.session_state["conversation_summary"]}
                    ] + st.session_state.chat_history[-2:]  # Keep recent messages

            elif memory_type == "Buffer of 5 questions":
                if len(st.session_state.chat_history) > 5:
                    st.session_state["conversation_summary"] = summarize_conversation(
                        st.session_state.chat_history[:5], model_to_use, client
                    )
                    st.session_state.chat_history = [
                        {"role": "system", "content": st.session_state["conversation_summary"]}
                    ] + st.session_state.chat_history[-5:]

            # Generate a response from the selected LLM provider using the appropriate model
            simple_prompt = f"Based on the provided URLs, answer the following question: {prompt_with_urls}"
            messages_for_gpt = st.session_state.chat_history.copy()
            messages_for_gpt[-1]['content'] = simple_prompt

            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_gpt,
                stream=True,
            )

            # Stream the assistant's response
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            # Append the assistant's response to the session state
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

            # Handle follow-up questions
            if "yes" in prompt.lower():
                st.session_state.chat_history.append(
                    {"role": "assistant",
                        "content": "Here's more information. Do you want more info?"}
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
