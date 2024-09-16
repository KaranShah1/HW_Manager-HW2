import streamlit as st
from openai import OpenAI

# Title and description
st.title("LAB-03-Karan Shah📄 Document question answering and Chatbot")
st.write("Interact with the chatbot. Ask a question and get an easy-to-understand answer! Add your OpenAI API key to use the app.")

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Set up session state to store chat history and if more info is needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hi! How can I help you today?"}
        ]

    if "awaiting_more_info" not in st.session_state:
        st.session_state.awaiting_more_info = False

    def manage_conversation_buffer():
        """Ensure the conversation buffer does not exceed 2 user messages and 2 responses."""
        conversation_buffer_size = 4  # 2 user messages + 2 assistant responses
        if len(st.session_state.chat_history) > conversation_buffer_size:
            st.session_state.chat_history = st.session_state.chat_history[-conversation_buffer_size:]

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.chat_history:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input
    if prompt := st.chat_input("Ask the chatbot a question:"):
        if st.session_state.awaiting_more_info:
            # Handle yes or no response for "Do you want more info?"
            if prompt.lower() in ["yes", "y"]:
                # Provide more information in simple terms
                more_info = "Here is more information: Think of it this way... (simple explanation)."
                st.session_state.chat_history.append({"role": "user", "content": "Yes, I want more info."})
                st.session_state.chat_history.append({"role": "assistant", "content": more_info})

                # Ask again if more info is needed
                st.session_state.chat_history.append({"role": "assistant", "content": "Do you want more info?"})

            else:
                # User said 'no', return to normal conversation
                st.session_state.chat_history.append({"role": "user", "content": "No, I don't need more info."})
                st.session_state.chat_history.append({"role": "assistant", "content": "What else can I help you with?"})
                st.session_state.awaiting_more_info = False  # Resetting after response
        else:
            # Append user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Generate an answer from OpenAI in simple terms
            messages = st.session_state.chat_history
            stream = client.chat.completions.create(
                model="gpt-4-o",
                messages=messages,
                stream=True,
            )

            # Stream the assistant's response
            with st.chat_message("assistant"):
                response = st.write_stream(stream)

            # Append the response to the chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            # Ask if the user wants more info
            st.session_state.chat_history.append({"role": "assistant", "content": "Do you want more info?"})
            st.session_state.awaiting_more_info = True

        # Ensure conversation buffer size is maintained
        manage_conversation_buffer()
