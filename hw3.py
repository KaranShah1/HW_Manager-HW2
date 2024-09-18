import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("LAB-03-Karan Shah📄 Document question answering and Chatbot")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # Sidebar options for summarizing 
    st.sidebar.title("Options")
    
    # Model selection
    openAI_model = st.sidebar.selectbox("Choose the GPT Model", ("mini", "regular"))
    model_to_use = "gpt-4o-mini" if openAI_model == "mini" else "gpt-4o"

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
        # Process the uploaded file
        document = uploaded_file.read().decode()

        # Instruction based on user selection on the sidebar menu
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the LLM
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Generate the summary using the OpenAI API
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            stream=True,
        )

        # Stream the summary response to the app
        st.write_stream(stream)

    # Set up the session state to hold chatbot messages with a buffer limit
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    # Define the conversation buffer size (2 user messages and 2 responses)
    conversation_buffer_size = 4  # 2 user messages + 2 assistant responses

    def manage_conversation_buffer():
        """Ensure the conversation buffer size does not exceed the limit."""
        if len(st.session_state.chat_history) > conversation_buffer_size:
            # Keep only the last conversation_buffer_size messages
            st.session_state.chat_history = st.session_state.chat_history[-conversation_buffer_size:]

    # Display the chatbot conversation
    st.write("## Chatbot Interaction")
    for msg in st.session_state.chat_history:
        chat_msg = st.chat_message(msg["role"])
        chat_msg.write(msg["content"])

    # Get user input for the chatbot
    if prompt := st.chat_input("Ask the chatbot a question or interact:"):
        # Append the user input to the session state
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Display the user input in the chat
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

        # Stream the assistant's response
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

        # Append the assistant's response to the session state
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Now, implement the logic to ask, "Do you want more info?"
        if "yes" in prompt.lower():
            follow_up_response = "Great! Here's more information: ..."
        elif "no" in prompt.lower():
            follow_up_response = "Okay! Feel free to ask anything else."

        # If not yes/no, the assistant will ask, "Do you want more info?"
        else:
            follow_up_response = "Do you want more info?"

        # Append the follow-up response to the session state and display
        st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
        st.chat_message("assistant").write(follow_up_response)

        # Ensure the conversation buffer size does not exceed the limit
        manage_conversation_buffer()
