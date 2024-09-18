import streamlit as st
import google.generativeai as genai

# Show title and description.
st.title("Test-Karan Shah📄 Document question answering and Chatbot - Gemini AI")
st.write(
    "Upload a document below and ask a question about it – Gemini AI will answer! "
    "You can also interact with the chatbot. "
    "To use this app, you need to provide your Gemini API key."
)
def google_dem(question_to_ask, api_key):
    
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-pro')

    messages = []

    gem_message = "\nPlease answer the following question: \n" + str(question_to_ask)

    messages.append({'role': 'user', 'parts': gem_message})
    response = model.generate_content(messages)

    return response.text

# Fetch the Gemini API key from Streamlit secrets
gemini_api_key = st.secrets["gemini"]

if not gemini_api_key:
    st.info("Please add your Gemini API key to continue.", icon="🗝")
else:
    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

    # Sidebar options for summarizing
    st.sidebar.title("Options")

    # Model selection (only Gemini in this case)
    model_option = st.sidebar.selectbox("Choose the Gemini Model", ("Gemini", "blank"))
    

    # Summary options
    summary_options = st.sidebar.radio(
        "Select a format for summarizing the document:",
        (
            "Summarize the document in 100 words",
            "Summarize the document in 2 connecting paragraphs",
            "Summarize the document in 5 bullet points"
        ),
    )

    # **New URL Input Section**
    st.sidebar.write("### Compare Content from Two URLs")
    url1 = st.sidebar.text_input("Enter the first URL")
    url2 = st.sidebar.text_input("Enter the second URL")

    if url1 and url2:
        st.sidebar.write(f"Comparing the content from the following URLs:\n- {url1}\n- {url2}")
        # You can process these URLs further to fetch the content using an API or web scraping.

    # **Conversation Memory Options**
    memory_type = st.sidebar.radio(
        "Choose the conversation memory type:",
        (
            "Buffer of 5 questions",
            "Conversation summary",
            "Buffer of 5,000 tokens"
        ),
    )

    # Function to generate a summary of the conversation (Placeholder for Gemini API)
    def generate_summary_gemini(messages):
        summary_request = {
            "model": model_to_use,
            "prompt": "Summarize this conversation: " + str(messages)
        }
        # Call Gemini API for summary
        summary_response = google_dem(summary_request, gemini_api_key)
        return summary_response

    # Function to manage token buffer (for 5,000 tokens)
    def manage_token_buffer(messages, token_limit=5000):
        token_count = sum(len(msg["content"].split()) * 4 for msg in messages)
        while token_count > token_limit:
            messages.pop(0)
            token_count = sum(len(msg["content"].split()) * 4 for msg in messages)
        return messages

    # Memory management based on user selection
    def manage_memory():
        if memory_type == "Buffer of 5 questions":
            if len(st.session_state.chat_history) > 5:
                st.session_state.chat_history = st.session_state.chat_history[-5:]
        elif memory_type == "Conversation summary":
            summary = generate_summary_gemini(st.session_state.chat_history)
            st.session_state.chat_history = [{"role": "assistant", "content": summary}]
        elif memory_type == "Buffer of 5,000 tokens":
            st.session_state.chat_history = manage_token_buffer(st.session_state.chat_history)

    # Display the document summary (if a document is uploaded)
    if uploaded_file:
        # Process the uploaded file
        document = uploaded_file.read().decode()

        # Instruction based on user selection on the sidebar menu
        instruction = f"Summarize the document in {summary_options.lower()}."

        # Prepare the messages for the Gemini API
        messages = {
            "model": model_to_use,
            "prompt": f"Here's a document: {document} \n\n---\n\n {instruction}",
        }

        # Generate the summary using the Gemini API
        response_text = google_dem(messages, gemini_api_key)

        # Display the summary response to the app
        st.write(response_text)

    # Set up the session state to hold chatbot messages with a buffer limit
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

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

        # Ensure the memory type is maintained
        manage_memory()

        # Generate a response from Gemini API
        messages = {
            "model": model_option,
            "prompt": prompt,
        }
        response_text = google_dem(messages, gemini_api_key)

        # Stream the assistant's response
        with st.chat_message("assistant"):
            st.write(response_text)

        # Append the assistant's response to the session state
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Manage memory again after response
        manage_memory()

        # Now, implement the logic to ask, "Do you want more info?"
        if "yes" in prompt.lower():
            follow_up_response = "Great! Here's more information: ..."
        elif "no" in prompt.lower():
            follow_up_response = "Okay! Feel free to ask anything else."
        else:
            follow_up_response = "Do you want more info?"

        # Append the follow-up response to the session state and display
        st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
        st.chat_message("assistant").write(follow_up_response)

        # Manage memory one final time
        manage_memory()
