import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("LAB-01-Karan Shah📄 Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Fetch the OpenAI API key from Streamlit secrets.
openai_api_key = st.secrets["somesection"]

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Sidebar options
    st.sidebar.header("Summary Options")
    summary_option = st.sidebar.selectbox(
        "Choose a summary type:",
        ["Summarize in 100 words", "Summarize in 2 connecting paragraphs", "Summarize in 5 bullet points"]
    )

    advanced_model = st.sidebar.checkbox("Use Advanced Model (GPT-4o)")

    # Let the user upload a file via st.file_uploader.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via st.text_area.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        
        # Set the model based on the checkbox
        model = "gpt-4o" if advanced_model else "gpt-4o-mini"

        # Create messages for the API request
        if summary_option == "Summarize in 100 words":
            prompt = f"Summarize the following document in 100 words: {document}"
        elif summary_option == "Summarize in 2 connecting paragraphs":
            prompt = f"Summarize the following document in 2 connecting paragraphs: {document}"
        elif summary_option == "Summarize in 5 bullet points":
            prompt = f"Summarize the following document in 5 bullet points: {document}"
        
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
