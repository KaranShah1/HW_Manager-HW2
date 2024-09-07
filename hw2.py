import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF for PDF
import docx
import textwrap

# Show title and description.
st.title("Karan Shah📄 Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# Fetch the OpenAI API key from Streamlit secrets.
openai_api_key = st.secrets.get("somesection", None)

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

    # File uploader with support for .txt, .md, .pdf, and .docx
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, .pdf, .docx)", type=("txt", "md", "pdf", "docx")
    )

    # Function to extract text from PDF
    def extract_text_from_pdf(file):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf.page_count):
            text += pdf[page_num].get_text()
        return text

    # Function to extract text from DOCX
    def extract_text_from_docx(file):
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    # Process the uploaded file and extract the text
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            document = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            document = extract_text_from_docx(uploaded_file)
        else:
            document = uploaded_file.read().decode()

        # Display document preview (first 500 characters)
        st.subheader("Document Preview:")
        st.text(document[:500] + "...")

        # Show word count
        word_count = len(document.split())
        st.write(f"Word count: {word_count}")

        # Ask the user for a question
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        # Function to chunk long documents
        def chunk_text(text, chunk_size=2000):
            return textwrap.wrap(text, chunk_size)

        # Chunk the document if needed
        chunks = chunk_text(document)

        # Display the number of chunks
        if len(chunks) > 1:
            st.write(f"Document split into {len(chunks)} chunks.")

        # Set the model based on the checkbox
        model = "gpt-4o" if advanced_model else "gpt-4o-mini"

        # Create prompt based on the summary option
        if summary_option == "Summarize in 100 words":
            prompt = f"Summarize the following document in 100 words: {document}"
        elif summary_option == "Summarize in 2 connecting paragraphs":
            prompt = f"Summarize the following document in 2 connecting paragraphs: {document}"
        elif summary_option == "Summarize in 5 bullet points":
            prompt = f"Summarize the following document in 5 bullet points: {document}"

        # Define message structure for OpenAI API request
        messages = [
            {
                "role": "user",
                "content": f"{prompt} \n\n---\n\n {question}",
            }
        ]

        @st.cache_data(show_spinner=False)
        def get_openai_response(model, messages):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

        if uploaded_file and question:
            try:
                # Add progress bar
                progress_bar = st.progress(0)

                # Loop through document chunks with progress update
                for i, chunk in enumerate(chunks):
                    progress_bar.progress((i + 1) / len(chunks))
                    st.write(f"Processing chunk {i + 1}/{len(chunks)}...")
                    
                    # Add chunk to the messages
                    messages = [
                        {
                            "role": "user",
                            "content": f"{prompt} \n\n---\n\n {chunk}",
                        }
                    ]

                    # Get response from OpenAI API (with caching)
                    stream = get_openai_response(model, messages)

                    # Display the response using streamlit's streaming feature
                    st.write_stream(stream)

                # Combine the response into text format for download
                response_text = "".join([resp["choices"][0]["message"]["content"] for resp in stream])

                # Download button for the response
                st.download_button(
                    label="Download Response",
                    data=response_text,
                    file_name="response.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
