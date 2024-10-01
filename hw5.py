import streamlit as st
import openai
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set API Keys from Streamlit secrets
client = openai.OpenAI(api_key=st.secrets["openai"])

# Placeholder for your vector embeddings (for courses or clubs)
course_info = {
    "Data Science": "Data Science is a multidisciplinary field focused on extracting knowledge from data sets, which are typically large.",
    "Machine Learning": "Machine Learning is a field of AI that uses statistical techniques to give computer systems the ability to 'learn' from data.",
    "Software Engineering": "Software Engineering is the application of engineering principles to the development of software."
}

# Pre-defined vectors (example) for the course info
course_embeddings = {
    "Data Science": np.array([0.5, 0.2, 0.1]),
    "Machine Learning": np.array([0.8, 0.3, 0.5]),
    "Software Engineering": np.array([0.1, 0.7, 0.6])
}

def vector_search(query):
    """Takes in a query and returns the most relevant course info based on vector similarity."""
    # Simulate query embedding (you would use OpenAI's API to get this in practice)
    query_embedding = np.array([0.6, 0.3, 0.2])  # Just an example embedding
    
    # Compute cosine similarity between query and stored course embeddings
    similarities = {}
    for course, embedding in course_embeddings.items():
        similarities[course] = cosine_similarity([query_embedding], [embedding])[0][0]
    
    # Get the course with the highest similarity score
    most_relevant_course = max(similarities, key=similarities.get)
    
    return course_info[most_relevant_course], most_relevant_course

def get_llm_response(course_info, course_name):
    """Invoke the LLM with the course information from the vector search."""
    prompt = (f"I am a student asking about {course_name}. "
              f"The information I have is: {course_info}. "
              f"Can you summarize this information and give me additional insights?")
    
    response = client.chat_completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides information on courses and clubs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("Short-Term Memory Chatbot")

# Get user input
user_query = st.text_input("Ask about a course (e.g., 'Tell me about Machine Learning'):")

if st.button("Get Information"):
    if user_query:
        # Perform vector search to get the relevant course info
        relevant_info, relevant_course = vector_search(user_query)
        
        # Call LLM with the results of vector search
        llm_response = get_llm_response(relevant_info, relevant_course)
        
        # Display results in the app
        st.write(f"Relevant course: {relevant_course}")
        st.write(f"Course Information: {relevant_info}")
        st.write(f"LLM Response: {llm_response}")
    else:
        st.write("Please enter a query.")
