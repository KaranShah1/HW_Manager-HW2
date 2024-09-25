import streamlit as st
from streamlit_option_menu import option_menu

# Set up a sidebar or navigation for different pages
st.set_page_config(page_title="Multi-Page App", layout="wide")

# Define navigation using a simple option menu
with st.sidebar:
    selected_page = option_menu(
        "HW Manager",
        ["First Homework", "Second Homework", "Third Homework", "Third Homework Retry", "Fourth Homework"],
        icons=['book', 'book', 'book', 'book', 'book'],
        menu_icon="cast", 
        default_index=0,
    )

# Load the appropriate page based on the user's selection
if selected_page == "First Homework":
    st.title("HW - 1 ")
    # Execute the hw1.py code
    exec(open("hw1.py").read())  # This will run the content of hw1.py

elif selected_page == "Second Homework":
    st.title("HW - 2")
    # Execute the hw2.py code
    exec(open("hw2.py").read())  # This will run the content of hw2.py

elif selected_page == "Third Homework":
    st.title("HW - 3")
    # Execute the hw3.py code
    exec(open("hw3.py").read())  # This will run the content of hw3.py
    
elif selected_page == "Third Homework Retry":
    st.title("HW - 3 Retry")
    # Execute the hw3_retry.py code
    exec(open("hw3_retry.py").read())

elif selected_page == "Fourth Homework":
    st.title("HW - 4")
    # Execute the hw4.py code
    exec(open("hw4.py").read())
