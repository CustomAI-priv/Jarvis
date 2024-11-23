import streamlit as st
import requests
import json
import time
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="AI Chat Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a clean look
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .main {
        padding: 2rem;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    .stream-container {
        margin: 1rem auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #F8F9FA;
        max-width: 800px;
    }
    .html-container {
        margin: 2rem auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        max-width: 800px;
    }
    .summary-container {
        margin: 2rem auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #F8F9FA;
        max-width: 800px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'current_html_files' not in st.session_state:
    st.session_state['current_html_files'] = []
if 'current_summary' not in st.session_state:
    st.session_state['current_summary'] = ""
if 'stream_content' not in st.session_state:
    st.session_state['stream_content'] = ""
if 'should_clear' not in st.session_state:
    st.session_state['should_clear'] = False


# Title
st.title("💬 AI Chat Assistant")
st.markdown("---")


def decode_html_files(html_files_bytes):
    """Convert bytes strings back to HTML content"""
    return [bytes_str.decode('utf-8') if isinstance(bytes_str, bytes) else bytes_str 
            for bytes_str in html_files_bytes]


def check_text_model_result():
    """Check text_model endpoint for results"""
    try:
        response = requests.get('http://5.78.113.143:8005/text_model/status')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'complete':
                summary, html_files_bytes = data.get('result', (None, []))
                if summary and html_files_bytes:
                    st.session_state['current_html_files'] = decode_html_files(html_files_bytes)
                    st.session_state['current_summary'] = summary
                    st.session_state['stream_content'] = ""
                    st.rerun()
    except Exception as e:
        st.error(f"Error checking status: {e}")


def send_message(message):
    if True: 
        # Query parameters
        url = 'http://5.78.113.143:8005/analytical_model/'
        params = {
            'user_id': '1',
            'chat_id': '2',
            'question': message
        }
        
        # Send the POST request with query parameters
        response = requests.post(url, params=params)
        
        if response.status_code == 200:
            result = response.json()  # Get the JSON response
            # Process the result as needed
            st.session_state['result'] = result  # Store the result in session state
            st.success(f"Response: {result}")  # Display the response
        else:
            st.error(f"Error: {response.status_code} - {response.text}")


def display_content():
    """Display HTML files and summary if available, otherwise show stream content"""
    if st.session_state['current_html_files']:
        # Display HTML files
        for html_content in st.session_state['current_html_files']:
            st.markdown(f"""
                <div class="html-container">
                    <div class="html-wrapper">
                        {html_content}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Display summary
        if st.session_state['current_summary']:
            st.markdown(f"""
                <div class="summary-container">
                    <h3>Summary</h3>
                    <p>{st.session_state['current_summary']}</p>
                </div>
            """, unsafe_allow_html=True)
    elif st.session_state['stream_content']:
        st.markdown(
            f'<div class="stream-container">{st.session_state["stream_content"]}</div>',
            unsafe_allow_html=True
        )

# Chat interface
chat_container = st.container()

with chat_container:
    # Display current content
    display_content()
    
    # Chat input
    user_input = st.text_input(
        "Your message:",
        key="user_input",
        placeholder="Type your message here..."
    )

    print(user_input)

    # Handle input
    if user_input and len(user_input.strip()) > 0:
        if st.session_state.get('last_input') != user_input:
            print("sending message")
            st.session_state['last_input'] = user_input
            st.session_state['current_html_files'] = []
            st.session_state['current_summary'] = ""
            st.session_state['stream_content'] = ""
            send_message(user_input)
            print('finished sending message')