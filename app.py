import streamlit as st
import sqlite3
from datetime import datetime
# from openai import OpenAI  # Uncomment if you use OpenAI
from db_operations import *
import os
import pandas as pd
import base64
import time
import threading
import uuid

# Initialize your OpenAI client if needed
# client = OpenAI(api_key='your_api_key_here')

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Function to start a new session
def start_new_session(session_name):
    st.session_state.current_session_name = session_name
    st.session_state.current_session_id = get_timestamp()
    add_session(st.session_state.current_session_id, st.session_state.current_session_name, st.session_state.user_id)
    st.experimental_rerun()


# Login and Signup functions
def show_signup():
    st.title("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Signup"):
        if register_user(username, password):
            st.success("User registered successfully!")
        else:
            st.error("Username already exists!")

    if st.button("Go to Login", key="goto_login"):
        st.session_state.show_signup = False
        st.experimental_rerun()


def show_login():
    st.title("Login")
    username = st.text_input("Username ")
    password = st.text_input("Password ", type='password')
    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password!")

    if st.button("Go to Signup", key="goto_signup"):
        st.session_state.show_signup = True
        st.experimental_rerun()


def generate_response(prompt):
    # Simulate a long-running operation
    time.sleep(5)  # Replace this with your actual model invocation
    return f"This is the result for: {prompt}"


def main():
    st.title("Jarvis")

    # Initialize session state for user authentication
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False

    # Main application logic
    if st.session_state.user_id is None:
        if st.session_state.show_signup:
            show_signup()
        else:
            show_login()
    else:
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None

        if 'current_session_name' not in st.session_state:
            st.session_state.current_session_name = None

        # Sidebar: list of sessions
        st.sidebar.title("Past Sessions")
        st.sidebar.subheader(f"Welcome, {st.session_state.username}!")
        sessions = get_sessions(st.session_state.user_id)

        # Choosing first session as current session if no session is selected
        if not st.session_state.current_session_id and sessions:
            st.session_state.current_session_id = sessions[0][0]
            st.session_state.current_session_name = sessions[0][1]

        with st.sidebar.expander("Start new session"):
            name = st.text_input("Enter your new Session Name:")
            if st.button("Start", key="start_new_session"):
                if name:
                    start_new_session(name)
                else:
                    st.error("Please enter a session name.")

        for idx, session in enumerate(sessions):
            cols = st.sidebar.columns([4, 1])

            if cols[0].button(f'{session[1].capitalize()}', key=f'session_{idx}'):
                st.session_state.current_session_id = session[0]
                st.session_state.current_session_name = session[1]
                st.experimental_rerun()

            if cols[1].button("🗑️", key=f"delete_{session[0]}"):
                delete_session(session[0])
                st.experimental_rerun()

        # Display chat history of the selected session
        if st.session_state.current_session_id:
            st.write("Current session Name: ", st.session_state.current_session_name)
            st.session_state.messages = get_chat_history(st.session_state.current_session_id)
        else:
            session_name = st.text_input("New Session Name:")
            if st.button("Start new session", key="start_new_session_main"):
                if session_name:
                    start_new_session(session_name)
                else:
                    st.error("Please enter a session name.")

        for message in st.session_state.messages:
            with st.chat_message(message[0]):
                st.markdown(message[1])

        prompt = st.chat_input("Enter Question: ")

        if prompt:
            st.session_state.messages.append(("user", prompt))
            save_message(st.session_state.current_session_id, 'user', prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Create a progress bar
                progress_text = "Generating response. Please wait."
                my_bar = st.progress(0, text=progress_text)

                # Placeholder for the response
                response_placeholder = st.empty()

                # Start the response generation in a separate thread
                result_container = {'result': None}

                def generate_and_store_result():
                    result_container['result'] = generate_response(prompt)

                thread = threading.Thread(target=generate_and_store_result)
                thread.start()

                # Update progress bar while waiting for the response
                progress = 0
                while thread.is_alive():
                    progress = (progress + 1) % 100
                    my_bar.progress(progress / 100, text=progress_text)
                    response_placeholder.text(f"Generating response... please wait")
                    time.sleep(0.1)

                # Ensure the thread has completed
                thread.join()

                # Clear the progress bar
                my_bar.empty()

                # Display the result
                result = result_container['result']
                response_placeholder.markdown(f"<div style='text-align: center;'><strong>Results:</strong> {result}</div>", unsafe_allow_html=True)

                st.session_state.messages.append(("assistant", result))
                save_message(st.session_state.current_session_id, 'assistant', result)

        # Use a fixed key for the Logout button
        if st.sidebar.button("Logout", key='logout_button'):
            st.session_state.clear()
            st.experimental_rerun()


if __name__ == "__main__":
    # Ensure the 'coding' directory exists
    if not os.path.exists('coding'):
        os.makedirs('coding')
    # Check if chat_history.db exists
    if not os.path.exists('chat_history.db'):
        init_db()
    main()
