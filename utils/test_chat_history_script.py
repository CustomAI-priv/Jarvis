# test_script.py

import logging
from chat_history_management import ChatHistoryManagement, UserManagement, CleanupOperations

def main():
    # Initialize the database and classes
    db_name = 'chat_history.db'  # Ensure all classes use the same database
    max_items_per_chat = 100
    max_chats_per_model = 10

    # Instantiate management classes
    chat_history_management = ChatHistoryManagement(
        db_name=db_name,
        max_items_per_chat=max_items_per_chat,
        max_chats_per_model=max_chats_per_model
    )
    user_management = UserManagement(
        db_name=db_name,
        max_items_per_chat=max_items_per_chat,
        max_chats_per_model=max_chats_per_model
    )
    cleanup_operations = CleanupOperations(
        db_name=db_name,
        max_items_per_chat=max_items_per_chat,
        max_chats_per_model=max_chats_per_model
    )

    # -------------------- User Registration and Login --------------------

    # Register a new user
    username = "john_doe"
    email = "john@example.com"
    password = "securepassword123"

    success = user_management.register_user(username=username, email=email, password=password)
    if not success:
        print("User registration failed.")
        return

    # Log in the user
    user_id = user_management.login_user(username=username, password=password)
    if user_id is None:
        print("User login failed.")
        return

    print(f"User '{username}' logged in with user ID: {user_id}")

    # -------------------- Adding a Session --------------------

    # Since there's no method to add a session, we'll insert it directly
    session_id = None
    model_type = 'text'

    with chat_history_management.connection:
        cursor = chat_history_management.connection.execute('''
            INSERT INTO sessions (user_id, model_type)
            VALUES (?, ?)
        ''', (user_id, model_type))
        session_id = cursor.lastrowid
        logging.info(f"Session '{session_id}' created for user ID {user_id}.")

    # -------------------- Chat History Management --------------------

    # Save messages to the session
    chat_history_management.save_message(
        session_id=session_id,
        sender='user',
        content='Hello, this is a test message from the user.'
    )
    chat_history_management.save_message(
        session_id=session_id,
        sender='assistant',
        content='Hello, this is a test response from the assistant.'
    )

    # Retrieve and display chat history
    history = chat_history_management.get_chat_history(
        session_id=session_id
    )
    print(f"\nChat history for session '{session_id}':")
    for sender, content, msg_timestamp in history:
        print(f"{msg_timestamp} - {sender}: {content}")

    # -------------------- Cleanup Operations --------------------

    # Run cleanup operations
    cleanup_operations.chat_cleanup_operation()
    print("\nCleanup operations completed.")

    # -------------------- Close Database Connections --------------------

    # Close all database connections
    chat_history_management.close()
    user_management.close()
    cleanup_operations.close()
    print("\nAll database connections closed.")

if __name__ == "__main__":
    main()
