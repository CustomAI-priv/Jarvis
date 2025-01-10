import os
import sqlite3
from datetime import datetime
import hashlib
import logging
from uuid import uuid4
from pathlib import Path
from application_state import ApplicationStateManager

# define the global state manager
state_manager = ApplicationStateManager()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DBUtilities:

    def __init__(self, db_name='chat_history.db', max_items_per_chat=100, max_chats_per_model=10):
        if not isinstance(max_items_per_chat, int) or max_items_per_chat <= 0:
            raise ValueError("max_items_per_chat must be a positive integer.")
        if not isinstance(max_chats_per_model, int) or max_chats_per_model <= 0:
            raise ValueError("max_chats_per_model must be a positive integer.")
        
        # Get the absolute path of the current file (chat_history_management.py)
        current_file = Path(__file__).resolve()
        
        # Get the backend directory (parent directory of current file)
        backend_dir = current_file.parent
        
        # Create backend directory if it doesn't exist
        os.makedirs(backend_dir, exist_ok=True)
        
        # Construct the full database path
        db_path = backend_dir / db_name
        logging.info(f"Using database path: {db_path}")
        
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys = ON;")  # Enable foreign key support

        self.max_items_per_chat = max_items_per_chat
        self.max_chats_per_model = max_chats_per_model

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


class ChatHistoryManagement(DBUtilities):

    """
    This class is used to manage the chat history database.
    """

    def __init__(self, db_name='chat_history.db', max_items_per_chat=100, max_chats_per_model=10):
        super().__init__(db_name, max_items_per_chat, max_chats_per_model)

        # get the database creation flag
        self.db_created_flag = state_manager.get_state('chat_history_db_created_flag')

        # if the database has not been created, create the tables
        if not self.db_created_flag:
            self.create_tables()
            self.create_indexes()
            self.ensure_user_id_column()
            state_manager.set_state('chat_history_db_created_flag', 1)

    def create_tables(self):
        with self.connection:
            # Create the Users table first
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    verification_code TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create the Chat History table with user_id column
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    model_id INTEGER NOT NULL,
                    chat_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')

    def ensure_user_id_column(self):
        """Ensure that the user_id column exists in the chat_history table."""
        with self.connection:
            cursor = self.connection.execute("PRAGMA table_info(chat_history)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'user_id' not in columns:
                # Add the user_id column if it's missing
                self.connection.execute('''
                    ALTER TABLE chat_history
                    ADD COLUMN user_id INTEGER NOT NULL DEFAULT 0
                ''')
                # Optionally, you may need to backfill user_id if your application logic requires it.

    # Add this method to the UserManagement class
    def add_verification_code_column(self):
        """Adds verification_code column to users table if it doesn't exist"""
        with self.connection:
            cursor = self.connection.execute('''
                SELECT name FROM pragma_table_info('users') WHERE name='verification_code'
            ''')
            if not cursor.fetchone():
                self.connection.execute('''
                    ALTER TABLE users
                    ADD COLUMN verification_code TEXT
                ''')
                logging.info("Added verification_code column to users table")

    def get_number_of_chats(self, user_id: int, model_id: int):
        with self.connection:
            cursor = self.connection.execute('''
                SELECT COUNT(DISTINCT chat_id) FROM chat_history
                WHERE user_id = ? AND model_id = ?
            ''', (user_id, model_id))
            return cursor.fetchone()[0]

    def create_indexes(self):
        """Create indexes after ensuring tables and columns exist"""
        with self.connection:
            # Check if the chat_history table exists
            cursor = self.connection.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='chat_history'
            """)
            
            if cursor.fetchone():
                # Create indexes only if the table exists
                self.connection.execute('''
                    CREATE INDEX IF NOT EXISTS idx_chat_history_user_id 
                    ON chat_history(user_id)
                ''')

                self.connection.execute('''
                    CREATE INDEX IF NOT EXISTS idx_chat_history_model_id 
                    ON chat_history(model_id)
                ''')

    def save_message(self, user_id: int, model_id: int, chat_id: int, question: str, answer: str):
        """Saves a question and answer pair to the chat history for a specific user, model, and chat."""
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with self.connection:
            # Check if the question and answer pair already exists
            cursor = self.connection.execute('''
                SELECT 1 FROM chat_history
                WHERE user_id = ? AND model_id = ? AND chat_id = ? AND question = ? AND answer = ?
            ''', (user_id, model_id, chat_id, question, answer))

            if cursor.fetchone() is None:
                # If the pair does not exist, insert it with timestamp
                self.connection.execute('''
                    INSERT INTO chat_history (user_id, model_id, chat_id, question, answer, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, model_id, chat_id, question, answer, current_timestamp))
                logging.info(f"Question and answer saved for user '{user_id}', model '{model_id}', and chat '{chat_id}' at {current_timestamp}.")
            else:
                # Update the timestamp for existing entry
                self.connection.execute('''
                    UPDATE chat_history
                    SET timestamp = ?
                    WHERE user_id = ? AND model_id = ? AND chat_id = ? AND question = ? AND answer = ?
                ''', (current_timestamp, user_id, model_id, chat_id, question, answer))
                logging.info(f"Updated timestamp for existing entry for user '{user_id}', model '{model_id}', and chat '{chat_id}' to {current_timestamp}.")

    def get_chat_history(self, user_id, model_id, chat_id):
        """Retrieves chat history for a specific user, model, and chat, limited by max_items_per_chat."""
        with self.connection:
            cursor = self.connection.execute('''
                SELECT ch.question, ch.answer, ch.timestamp
                FROM chat_history ch
                WHERE ch.user_id = ? AND ch.model_id = ? AND ch.chat_id = ?
                ORDER BY ch.timestamp DESC
                LIMIT ?
            ''', (user_id, model_id, chat_id, self.max_items_per_chat))
            chat_history = cursor.fetchall()
            logging.info(f"Retrieved {len(chat_history)} messages for user '{user_id}', model '{model_id}', and chat '{chat_id}'.")
            return chat_history

    def get_total_items_in_chat(self, user_id: int, model_id: int, chat_id: int):
        with self.connection:
            cursor = self.connection.execute('''
                SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND model_id = ? AND chat_id = ?
            ''', (user_id, model_id, chat_id))
            return cursor.fetchone()[0]

    def get_chat_ids_sorted_by_timestamp(self, user_id: int, model_id: int = None, chat_id: int = None):
        """
        Retrieves all chat messages for a specific user and optionally filters by model_id and chat_id.
        
        Args:
            user_id (int): The user ID to fetch messages for
            model_id (int, optional): Filter messages by model ID
            chat_id (int, optional): Filter messages by chat ID
        """
        with self.connection:
            # Build the query dynamically based on provided parameters
            query = '''
                SELECT DISTINCT message_id, chat_id, model_id, question, answer, timestamp
                FROM chat_history
                WHERE user_id = ?
            '''
            params = [user_id]

            if model_id is not None:
                query += ' AND model_id = ?'
                params.append(model_id)
            
            if chat_id is not None:
                query += ' AND chat_id = ?'
                params.append(chat_id)

            query += ' ORDER BY timestamp DESC'
            
            cursor = self.connection.execute(query, params)
            messages = cursor.fetchall()

            # Create a list of dictionaries containing message details
            sorted_messages = [{
                "message_id": msg[0],
                "chat_id": msg[1],
                "model_id": msg[2],
                "question": msg[3],
                "answer": msg[4],
                "timestamp": msg[5],
                "user_id": user_id
            } for msg in messages]
            
            logging.info(f"Retrieved {len(sorted_messages)} messages for user '{user_id}'" + 
                        (f", model '{model_id}'" if model_id else "") + 
                        (f", chat '{chat_id}'" if chat_id else ""))
            return sorted_messages


class UserManagement(DBUtilities):
    def __init__(self, db_name='chat_history.db', max_items_per_chat=100, max_chats_per_model=10):
        super().__init__(db_name, max_items_per_chat, max_chats_per_model)

    def generate_password_hash(self, password):
        """Generates a SHA256 hash for the given password."""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def check_password_hash(self, stored_hash, provided_password):
        """Checks if the provided password matches the stored hash."""
        return stored_hash == self.generate_password_hash(provided_password)

    def enter_verification_code(self, email: str, password: str, verification_code: str):
        """Updates the verification code for a user after validating their credentials."""
        with self.connection:
            # Fetch and print all users in the database for debugging
            cursor = self.connection.execute('SELECT id, username, email, password_hash FROM users')
            all_users = cursor.fetchall()

            print("\nAll users in the database:")
            for user in all_users:
                print(user)
                print(f"User ID: {user[0]}, Username: {user[1]}, Email: {user[2]}")

            # First verify the user exists and password is correct using email
            cursor = self.connection.execute('''
                SELECT id, password_hash FROM users WHERE email = ?
            ''', (email,))
            user = cursor.fetchone()

            if not user:
                logging.warning(f"No user found with email '{email}'")
                return False

            if not self.check_password_hash(user[1], password):
                print('password hash issue')
                logging.warning(f"Invalid password for email '{email}'")
                return False

            # Update the verification code using the found user ID
            print('this is the user id:', user[0])
            self.connection.execute('''
                UPDATE users
                SET verification_code = ?
                WHERE id = ?
            ''', (verification_code, user[0]))

            logging.info(f"Verification code updated for user ID '{user[0]}' with email '{email}'")
            return True

    def verify_verification_code(self, email: str, verification_code: str):
        """Verifies the verification code for a user."""
        with self.connection:
            cursor = self.connection.execute('''
                SELECT 1 FROM users WHERE email = ? AND verification_code = ?
            ''', (email, verification_code))
            return cursor.fetchone() is not None

    def register_user(self, username, email, password):
        """Registers a new user with a username, email, and password."""
        password_hash = self.generate_password_hash(password)
        try:
            with self.connection:
                self.connection.execute('''
                    INSERT INTO users (username, email, password_hash, verification_code)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, password_hash, 'placeholder'))  # Set verification_code to None initially
            logging.info(f"User '{username}' registered successfully.")
            return True
        except sqlite3.IntegrityError as e:
            logging.error(f"Failed to register user '{username}': {e}")
            return False  # Username or email already exists

    def login_user(self, username, password):
        """Logs in a user by verifying their username and password."""
        with self.connection:
            cursor = self.connection.execute('''
                SELECT id, password_hash FROM users WHERE username = ?
            ''', (username,))
            user = cursor.fetchone()
            print(user)
            if user and self.check_password_hash(user[1], password):
                print('this is the user id:', user[0])
                print('the login worked!')
                logging.info(f"User '{username}' logged in successfully.")
                return user[0]  # Return user ID
            logging.warning(f"Login failed for user '{username}'.")
            return None

    def get_user(self, user_id):
        """Retrieves user information by user ID."""
        with self.connection:
            cursor = self.connection.execute('''
                SELECT id, username, email, timestamp FROM users WHERE id = ?
            ''', (user_id,))
            return cursor.fetchone()


class CleanupOperations(DBUtilities):
    def __init__(self, db_name='chat_history.db', max_items_per_chat=100, max_chats_per_model=10):
        super().__init__(db_name, max_items_per_chat, max_chats_per_model)

    def _clean_chats(self):
        """Cleans up the chat_history table to remove chat_ids that have no messages."""
        with self.connection:
            # Find chat_ids with no messages
            cursor = self.connection.execute('''
                SELECT chat_id FROM chat_history
                GROUP BY chat_id
                HAVING COUNT(*) = 0
            ''')
            empty_chats = [row[0] for row in cursor.fetchall()]

            if empty_chats:
                # Delete chat_ids with no messages
                placeholders = ','.join(['?'] * len(empty_chats))
                self.connection.execute(f'''
                    DELETE FROM chat_history
                    WHERE chat_id IN ({placeholders})
                ''', empty_chats)
                logging.info(f"Deleted chat_ids with no messages: {empty_chats}")
            else:
                logging.info("No empty chat_ids found to delete.")

    def clean_old_chats_by_model(self):
        """
        For each chat_id, keep only the 100 most recent chat entries and delete the older ones.
        """
        models = {1: 'text', 2: 'analytical'}
        with self.connection:
            for model_id, model_name in models.items():
                # Select chat_ids to process
                cursor = self.connection.execute('''
                    SELECT DISTINCT chat_id FROM chat_history
                    WHERE model_id = ?
                ''', (model_id,))
                chat_ids = [row[0] for row in cursor.fetchall()]

                for chat_id in chat_ids:
                    # Select message_ids to keep (the 100 most recent per chat_id)
                    cursor = self.connection.execute('''
                        SELECT message_id FROM chat_history
                        WHERE chat_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                    ''', (chat_id,))
                    messages_to_keep = [row[0] for row in cursor.fetchall()]

                    if messages_to_keep:
                        # Delete messages of this chat_id that are not in the messages_to_keep list
                        placeholders = ','.join(['?'] * len(messages_to_keep))
                        query = f'''
                            DELETE FROM chat_history
                            WHERE chat_id = ?
                            AND message_id NOT IN ({placeholders})
                        '''
                        parameters = [chat_id] + messages_to_keep
                        self.connection.execute(query, parameters)
                        logging.info(f"Deleted old messages for chat_id '{chat_id}' in model '{model_name}'.")
                    else:
                        logging.info(f"No messages to delete for chat_id '{chat_id}' in model '{model_name}'.")

    def chat_cleanup_operation(self):
        """
        Runs the chat cleanup operation to maintain only the most recent sessions for each model type.
        """
        try:
            self.clean_old_chats_by_model()
            self._clean_chats()
            logging.info("Chat cleanup operation completed successfully.")
        except Exception as e:
            logging.error(f"An error occurred during chat cleanup: {e}")


username = 'testuser2'
email = 'bigbrideai@gmail.com'
password = 'securepassword'
model_type = 1 # text model
chat_id = 4

# register a user
chat_history_management = ChatHistoryManagement()
user_manager = UserManagement()

def test(): 
    # register a user
    print('register user:', user_manager.register_user(username, email, password))

    # send the verification code
    print('enter verification code:', user_manager.enter_verification_code(email, password, '123456'))

    # verify the verification code
    print('verify verification code:', user_manager.verify_verification_code(email, '123456'))

    # get the user id
    user_id = user_manager.login_user(username, password)
    print('login user:', user_id)

    # Insert two messages into the chat history
    chat_history_management.save_message(user_id, model_type, chat_id, "Hello, how are you?", "I'm good, thank you!")
    chat_history_management.save_message(user_id, model_type, chat_id, "What can you do?", "I can assist you with various tasks!")

    # Retrieve the chat history for the user
    chat_history = chat_history_management.get_chat_ids_sorted_by_timestamp(user_id, model_type, chat_id)
    print("Chat History:")
    for message in chat_history:
        print(f"Chat ID: {message['chat_id']}")
        print(f"Question: {message['question']}")
        print(f"Answer: {message['answer']}")
        print(f"Timestamp: {message['timestamp']}\n")

