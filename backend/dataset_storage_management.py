import pandas as pd
import psycopg2
from psycopg2.extensions import Binary
import io
import argparse
import gzip
import subprocess
import time
import concurrent.futures

try: 
    from backend.application_state import ApplicationStateManager
except: 
    from application_state import ApplicationStateManager


# define the global state manager
state_manager = ApplicationStateManager()


class CSVDatabaseHandler():
    """
    This class is used to upload and retrieve CSV files to a PostgreSQL database.
    """

    def __init__(self, database="analytical_agent_db", user="postgres", password="CustomAI1234", host="localhost", port=5432):
        """Initialize database connection with default local settings"""

        # get the database creation flag
        self.db_created_flag = state_manager.get_state('db_created_flag')

        # define the database parameters
        self.db_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }

        # Try to connect to the database first
        if not self.db_created_flag:
            self.setup_database()
            # set the database creation flag to True
            state_manager.set_state('db_created_flag', 1)

    def setup_database(self):
        """Set up PostgreSQL database on Colab environment"""

        # define the commands to setup the database
        commands = [
            "apt-get -y install postgresql postgresql-contrib",
            "service postgresql start",
            "sudo -u postgres psql -c \"ALTER USER postgres PASSWORD 'CustomAI1234';\"",
            "sudo -u postgres psql -c \"CREATE DATABASE analytical_agent_db;\""
        ]

        # execute the commands
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"Successfully executed: {cmd}")
                time.sleep(1)  # Small delay between commands
            except subprocess.CalledProcessError as e:
                if "already exists" in str(e):
                    print(f"Database/user already exists, continuing...")
                else:
                    print(f"Error executing {cmd}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error during setup: {str(e)}")

        print("Database setup completed")

        # Test connection
        try:
            # get the connection
            conn = self._get_connection()
            # close the connection
            conn.close()
        except Exception as e:
            # print the error message
            print(f"Connection test failed: {str(e)}")

    def _get_connection(self):
        """Create and return database connection"""

        # get the connection and return it
        return psycopg2.connect(**self.db_params)

    def upload_csv(self, csv_path, table_name, name):
        """
        Compress and upload CSV file to database
        """
        # Read CSV and compress directly using gzip
        with open(csv_path, 'rb') as f:
            csv_data = pd.read_csv(f).to_csv(index=False).encode()
        compressed_data = Binary(gzip.compress(csv_data))

        # Store in database
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Create table if it doesn't exist
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255),
                    compressed_data BYTEA,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')

            # Insert compressed data
            cursor.execute(
                f"INSERT INTO {table_name} (name, compressed_data) VALUES (%s, %s)",
                (name, compressed_data)
            )

            conn.commit()

        # handle the exception
        except Exception as e:
            conn.rollback()
            raise e

        finally:
            # close the cursor
            cursor.close()
            # close the connection
            conn.close()

    def retrieve_csv(self, table_name, name, output_path=None):
        """
        Retrieve and decompress CSV data from database
        """

        # get the connection
        conn = self._get_connection()
        # get the cursor
        cursor = conn.cursor()

        try:
            # get the latest compressed data
            cursor.execute(f"""
                SELECT compressed_data
                FROM {table_name}
                WHERE name = %s
            """, (name,))
            result = cursor.fetchone()

            # check if the result is not None
            if not result:
                raise ValueError(f"No data found in table '{table_name}'. Here is name we are searching for: {name}")

            # get the compressed data
            compressed_data = result[0]

        finally:
            # close the cursor
            cursor.close()
            # close the connection
            conn.close()

        # decompress the data
        df = pd.read_csv(io.BytesIO(compressed_data), compression='gzip')

        # save to file if output path provided
        if output_path:
            df.to_csv(output_path, index=False)

        # return the dataframe
        return df

    def retrieve_multiple_csv(self, table_name, file_names, output_paths=None):
        """
        Retrieve and decompress multiple CSV files concurrently from database

        Args:
            table_name (str): Name of the database table
            file_names (list): List of file names to retrieve
            output_paths (list, optional): List of output paths corresponding to file_names

        Returns:
            dict: Dictionary mapping file names to their corresponding DataFrames
        """

        # check if the output paths match the file names
        if output_paths and len(output_paths) != len(file_names):
            raise ValueError("If output_paths provided, must match length of file_names")

        # if the output paths are not provided, set them to None
        if output_paths is None:
            output_paths = [None] * len(file_names)

        # create the args list for concurrent execution
        args = [(table_name, name, path) for name, path in zip(file_names, output_paths)]

        # use the ThreadPoolExecutor for concurrent IO operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # map the retrieve_csv to all files
            results = list(executor.map(
                lambda x: self.retrieve_csv(*x),
                args
            ))

        # create the dictionary mapping file names to DataFrames
        return {name: df for name, df in zip(file_names, results)}

    def list_tables(self):
        """
        List all tables in the database and their contents summary

        Returns:
            dict: Dictionary containing table names and their row counts
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Query to get all tables and their row counts
            cursor.execute("""
                SELECT
                    tablename,
                    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name=tables.tablename) as column_count,
                    (SELECT COUNT(*) FROM ONLY pg_catalog.pg_tables WHERE tablename=tables.tablename) as row_count
                FROM pg_catalog.pg_tables tables
                WHERE schemaname != 'pg_catalog'
                AND schemaname != 'information_schema';
            """)

            tables_info = {}
            for table, column_count, row_count in cursor.fetchall():
                # For each table, get the list of stored file names
                cursor.execute(f"""
                    SELECT name, upload_timestamp
                    FROM {table}
                    ORDER BY upload_timestamp DESC;
                """)
                files = cursor.fetchall()

                tables_info[table] = {
                    'column_count': column_count,
                    'row_count': row_count,
                    'stored_files': [{'name': name, 'timestamp': ts} for name, ts in files]
                }

            return tables_info

        finally:
            cursor.close()
            conn.close()

    def print_tables_summary(self):
        """
        Print a formatted summary of all tables in the database
        """
        tables_info = self.list_tables()

        print("\n=== Database Tables Summary ===")
        for table_name, info in tables_info.items():
            print(f"\nTable: {table_name}")
            print(f"Columns: {info['column_count']}")
            print(f"Total Rows: {info['row_count']}")
            print("Stored Files:")
            for file in info['stored_files']:
                print(f"  - {file['name']} (uploaded: {file['timestamp']})")
        print("\n===========================")

def test():
    # Example usage
    handler = CSVDatabaseHandler(
        database="analytical_agent_db",  # Create this database first in PostgreSQL
        password="CustomAI1234",       # The password you set during PostgreSQL installation
    )

    # Upload example
    handler.upload_csv("/content/manual_upload/customer_by_employee_data.csv", "stone_insurance", "customer_by_employee_data.csv")

    # Retrieve example
    df = handler.retrieve_csv("stone_insurance", "customer_by_employee_data.csv", "customer_by_employee_data.csv")

    # print the dataframe
    print(df)

def test_tables(): 
    handler = CSVDatabaseHandler(
        database="analytical_agent_db",  # Create this database first in PostgreSQL
        password="CustomAI1234",       # The password you set during PostgreSQL installation
    )
    handler.print_tables_summary()
test_tables()