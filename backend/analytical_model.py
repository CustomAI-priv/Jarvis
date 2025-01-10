# Install necessary packages
#!pip install -qqq anthropic==0.34.1 aquarel mplcyberpunk matplotx[all] matplotlib numpy pyautogen["anthropic"] kaleido

# Imports and environment setup
import os
from datetime import datetime
from typing import Callable, Dict, Literal, Optional, Union, List
from typing_extensions import Annotated
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
import aide
import shutil
import tiktoken
from langchain_core.messages import HumanMessage
import re
import tempfile
from pprint import pprint
import functools
import ujson as json
import base64

# Import autogen and related classes
from autogen import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    config_list_from_json,
    register_function,
)
from autogen.agentchat.contrib import agent_builder
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
import autogen


# import langgraph tools
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import operator


# Other imports
from IPython.display import Image, display

import os
import sys
import json
import ast
import time
import copy
from typing import Annotated, List, Dict, Any, Tuple, Optional, Annotated, Sequence
from typing_extensions import TypedDict

import pandas as pd
import numpy as np
import lancedb
from pydantic import BaseModel, Field

from langchain.vectorstores import LanceDB
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
import sys

# Add the project root to the Python path
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    pass

# Import required modules
try:
    from backend.setup import DataLoader
    from backend.api_keys import api_key, claude_api_key, azure_api_key, azure_endpoint, voyage_api_key
    from backend.text_model import ModelSpecifications, LanceTableRetriever, LanceRetriever
    from backend.chat_history_management import ChatHistoryManagement
except ImportError:
    try:
        from setup import DataLoader
        from api_keys import api_key, claude_api_key, azure_api_key, azure_endpoint, voyage_api_key
        from text_model import ModelSpecifications, LanceTableRetriever, LanceRetriever
        from chat_history_management import ChatHistoryManagement
    except ImportError as e:
        print(f"Error importing modules: {e}")
        raise

import autogen
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from agent_prompts import system_msg_engineer, system_msg_planner, system_msg_critic, system_msg_summary, system_msg_question_enhancer, system_msg_debugger
except:
    from backend.agent_prompts import system_msg_engineer, system_msg_planner, system_msg_critic, system_msg_summary, system_msg_question_enhancer, system_msg_debugger
    pass

import io
from datetime import datetime
from pathlib import Path
from typing import Iterable
from PIL import Image
import imgkit
import aide
from abc import ABC, abstractmethod

from openai import AzureOpenAI
from openai.types import FileObject
from openai.types.beta.threads import Message, TextContentBlock, ImageFileContentBlock

from langchain.callbacks.base import BaseCallbackHandler
import requests
from functools import lru_cache


@lru_cache(maxsize=1)
def get_data_loader():
    """Cached data loader initialization"""
    data_loader = DataLoader()
    return data_loader


class MyStreamHandler(BaseCallbackHandler):
    def __init__(self, url: str = 'http://5.78.113.143:8005/update_stream', start_token=""):
        self.text = start_token
        self.url = url

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text = token
        try:
            response = requests.post(self.url, json={"output": self.text})
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data to {self.url}: {e}")

    def update_url(self, new_url: str) -> None:
        """Update the stream handler's URL"""
        self.url = new_url


class CSVDescription(BaseModel):
    summary: str = Field(description="A brief summary of the CSV file's contents")
    column_descriptions: Dict[str, str] = Field(description="Descriptions of each column's purpose and data type")
    potential_uses: List[str] = Field(description="Potential use cases or analyses that could be performed with this data")
    data_quality_issues: List[str] = Field(description="Any potential data quality issues or anomalies detected")


class CodeOutput(BaseModel):
    """Schema for code solutions."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class AgentGeneralSettings(BaseModel):
    """define the general settings for the agent"""

    # define the available plots
    plotly_charts: dict = {
        "Scatter Plot": "Use to visualize relationships between two continuous variables, like correlations or trends.",
        "Line Plot": "Ideal for showing trends over time or continuous data, such as stock prices or time series data.",
        "Bar Chart": "Use to compare categorical data or show distribution for discrete variables, such as sales by region.",
        "Histogram": "Best for displaying the distribution of a single continuous variable, such as frequency distributions.",
        "Box Plot": "Use to show data spread and detect outliers, especially when comparing groups or categories.",
        "Violin Plot": "Good for visualizing data distributions with added density information, particularly for comparing categories.",
        "Pie Chart": "Useful for visualizing proportions or parts of a whole, such as market share breakdowns.",
        "Sunburst Chart": "Best for hierarchical data visualization where categories are nested, like organizational structures.",
        "Treemap": "Great for visualizing hierarchical data with relative sizes, such as budget breakdowns or part-to-whole relationships.",
        "Funnel Chart": "Use to visualize stages in a process, such as sales pipelines or drop-off rates in conversions.",
        "Density Heatmap": "Helpful for visualizing data density in two dimensions, identifying patterns or clusters.",
        "Contour Plot": "Good for visualizing continuous data with contour lines, such as showing regions of high concentration.",
        "Bubble Chart": "Useful for comparing three variables at once, where the third variable is represented by bubble size.",
        "Polar Chart": "Best for displaying data in a radial coordinate system, such as directional or cyclical data.",
        "Radar Chart": "Ideal for comparing multiple variables for several entities, such as benchmarking performance metrics.",
        "Area Chart": "Similar to a line plot, use to show cumulative data over time, like total sales or stock volume.",
        "Bubble Map": "Great for geographical scatter plots where bubble size represents data, such as population by region.",
        "Choropleth Map": "Use to visualize values over geographic regions, such as GDP or population density across countries.",
        "3D Scatter Plot": "Useful for plotting relationships between three continuous variables in a 3D space.",
        "3D Surface Plot": "Best for visualizing complex three-dimensional surfaces, such as terrain or mathematical functions.",
        "Waterfall Chart": "Ideal for showing the cumulative effect of sequential values, such as profit and loss breakdowns."
    }


class AgentLLMConfig(BaseModel):
    """define the language models to use for the agents"""

    # define the private LLM variables
    model_id: str = 'gpt-4o'
    model_deployment_name: str = 'jarvis_llm'
    private_llm_mode: bool = False

    # define the variables for primary engineer LLM
    llm_primary_engineer_temperature: float = 0
    llm_primary_engineer_model: str = 'gpt-4o'
    llm_primary_engineer_max_tokens: int = 1000

    # define the variables for the planner LLM
    llm_planner_temperature: float = 0
    llm_planner_model: str = 'gpt-4o'
    llm_planner_max_tokens: int = 1000

    # define llm for the follow up task
    llm_follow_up_task_model: str = 'gpt-4o-mini'
    llm_follow_up_task_temperature: float = 0
    llm_follow_up_task_max_tokens: int = 50

    # define llm for the summary task
    llm_summary_model: str = 'gpt-4o-mini'
    llm_summary_temperature: float = 0
    llm_summary_max_tokens: int = 200

    # path to save the charts
    output_save_path: str = 'charts/'
    file_save_path: str = 'coding/'
    top_k_csv_files: int = 3
    csv_file_manager: int = 1 # 1 is for DB manager and 2 is for local file manager (non-DB)


class AgentGeneralTools():
    """define the general tools for the agent"""

    def __init__(self):
        """initialize the general tools"""

        # define the chat history management
        self.chat_history_management = ChatHistoryManagement()

        # define the data loader
        self.data_loader = DataLoader()

    # count the tokens from a string that enter the language model
    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def count_tokens(self, planner_template):
        token_count = self.num_tokens_from_string(str(planner_template), "cl100k_base")  # Use appropriate encoding
        print(f"Number of tokens: {token_count}")

    # clean up the temporary directory
    @staticmethod
    def temp_file_cleanup(temp_dir: str):
        """clean up the temporary directory"""
        temp_dir.cleanup()

    # handle chat history with the user database
    def get_chat_history_from_management(self, user_id: int, model_id: int, chat_id: int):
        """Get and format the chat history from the chat history management."""
        chat_hist_result = self.chat_history_management.get_chat_history(user_id=user_id, model_id=model_id, chat_id=chat_id)

        # Check if there are any items in the chat history
        if not chat_hist_result:
            return '', ''

        # Get the last item in the chat history
        question, answer, _ = chat_hist_result[0]

        # Return the last question and answer as a tuple
        return question, answer

    def add_chat_history_to_management(self, user_id: int, model_id: int, chat_id: int, question_answer_pair: tuple) -> None:
        """Add the chat history to the chat history management."""
        print('question: ', question_answer_pair[0])
        print('answer: ', question_answer_pair[1])
        self.chat_history_management.save_message(user_id=user_id, model_id=model_id, chat_id=chat_id, question=question_answer_pair[0], answer=question_answer_pair[1])

    @staticmethod
    def _get_dataframe_info(df: pd.DataFrame):
        """
        Get the DataFrame info as a string.

        This method captures the output of df.info() in a string buffer
        instead of printing it to the console.

        Args:
            df (pd.DataFrame): The DataFrame to get info from.

        Returns:
            str: A string containing the DataFrame info.
        """
        # Create a string buffer to capture the output
        buffer = StringIO()

        # Write the DataFrame info to the buffer instead of stdout
        df.info(buf=buffer)

        # Get the string value from the buffer and return it
        return buffer.getvalue()

    def move_html_files_from_charts_to_reserves(self):
        """Move html files from charts to reserves directory, overwriting existing files."""

        # Get the list of all html files in the charts directory
        html_files = [f for f in os.listdir(self.data_loader.download_path_model.charts_path) if f.endswith('.html')]

        moved_files = []
        for html_file in html_files:
            source = Path(self.data_loader.download_path_model.charts_path) / html_file
            dest = Path(self.data_loader.download_path_model.reserves_path) / html_file

            try:
                # Remove destination file if it exists
                if dest.exists():
                    dest.unlink()

                shutil.move(str(source), str(dest))
                moved_files.append(html_file)

            except Exception as e:
                print(f"Error moving file '{html_file}': {str(e)}")

        return moved_files

    def cleanup_files(self, file_names: list):
        """
        Delete specified files from the coding directory.

        This method iterates through a list of file names or paths, attempts to delete
        each file from the 'coding' directory, and provides feedback on the operation.

        Args:
            file_names (list): List of file names or file paths to be deleted.
        """
        for file_path in file_names:
            # Extract the base name of the file (removes any directory path)
            file_name = os.path.basename(file_path)

            # Construct the full path to the file in the 'coding' directory
            full_path = os.path.join(self.data_loader.download_path_model.coding_path, file_name)

            # Check if the file exists at the constructed path
            if os.path.exists(full_path):
                try:
                    # Attempt to remove the file
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    # If an error occurs during deletion, print the error message
                    print(f"Error deleting {full_path}: {str(e)}")
            else:
                # If the file doesn't exist, inform the user
                print(f"File not found: {full_path}")

    def html_to_byte_strings_in_directory(self, directory_path: str) -> list:
        """
        Convert all HTML files in a directory into a list of byte strings.

        Args:
            directory_path (str): The path to the directory containing HTML files.

        Returns:
            list: A list of byte string representations of the HTML files.
        """
        byte_strings = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.html'):
                file_path = os.path.join(directory_path, file_name)
                with open(file_path, 'rb') as file:
                    byte_strings.append(file.read())
        return byte_strings

    def remove_duplicates_from_csvs(self, charts_directory: str) -> None:
        """
        Find all CSV files in the specified directory and remove duplicate rows from each file.

        Args:
            charts_directory (str): Path to the directory containing CSV files
        """
        # Get list of all CSV files in directory
        csv_files = [f for f in os.listdir(charts_directory) if f.endswith('.csv')]

        # remove duplicates from the csv files
        for csv_file in csv_files:
            file_path = os.path.abspath(os.path.join(charts_directory, csv_file))
            try:
                # Read CSV file
                df = pd.read_csv(file_path)

                # Drop duplicates and overwrite file
                df.drop_duplicates(inplace=True)
                df.to_csv(file_path, index=False)

            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")


class AgentFileHandler(AgentGeneralTools):
    """handle the file operations for the agent"""

    def __init__(self):
        """initialize the file handler"""

        # initialize the general tools
        super().__init__()

    def _process_csv_file_downloaded(self, file_name: str, renamed_headers: Dict[str, str], description: str) -> Dict[str, Any]:
        """
        Process a CSV file and return its details.

        Args:
            file_name (str): Name of the CSV file.
            renamed_headers (Dict[str, str]): Dictionary of renamed headers.
            description (str): Description of the file content.

        Returns:
            Dict[str, Any]: Dictionary containing file details.
        """
        max_retries = 3
        retry_delay = 1  # second

        for attempt in range(max_retries):
            try:
                # Construct the correct file path
                print(f"This is the filename inside _process_csv_file: {file_name}")
                file_path = self.data_loader.download_path_model.coding_path + '/' + file_name
                print(f"This is the file path inside _process_csv_file: {file_path}")

                # Check if the file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Get the original headers
                original_headers = df.columns.tolist()

                return {
                    'file_name': file_name,
                    'original_headers': original_headers,
                    'renamed_headers': renamed_headers,
                    'description': description
                }

            except (FileNotFoundError, IOError, pd.errors.EmptyDataError) as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"Error reading CSV file {file_name}: {str(e)}")
                    return {
                        'file_name': file_name,
                        'original_headers': [],
                        'renamed_headers': renamed_headers,
                        'description': description
                    }

    def _process_csv_file_json(self, file_name: str, renamed_headers: Dict[str, str], description: str, json_data: str) -> Dict[str, Any]:
        """
        Process a CSV file from JSON data and return its details.

        Args:
            file_name (str): Name of the CSV file.
            renamed_headers (Dict[str, str]): Dictionary of renamed headers.
            description (str): Description of the file content.
            json_data (str): JSON string containing the CSV data.

        Returns:
            Dict[str, Any]: Dictionary containing file details.
        """
        try:
            # Extract the full_dataframe from additional_info
            full_dataframe_json_string = str(json_data).replace("'", '"').replace("None", "null").replace("nan", "null")

            # Convert the JSON dataframe (list of dictionaries) back into a pandas DataFrame
            df = pd.read_json(full_dataframe_json_string)

            # Save the DataFrame as a CSV file
            df.to_csv(os.path.join(self.data_loader.download_path_model.coding_path, file_name), index=False)

            # Get the original headers
            original_headers = df.columns.tolist()

            return {
                'file_name': file_name,
                'original_headers': original_headers,
                'renamed_headers': renamed_headers,
                'description': description
            }

        except Exception as e:
            print(f"Error processing JSON data for file {file_name}: {str(e)}")
            return {
                'file_name': file_name,
                'original_headers': [],
                'renamed_headers': renamed_headers,
                'description': description
            }

    def _process_csv_file_predownloaded(self, file_name: str, renamed_headers: Dict[str, str], description: str) -> Dict[str, Any]:
        """process the csv file if it is predownloaded"""

        # output the response packet
        return {
            'file_name': file_name,
            'original_headers': [],
            'renamed_headers': {}, #pd.read_csv(os.path.join(self.download_path_model.coding_path, file_name)).columns.to_list(),
            'description': description
        }


class AnalyticalModelRetrieverProcessing(AgentFileHandler):
    """all the retriever functionality for the analytical model"""

    # set the json mode to true
    json_mode: bool = False # True
    predownloaded_mode: bool = True

    def __init__(self, stream_url: str = 'http://127.0.0.1:8005/stream'):
        # create directory called coding and charting
        os.makedirs("coding", exist_ok=True)

        # set the stream url
        self.stream_url = stream_url

        # define the data loader and handle class inheritance
        super().__init__()

        # initialize the agent llm config
        self.agent_llm_config = AgentLLMConfig()

        # define the general settings
        self.agent_general_settings = AgentGeneralSettings()

        # define the k top
        self.k = self.agent_llm_config.top_k_csv_files #self.data_loader.settings.top_k_csv_files

        # create the lancedb retriever
        self.table_descriptions_retriever = self._create_table_descriptions_retriever(k_top=self.k)

        self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")

        # define the llms
        self.hyde_llm, self.llm_primary_engineer, self.column_header_llm, self.planner_llm, self.follow_up_task_llm = self._define_llms_arr()

        # create the hyde prompt
        self.hyde_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant tasked with generating a hypothetical document description that could answer the following question about data analysis:

            {question}

            Provide a concise yet informative response that directly addresses the question. Your response should be written as if it were an excerpt from a relevant document or article describing a CSV file or dataset. Include details such as:

            1. The type of data the CSV might contain
            2. Potential column names and their descriptions
            3. The kind of analysis that could be performed with this data

            Your description should help in identifying the most relevant CSV files for this analysis question."""
        )

        # define the hyde chain
        self.hyde_chain = LLMChain(llm=self.hyde_llm, prompt=self.hyde_prompt)

        # define the streamed response
        self.streamed_response: str = ''

        # define the index for specalized file management
        self.index_specialized_file_management: dict = {1: self.specialized_file_management_db, 2: self.specialized_file_management}

    def _define_llms_arr(self):
        """define the llms for the analytical model"""
        try:
            if self.agent_llm_config.private_llm_mode:
                os.environ["OPENAI_API_VERSION"] = "2024-02-01"
                os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

                # self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")

                llm = AzureChatOpenAI(
                    temperature=self.data_loader.settings.summary_temp,
                    azure_deployment=self.agent_llm_config.model_deployment_name,
                    # max_retries=2,
                    max_tokens=self.agent_llm_config.llm_summary_max_tokens,
                    callbacks=[self.stream_handler]
                )
            else:
                # create the llm for the csv analysis
                llm = ChatOpenAI(model_name=self.data_loader.settings.csv_analysis_model,
                                temperature=self.data_loader.settings.summary_temp,
                                #callbacks=[self.stream_handler]
                                )

        except:
            raise Exception("Invalid API key provided")

        # define llm for the primary analysis
        try:
            if self.agent_llm_config.private_llm_mode:
                os.environ["OPENAI_API_VERSION"] = "2024-02-01"
                os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

                # self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")

                # define the primary engineer llm
                llm_primary_engineer = AzureChatOpenAI(
                    temperature=self.agent_llm_config.llm_primary_engineer_temperature,
                    azure_deployment=self.agent_llm_config.model_deployment_name,
                    max_tokens=self.agent_llm_config.llm_primary_engineer_max_tokens,
                    # max_retries=2,
                    # max_tokens=self.model_specs.max_tokens,
                    callbacks=[self.stream_handler]
                )
            else:
                # define the engineer llm
                llm_primary_engineer = ChatOpenAI(model_name=self.agent_llm_config.llm_primary_engineer_model,
                                        temperature=self.agent_llm_config.llm_primary_engineer_temperature,
                                        max_tokens=self.agent_llm_config.llm_primary_engineer_max_tokens,
                                        #callbacks=[self.stream_handler]
                                        )

        except:
            raise Exception("Invalid API key provided")

        # check if the column header model is gpt-4o-mini so that we can use the same llm for the csv analysis and the column header model
        if self.data_loader.settings.csv_analysis_model == 'gpt-4o-mini':
            column_header_llm = llm
        else:
            # Initialize the LangChain ChatOpenAI model
            try:
                if self.agent_llm_config.private_llm_mode:
                    os.environ["OPENAI_API_VERSION"] = "2024-02-01"
                    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                    os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

                    # self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")

                    column_header_llm = AzureChatOpenAI(
                        temperature=0,
                        azure_deployment=self.agent_llm_config.model_deployment_name,
                        # max_retries=2,
                        max_tokens=100,
                        callbacks=[self.stream_handler]
                    )
                else:
                    # create the llm for the csv analysis
                    column_header_llm = ChatOpenAI(model_name='gpt-4o-mini',
                                                temperature=0,
                                                openai_api_key=api_key,
                                                #callbacks=[self.stream_handler]
                                                )

            except:
                raise Exception("Invalid API key provided")

        try:
            if self.agent_llm_config.private_llm_mode:
                os.environ["OPENAI_API_VERSION"] = "2024-02-01"
                os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint
                os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key

                # self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")

                # define the planner llm
                planner_llm = AzureChatOpenAI(
                    temperature=self.agent_llm_config.llm_planner_temperature,
                    azure_deployment=self.agent_llm_config.model_deployment_name,
                    # max_retries=2,
                    max_tokens=self.agent_llm_config.llm_planner_max_tokens,
                    callbacks=[self.stream_handler]
                )

                # define the follow up task llm
                follow_up_task_llm = AzureChatOpenAI(
                    temperature=self.agent_llm_config.llm_follow_up_task_temperature,
                    azure_deployment=self.agent_llm_config.model_deployment_name,
                    max_tokens=self.agent_llm_config.llm_follow_up_task_max_tokens,
                    # max_retries=2,
                    callbacks=[self.stream_handler]
                )

            else:
                # define the planner llm as different model because of max tokens settings
                planner_llm = ChatOpenAI(model_name=self.agent_llm_config.llm_planner_model,
                                        temperature=self.agent_llm_config.llm_planner_temperature,
                                        #callbacks=[self.stream_handler]
                                        )

                # define the follow up task llm
                follow_up_task_llm = ChatOpenAI(model_name=self.agent_llm_config.llm_follow_up_task_model,
                                                temperature=self.agent_llm_config.llm_follow_up_task_temperature,
                                                max_tokens=self.agent_llm_config.llm_follow_up_task_max_tokens,
                                                #callbacks=[self.stream_handler]
                                                )

        except:
            raise Exception("Invalid API key provided")


        return llm, llm_primary_engineer, column_header_llm, planner_llm, follow_up_task_llm

    def update_stream_url(self, new_url: str) -> None:
        """Update the stream URL and reinitialize LLMs with new stream handler"""
        self.stream_url = new_url
        self.stream_handler.update_url(new_url)
        #self._define_llms_arr()

    def _create_table_descriptions_retriever(self, k_top: int = None):
        """Create the retriever for the LanceDB database table descriptions"""

        # Define the reranker
        reranker = LinearCombinationReranker(
            weight=self.data_loader.settings.hybrid_search_ratio,
        )

        # Initialize the custom retriever object for the TABLE DESCRIPTIONS retriever
        table_retriever = LanceTableRetriever(
            table=self.data_loader.lancedb_client[3],
            reranker=reranker,
            k=k_top if k_top is not None else self.data_loader.settings.k_top * self.data_loader.settings.k_tab_multiplier,  # Ensure k is set
            mode='fts'  # Ensure mode is set
        )
        print(table_retriever)

        return table_retriever

    # return pandas dataframes as a list of dictionaries
    def fetch_relevant_tables(self, query: str) -> List[Dict[str, Any]]:
        """Fetch the relevant tables for the query"""

        # Use the custom table descriptions retriever and select to always use tabular data (True returns an empty list)
        relevant_docs = self.table_descriptions_retriever.get_relevant_documents(query, disable_tabular=False)[:self.k]

        results = []
        for doc in relevant_docs:  # Iterate through each relevant document
            # check if the file is predownloaded
            if self.predownloaded_mode:
                file_details = self._process_csv_file_predownloaded(doc.metadata['source'], {}, doc.page_content)
                results.append(file_details)
                continue

            try:
                # Attempt to parse the JSON string into a dictionary
                additional_info = json.loads(doc.metadata['additional_info'])
            except json.JSONDecodeError:
                # If JSON parsing fails, log an error message
                print(f"Error decoding additional_info JSON for document: {doc.metadata['source']}")

            # Extract renamed headers from the parsed additional_info
            renamed_headers = additional_info['renamed_headers']

            # Get the file name from metadata, default to 'Unknown' if not found
            file_name = doc.metadata['source']

            # Process the file based on the current mode (JSON or downloaded CSV)
            if not self.json_mode:
                # If in JSON mode, process the downloaded CSV file
                file_details = self._process_csv_file_downloaded(file_name, renamed_headers, doc.page_content)
            else:
                # get the full dataframe from the additional info section
                json_csv_data = str(additional_info['full_dataframe'])

                # process the json data by downloading the file into CSV and reading it as string
                file_details = self._process_csv_file_json(file_name, renamed_headers, doc.page_content, json_csv_data)

            # Add the processed file details to the results list
            results.append(file_details)

        self.streamed_response += "Identified the following relevant files: " + str(results) + "\n"
        self.streamed_response +=  "-----------------------------------------\n"
        #print(self.streamed_response)
        return results

    def generate_hypothetical_document(self, question: str) -> str:
        """Generate a hypothetical document description based on the question."""
        return self.hyde_chain.run(question)

    def retrieve_with_hyde(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using HyDE."""

        # generate the hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(question)

        # Use the table_descriptions_retriever to find similar documents
        relevant_docs = self.table_descriptions_retriever.get_relevant_documents(hypothetical_doc)

        results = []
        for doc in relevant_docs[:k]:  # Limit to top k results
            # get the file name
            file_name = doc.metadata.get('source', 'Unknown')

            # get the renamed headers
            renamed_headers = doc.metadata.get('additional_info', {}).get('renamed_headers', {})

            # Use the _process_csv_file method to get file details
            file_details = self._process_csv_file(file_name, renamed_headers, doc.page_content)
            results.append(file_details)

        return results

    def create_user_queries(self, context: str) -> str:
        """Create a user query based on the context from the chat."""

        # Use HyDE to retrieve relevant documents
        relevant_docs = self.retrieve_with_hyde(context)

        # Generate an insightful query based on the relevant documents
        query_generation_prompt = PromptTemplate(
            input_variables=["context", "relevant_docs"],
            template="""Given the following context and available relevant CSV files, suggest one insightful data analysis question:

            Context: {context}

            Available CSV files:
            {relevant_docs}

            Generate one specific, analytical question that can be answered using the available data. This question should:
            1. Require in-depth data analysis
            2. Potentially involve data visualization
            3. Provide valuable insights related to the context
            4. Be answerable using the available CSV files

            Ensure the question is clear, concise, and directly relevant to the given context and data."""
        )

        # create the query generation chain
        query_generation_chain = LLMChain(llm=self.hyde_llm, prompt=query_generation_prompt)

        # run the query generation chain
        query = query_generation_chain.run(context=context, relevant_docs=json.dumps(relevant_docs, indent=2))

        # Return the query as a single-item list to maintain compatibility with existing code
        return query.strip()

    # WITHOUT using postgres database to store CSV files
    def specialized_file_management(self, user_question: str, **kwargs) -> List[list]:
        """The main method to call the class which also downloads"""

        # Retrieve the kwargs argument
        file_name = kwargs.get('file_name', None)

        # If the file name is not None, use the file name
        if file_name is not None:
            # Download the file to the coding directory
            try:
                self.data_loader.downloader.download(f'/{file_name}', source_dir=self.data_loader.download_path_model.table_path, local_dir=self.data_loader.download_path_model.coding_path, move=True)
                print(f"File '{file_name}' downloaded successfully to {self.data_loader.download_path_model.coding_path}")
            except Exception as e:
                raise Exception(f"Error downloading file '{file_name}': {str(e)}")

            # process file with csv processor function
            return [[self._process_csv_file_downloaded(f'{self.data_loader.download_path_model.coding_path}/{file_name}', {}, '')], [f'/{file_name}']]

        # If the file name is None, use the user question to find relevant files
        else:
            # Fetch relevant tables based on the user question
            relevant_tables: list = self.fetch_relevant_tables(user_question)

            # if no relevant tables are found, raise an exception
            if not relevant_tables:
                raise Exception("No relevant files found for the given question.")

            # download the files
            file_paths: list = []
            for table in relevant_tables:
                file_name = table['file_name']
                try:
                    if not self.json_mode:
                        # Download the file to the coding directory
                        self.data_loader.downloader.download(f'/{file_name}', source_dir=self.data_loader.download_path_model.table_path, local_dir=self.data_loader.download_path_model.coding_path, move=True)

                    # construct the file path and add it to the list
                    file_paths.append(f'{self.data_loader.download_path_model.coding_path}/{file_name}')
                except Exception as e:
                    raise Exception(f"Error downloading file '{file_name}': {str(e)}")

            # return the relevant tables and file paths as a list of dictionaries
            return [relevant_tables, file_paths]

    # WITH using postgres database to store CSV files
    def specialized_file_management_db(self, user_question: str, **kwargs) -> List[list]:
        """
        Alternative version of specialized_file_management that uses database retrieval
        instead of file downloads.

        Args:
            user_question (str): The user's question to analyze
            **kwargs: Additional arguments, may include 'file_name'

        Returns:
            List[list]: A list containing [relevant_tables, file_paths]
        """

        # Handle single file case
        file_name = kwargs.get('file_name', None)
        if file_name is not None:
            try:
                # Retrieve single file from database
                df = self.data_loader.csv_db_handler.retrieve_csv(
                    table_name=self.data_loader.storage_settings.csv_db_table_name,
                    name=f'{self.data_loader.download_path_model.table_path}/{file_name}',
                    output_path=f'{self.data_loader.download_path_model.coding_path}/{file_name}'
                )

                # Process the dataframe
                processed_result = self._process_csv_file_downloaded(df, {}, '')
                return [[processed_result], [f'/{file_name}']]

            except Exception as e:
                raise Exception(f"Error retrieving file '{file_name}' from database: {str(e)}")

        # Handle multiple files case
        else:
            # Fetch relevant tables based on the user question
            relevant_tables: list = self.fetch_relevant_tables(user_question)
            print('len of relevant tables: ', len(relevant_tables))

            if not relevant_tables:
                raise Exception("No relevant files found for the given question.")

            try:
                # Extract file names from relevant_tables
                file_names = [table['file_name'] for table in relevant_tables]

                # Create file paths (virtual paths since we're not actually downloading)
                file_paths_retrieval = [f'{self.data_loader.download_path_model.table_path}/{name}'
                            for name in file_names]

                # create the file paths for the output
                file_paths = [f'{self.data_loader.download_path_model.coding_path}/{name}'
                            for name in file_names]

                # Use the database handler to retrieve dataframes
                dataframes = self.data_loader.csv_db_handler.retrieve_multiple_csv(
                    table_name=self.data_loader.storage_settings.csv_db_table_name,
                    file_names=file_paths_retrieval,
                    output_paths=file_paths
                )

                # Update relevant_tables with the retrieved dataframes
                processed_results = []
                for file_name, dataframe in dataframes.items():
                    #processed_result = self._process_csv_file_downloaded(dataframe, {}, '')
                    processed_results.append(dataframe)
                return [processed_results, file_paths]

            except Exception as e:
                raise Exception(f"Error retrieving files from database: {str(e)}")

    def search_metadata_by_filename(self, file_name: str) -> bool:
        """
        Search the LanceDB database for a file name in the source metadata field.

        Args:
            file_name (str): The file name/path to search for

        Returns:
            bool: True if file is found in database, False otherwise
        """
        try:
            # Get the table from LanceDB client
            table = self.data_loader.lancedb_client[3]

            # Search the source field in metadata for the file name
            results = table.search().where(f"source = '{file_name}'").to_list()

            # Return True if any results found, False otherwise
            return len(results) > 0

        except Exception as e:
            print(f"Error searching metadata: {str(e)}")
            return False


class AgentUtilities(AnalyticalModelRetrieverProcessing):
    """class to configure the agent and provide runtime capabilities"""

    def __init__(self, stream_url: str):
        super().__init__(stream_url)

        # define the system messages and engineer planner prompts
        self.system_msg_engineer, self.system_msg_planner, \
            self.system_msg_critic, self.system_msg_summary, self.system_msg_question_enhancer, self.system_msg_debugger = system_msg_engineer, system_msg_planner, system_msg_critic, system_msg_summary, system_msg_question_enhancer, system_msg_debugger

        # define the structured output
        self.code_output_model = CodeOutput

        # number of columns after filtering
        self.number_columns_after_filtering = 15

        # column threshold - this is the size of the spreadsheet beyond which we need column filtering
        self.column_threshold = 200

    def llm_column_filtering(self, df, question):
        """call the language model to get the relevant columns"""

        # Get the list of column names from the DataFrame
        columns = df.columns.tolist()

        # Prepare the prompt
        prompt = """
        Given the following list of CSV file column names:
        {columns}

        And the following question:
        "{question}"

        Please provide a list of up to {number_columns_after_filtering} column names from the DataFrame that are most relevant to answering the question.
        Example output is: ["column1", "column2", "column3", etc...]. Do not return Python code.
        """

        # define the content
        content: list = []
        content.append({"type": "text", "text": prompt.format(columns=columns, question=question, number_columns_after_filtering=self.number_columns_after_filtering)})

        # create the messages
        messages = [HumanMessage(content=content)]

        # Use the LLM to generate the response
        response = self.column_header_llm.invoke(messages)

        # Extract the assistant's reply
        assistant_reply = response.content.strip()

        # Parse the response into a list of column names
        try:
            relevant_columns = ast.literal_eval(assistant_reply)
            return relevant_columns, columns
        except:
            try:
                # call the response agan
                response = self.column_header_llm([HumanMessage(content=prompt)])
                assistant_reply = response.content.strip()
                relevant_columns = ast.literal_eval(assistant_reply)
                return relevant_columns, columns
            except:
                raise ValueError("Failed to parse the assistant's reply into a Python list.")

    def get_relevant_columns_with_lm_singular(self, question, file_path: str, df: pd.DataFrame = None, postgres_mode: bool = True):
        # read the pandas dataframe from the file information
        if not postgres_mode:
            df = pd.read_csv(file_path)

        # find the condition for column filtering
        column_filtering_condition = df.shape[1] < self.column_threshold

        # add a condition if the dataframe is smaller than 50 columns then just use the original dataframe
        if column_filtering_condition:
            relevant_columns = df.columns.tolist()
            return relevant_columns, df, df.head(3), self._get_dataframe_info(df)

        # get the relevant columns
        relevant_columns, columns = self.llm_column_filtering(df, question)
        self.streamed_response += "Identified the following relevant columns: " + str(relevant_columns) + "\n"
        self.streamed_response +=  "-----------------------------------------\n"

        # If a column does not exist, use cosine similarity to find the closest matching column
        valid_columns = []
        for col in relevant_columns:
            if col in columns:
                valid_columns.append(col)
            else:
                # Find the most similar column
                most_similar_column = self.revise_column_name_cosine(col, columns)
                valid_columns.append(most_similar_column)

        # Remove duplicates and limit to up to 20 columns
        valid_columns = list(dict.fromkeys(valid_columns))[:20]

        # Subset the DataFrame with the relevant columns
        subset_df = df[valid_columns]

        # save the new dataframe
        if column_filtering_condition:
            self.save_dataframe(subset_df, filename=file_path) # specify a file name in general

        # generate the dataframe info and description
        return valid_columns, subset_df, subset_df.head(10), str(subset_df.describe()) #self._get_dataframe_info(subset_df)

    def get_relevant_columns_with_lm(self, question, file_paths: list = None, dataframes: list = None, postgres_mode: bool = True):
        """get the relevant columns with the language model for multiple files"""

        # implement concurrent.futures to call the function for each file name
        with ThreadPoolExecutor(max_workers=len(file_paths)) as executor:
            # map the function to the file names
            results = list(executor.map(self.get_relevant_columns_with_lm_singular,
                                    [question] * len(file_paths),
                                    file_paths,
                                    dataframes,
                                    [postgres_mode] * len(file_paths)))

        # combine the results by category
        relevant_columns_all = []
        subset_dfs_all = []
        subset_df_heads_all = []
        subset_df_infos_all = []

        # iterate through the results
        for result in results:
            # unpack the result
            relevant_columns, subset_df, subset_df_head, subset_df_info = result

            # append the result to the list
            relevant_columns_all.append(relevant_columns)
            subset_dfs_all.append(subset_df)
            subset_df_heads_all.append(subset_df_head)
            subset_df_infos_all.append(subset_df_info)

        # return a list of 4 lists
        return [relevant_columns_all, subset_dfs_all, subset_df_heads_all, subset_df_infos_all]

    def revise_column_name_cosine(self, col, columns):
        """find the most similar column name from a list of columns using cosine similarity"""

        # create the vectorizer
        vectorizer = TfidfVectorizer().fit_transform([col] + columns)

        # convert the vectorizer to an array
        vectors = vectorizer.toarray()

        # calculate the cosine similarity
        cosine_matrix = cosine_similarity(vectors[0:1], vectors[1:])

        # find the most similar column
        most_similar_index = cosine_matrix[0].argmax()

        # return the most similar column
        most_similar_column = columns[most_similar_index]
        return most_similar_column

    def save_dataframe(self, df, filename=None):
        # Generate a unique filename if not provided
        if filename is None:
            filename = f"dataframe_{uuid4().hex}.csv"
        else:
            # Ensure the filename has a proper extension
            if not filename.endswith('.csv'):
                filename += '.csv'

        # Save the DataFrame to the specified file
        df.to_csv(filename, index=False)
        print(f"DataFrame saved to {filename}")
        return filename

    def clean_chat_history(self, chat_history: List[Dict]) -> List[Dict]:
        """
        Clean up the chat history by removing messages that are blank or do not contain code.

        This method filters the chat history to keep only non-empty messages that contain
        code blocks (indicated by triple backticks ```).

        Args:
            chat_history (List[Dict]): The list of message dictionaries from the chat history.

        Returns:
            List[Dict]: The cleaned chat history, containing only messages with code blocks.
        """
        cleaned_history = []  # Initialize an empty list to store the cleaned messages

        print(chat_history)  # Debug: Print the entire chat history

        # Iterate through each message in the chat history
        for message in chat_history:
            # Extract the content of the message, remove leading/trailing whitespace
            content = message.get('content', '').strip()

            # Check if the content is not empty and contains code block indicators
            if content and '```' in content:
                # If the message has content and contains a code block, add it to the cleaned history
                cleaned_history.append(message)

        # Return the list of cleaned messages
        return cleaned_history

    def clean_coding_directory(self, file_paths: list):
        """
        Remove specified files from the coding directory

        Args:
            file_paths: List of file paths to remove
        """
        coding_dir = self.data_loader.download_path_model.coding_path
        if not os.path.exists(coding_dir):
            return

        for file_path in file_paths:
            # Get just the filename from the full path
            filename = os.path.basename(file_path)
            full_path = os.path.join(coding_dir, filename)

            # Remove file if it exists
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                    print(f"Removed file: {filename}")
                except Exception as e:
                    print(f"Error removing file {filename}: {str(e)}")
            else:
                print(f"File not found: {filename}")

    def create_dataframe_information(self, file_paths: list, dataframes_infos: list, dataframes_heads: list):
        """create the dataframe info"""

        # Prepare relevant_columns_info
        relevant_columns_info = "\n".join([f"{file}: {', '.join(df.columns)}" for file, df in zip(file_paths, dataframes_heads)])

        # Prepare dataframes_info
        dataframes_info = ""
        for file_path, _, df_head in zip(file_paths, dataframes_infos, dataframes_heads):
            dataframes_info += f"File Path: {file_path}\n"
            dataframes_info += f"Dataframe Column Names:\n{df_head.columns.tolist()}\n\n"
            #dataframes_info += f"Dataframe Info:\n{df_info}\n\n"
            dataframes_info += f"Dataframe Head:\n{df_head.to_string()}\n\n"

        return dataframes_info, relevant_columns_info


class ProcessAgentResponse(AgentUtilities):
    """class to process the agent response"""

    def __init__(self, stream_url: str):
        super().__init__(stream_url)

        # define the imgkit options
        self.imgkit_options = {
            'quality': 50,  # Lower quality for faster conversion
            'width': 1024,  # Set a fixed width
            'disable-smart-width': '',  # Disable smart width calculation
            'quiet': ''  # Reduce console output
        }

    @staticmethod
    def _encode_image_to_base64(image_file_path: str) -> str:
        """encode an image to base64"""

        # open the image file
        with open(image_file_path, "rb") as image_file:
            # encode the image to base64
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return encoded_image

    def _convert_HTML_to_PNG_and_encode(self, html_file_path: str) -> str:
        """
        Convert an HTML file to a PNG file and encode it to base64 for GPT-4o vision model.

        Args:
            html_file_path (str): Path to the HTML file.

        Returns:
            str: Base64 encoded string of the generated PNG image.

        Raises:
            FileNotFoundError: If the HTML file doesn't exist.
            imgkit.IMGKitError: If there's an error during conversion.
        """

        # check if the html file exists
        if not os.path.exists(html_file_path):
            raise FileNotFoundError(f"HTML file not found: {html_file_path}")

        # Generate output PNG filename
        png_file_path = os.path.splitext(html_file_path)[0] + '.png'

        try:
            # Convert HTML to PNG
            imgkit.from_file(html_file_path, png_file_path, options=self.imgkit_options)

            # encode the image to base64
            encoded_image = self._encode_image_to_base64(png_file_path)

            # Clean up the temporary PNG file
            os.remove(png_file_path)

            return encoded_image

        except imgkit.IMGKitError as e:
            print(f"Error converting HTML to PNG: {e}")
            raise
        except Exception as e:
            print(f"Error encoding image to base64: {e}")
            raise

    def _define_results_analysis_prompt(self, question: str, image_data: Any):
        """define the results analysis prompt which can be used to analyze the results of the agent's work"""

        # define the prompt
        prompt = """
        You are an expert data analyst. Your task is to analyze the results of the agent's work and provide a summary of the results.
        The results are charts in image form. You are also given the original question that the agent was asked.

        User Question: {question}
        """

        # define the message for the language model to analyze the results
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt.format(question=question)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )

        return message

    def fetch_results_analysis(self, question: str, image_paths: List[str]) -> str:
        """
        Get the analysis of the results produced by the analytics model for multiple images.

        Args:
            question (str): The original question asked by the user.
            image_paths (List[str]): A list of paths to the image files (HTML or PNG).

        Returns:
            str: The combined analysis of all images.
        """
        def process_image(image_path):
            if image_path.lower().endswith('.html'):
                image_data = self._convert_HTML_to_PNG_and_encode(image_path)
            elif image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                image_data = self._encode_image_to_base64(image_path)
            else:
                return None
            return self._define_results_analysis_prompt(question=question, image_data=image_data)

        all_messages = []

        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_image = {executor.submit(process_image, image_path): image_path for image_path in image_paths}
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_messages.extend(result)
                except Exception as exc:
                    print(f'{image_path} generated an exception: {exc}')

        # Get the result from the language model
        response = self.llm_primary_engineer.invoke(all_messages)

        return response.content


# define the agent graph class
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


class ConstructAgentGraph(AgentUtilities):
    """class to construct the agent graph"""

    # conditional mode
    conditional_mode = True

    # define if code should run task decomposition or not
    run_task_decomposition: bool = False

    def __init__(self, stream_url: str):
        super().__init__(stream_url)

        # define the graph builder
        self.graph_builder = StateGraph(AgentState)

        # define the command line code
        self.executor = lambda x: LocalCommandLineCodeExecutor(
            timeout=60,  # Timeout for each code execution in seconds.
            work_dir=x.name,  # Use the temporary directory to store the code files.
        )

        # define the state variables container
        self.state_variables: dict = {
            'code_execution_output': '',
            'critic_feedback': '',
            'current_code': '',
            'question': ''
        }

        # define the model id
        self.model_id: int = 2

    def _process_streamed_response(self, response: str) -> str:
        """Process the streamed response to display chunks cleanly

        Args:
            response (str): Raw streamed response

        Returns:
            str: Complete concatenated response
        """

        # local response
        local_response: str = ''

        # Process each chunk
        for chunk in response:
            # Get chunk content and add to response
            chunk_content = chunk.content

            # add the chunk content to the streamed response both globally and locally
            self.streamed_response += chunk_content
            local_response += chunk_content

            # Print only the new content
            if chunk_content.strip():  # Only print if there's content after stripping whitespace
                print(chunk_content, end='', flush=True)  # Use end='' to avoid extra newlines

        return local_response

    def critic_router(self, state):
        """route for the critic"""

        # This is the router
        messages = state["messages"]
        if self.conditional_mode:
            last_message = messages[-1][-1]
        else:
            last_message = messages[-1]
        if "TERMINATE" in last_message:
            # Any agent decided the work is done
            return 'summary_exit'
        return "continue"

    def _create_task_decomposition_call(self, question: str, dataframes_infos: List[str], dataframes_heads: List[pd.DataFrame], relevant_columns: List[List[str]], file_paths: list):
        """Create the task decomposition call for the planner"""

        # create the dataframe information
        dataframes_info = self.create_dataframe_information(file_paths, dataframes_infos, dataframes_heads)

        # Prepare relevant_columns_info
        relevant_columns_info = "\n".join([f"{file}: {', '.join(df.columns)}" for file, df in zip(file_paths, dataframes_heads)])

        # define the content messages container
        content: list = []

        # fill in the planner template
        planner_template = self.system_msg_planner.format(question=question, dataframes_info=dataframes_info, relevant_columns=relevant_columns_info, plotly_charts=self.agent_general_settings.plotly_charts, output_directory=self.agent_llm_config.file_save_path)
        content.append({"type": "text", "text": planner_template})

        # define the human message
        messages = [HumanMessage(content)]

        print('input token count task decomposition:')
        self.count_tokens(planner_template)

        # Call the language model to generate the response
        planner_response = self.llm_primary_engineer.stream(messages)

        return self._process_streamed_response(planner_response), dataframes_info, relevant_columns_info

    def _create_prompt(self, query: str, image_input: str = ''):
        """create the LLM prompt - identify the correct file as well"""

        # get the correct file to use
        start_time = time.time()
        dataframes, file_paths = self.index_specialized_file_management[self.agent_llm_config.csv_file_manager](query, file_name=self.specific_file_name)
        end_time = time.time()
        print(f"Time taken to fetch file paths from inside create_prompt: {end_time - start_time} seconds")

        # extract the relevant columns from the
        relevant_columns, _, subset_df_head_list, subset_df_info_list = self.get_relevant_columns_with_lm(
            question=query,
            file_paths=file_paths,
            dataframes=dataframes,
            postgres_mode=self.data_loader.settings.postgres_mode
        )

        # Call _create_task_decomposition_call with correct arguments
        if self.run_task_decomposition:
            start_time = time.time()
            task, dataframes_info, dataframe_columns = self._create_task_decomposition_call(
                question=query,
                dataframes_infos=subset_df_info_list,
                dataframes_heads=subset_df_head_list,
                relevant_columns=relevant_columns,  # This should be a list of lists, not a dict
                file_paths=file_paths
            )
            end_time = time.time()
            print(f"Time taken to create task: {end_time - start_time} seconds")
        else:
            dataframes_info, dataframe_columns = self.create_dataframe_information(file_paths, subset_df_info_list, subset_df_head_list)
            task = query

        return task, dataframes_info, dataframe_columns, file_paths

    def initial_code_generation_node(self, prompt: str, dataframes_info: str, dataframe_columns: list, image_input: str = None):
        """
        Generate the starter code using the LLM with support for text and image inputs.

        Args:
            prompt (str): The text prompt for code generation
            dataframes_info (str): The dataframes information
            image_input (str, optional): Base64 encoded image string
        """

        # define the content of the prompt
        content: list = []

        if True:
            # Create the system message
            message = self.system_msg_engineer.format(task=prompt, dataframes_info=dataframes_info,
                                                      code_directory_path=self.agent_llm_config.file_save_path)
            # add the prompt to the content
            content.append({"type": "text", "text": message})

            # add the image to the content
            if image_input:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_input}"},
                })

            # create the human message
            message = [HumanMessage(content)]

            # Print token count for monitoring
            print('input token count code generation:')
            self.count_tokens(str(message))

            # call the language model
            response = self.llm_primary_engineer.stream(message)

            return self._process_streamed_response(response)

        #except Exception as e:
        #    print(f"Error generating code: {str(e)}")
        #    return None

    def create_first_draft_code_node_modified(self, question: str, manual_mode: bool = True, image_input: str = ''):
        """create the first draft code agent"""
        task, dataframes_info, dataframe_columns, file_paths = self._create_prompt(question, image_input)

        if manual_mode:
            # call the llm directly and get the response
            return self.initial_code_generation_node(task, dataframes_info, dataframe_columns, image_input), dataframes_info, file_paths
        else:
            first_draft_agent = self.create_agent_wrapper(self.llm_primary_engineer,
                                                        [self.execute_code],
                                                        system_message=task)

            first_draft_node = functools.partial(self.agent_node, agent=first_draft_agent, name="First Draft Agent")

            return first_draft_node

    def get_previous_chat_history(self):
        """
        Get the last chat history entry along with the code created for the question.

        Returns:
            Tuple[str, str] | None: A tuple containing (question, code_created)
                                    for the last interaction, or None if history is empty.
        """
        if not self.chat_history:
            return (None, None)

        last_interaction = self.chat_history[-1]
        question = last_interaction.get('question', '')
        code_created = last_interaction.get('code_created', '')
        return (question, code_created)

    def follow_up_task_router(self, state):
        """router for the follow up task"""

        # define the reference message
        if self.conditional_mode:
            reference_message = state['messages'][-1][-1]
        else:
            reference_message = state['messages']

        if  'yes' in reference_message.lower():
            # The previous agent is invoking a tool
            return "call_edit"
        if "no" in reference_message.lower():
            # Any agent decided the work is done
            return 'call_draft_code'

    def _parse_code_execution_response(self, response: str):
        """parse the code execution response and remove warnings from output"""

        # Parse the response
        content = response
        lines = content.split('\n')

        execution_status = ''
        code_output = ''
        error_message = ''

        # Process lines for code output, removing warnings
        output_lines = []
        warning_line = False

        for line in lines:
            if line.startswith('exitcode:'):
                execution_status = 'success' if 'execution succeeded' in line else 'failed'
            elif line.startswith('Code output:'):
                # Get all lines after 'Code output:'
                output_section = lines[lines.index(line)+1:]

                # Filter out warning lines
                for out_line in output_section:
                    # Skip lines containing warnings
                    if any(warn in out_line for warn in ['Warning:', 'FutureWarning:', 'UserWarning:', 'DeprecationWarning:']):
                        continue
                    # Skip lines that are continuations of warnings (usually indented)
                    if out_line.startswith('  ') and warning_line:
                        continue
                    # Reset warning line flag and keep the line
                    warning_line = False
                    output_lines.append(out_line)

                code_output = '\n'.join(line for line in output_lines if line.strip())
                break

        if execution_status == 'failed':
            error_message = code_output
            code_output = ''

        return {
            'execution_status': execution_status,
            'console_output': code_output,
            'error': error_message
        }

    def execute_code(self, starter_code: Annotated[str, "The starter code to execute"]):
        """create the code execution agent"""

        # Create a temporary directory to store the code files.
        temp_dir = tempfile.TemporaryDirectory()

        # Create an agent with code executor configuration.
        code_executor_agent = ConversableAgent(
            name="code_executor_agent",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": self.executor(temp_dir)},  # Use the local command line code executor.
            human_input_mode="NEVER",  # Never take human input for this agent for automation.
        )

        # Generate reply (execute the code)
        response = code_executor_agent.generate_reply(messages=[{"role": "user", "content": starter_code}])

        # Clean up the temporary directory
        self.temp_file_cleanup(temp_dir)

        return self._parse_code_execution_response(response)

    def create_enhanced_question_node(self, question: Annotated[str, "The user's question"]):
        """create a function that enhances a user's question when it is unclear"""

        # define the prompt
        content: list = []
        content.append({
            "type": "text",
            "text": self.system_msg_question_enhancer.format(
                question=question,
                industry=self.data_loader.sector_settings.company_sector,
                business_description=self.data_loader.sector_settings.company_description
            )
        })

        # define the messages
        messages = [HumanMessage(content)]

        # call the llm directly and get the response
        response = self.follow_up_task_llm.stream(messages)
        return self._process_streamed_response(response)

    @tool
    def code_executor_tool(self, starter_code: Annotated[str, "The starter code to execute"]):
        """tool to execute code"""
        return self.execute_code(starter_code)

    def create_critic_node(self, question: str, code_output: Annotated[str, "The code output to critique"], starter_code: Annotated[str, "The starter code to execute"], dataframes_info: Annotated[str, "The dataframes info"], manual_mode: bool = True):
        """create the critic agent"""

        if manual_mode:
            # define the content
            content: list = []
            content.append({"type": "text", "text": self.system_msg_critic.format(question=question, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info, code_directory_path=self.agent_llm_config.output_save_path)})

            # define the messages
            messages = [HumanMessage(content)]

            # call the llm directly and get the response
            response = self.llm_primary_engineer.stream(messages)
            return self._process_streamed_response(response)
        else:
            # create the agent
            critic_agent = self.create_agent_wrapper(self.llm_primary_engineer, [self.dummy_tool], system_message=self.system_msg_critic.format(question=question, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info))
            critic_node = functools.partial(self.agent_node, agent=critic_agent, name="Critic")

            return critic_node

    def create_code_engineer_node(self, task: str,
                                   critic_feedback: Annotated[str, "The feedback from the critic"],
                                   code_output: Annotated[str, "The code output to critique"],
                                   starter_code: Annotated[str, "The starter code to execute"],
                                   dataframes_info: Annotated[str, "The dataframes info"],
                                   manual_mode: bool = True):
        """create the code engineer agent"""


        if manual_mode:
            # define the content
            content: list = []
            content.append({"type": "text", "text": self.system_msg_debugger.format(task=task, critic_feedback=critic_feedback, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info, code_directory_path=self.agent_llm_config.output_save_path)})

            # define the messages
            messages = [HumanMessage(content)]

            # call the llm directly and get the response
            response = self.llm_primary_engineer.stream(messages)
            return self._process_streamed_response(response)
        else:
            # create the agent
            code_engineer_agent = self.create_agent_wrapper(self.llm_primary_engineer, [self.execute_code], system_message=self.system_msg_debugger.format(task=task, critic_feedback=critic_feedback, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info))
            engineer_node = functools.partial(self.agent_node, agent=code_engineer_agent, name="Code Engineer")

            return engineer_node

    def create_followup_code_node(self, question: str, starter_code: str, dataframes_info: Annotated[str, "The dataframes info"], manual_mode: bool = True, image_input: str = ''):
        """
        Create the draft started code agent with support for image inputs.

        Args:
            question (str): User's follow-up question
            starter_code (str): Previous code to modify
            dataframes_info (str): Information about available dataframes
            manual_mode (bool): Whether to use manual mode or agent mode
            image_input (str): Base64 encoded image string
        """


        # define the content
        content: list = []

        # add the system message first
        content.append({"type": "text", "text": """You are a Python Code Engineer. Your task is to implement changes based on the user's question and image if provided. Focus on:
                1. Addressing the user's question directly.
                2. Fixing any errors or bugs in the code.
                3. Ensuring the code runs without errors and answers the user's follow up question.
                4. Make sure all charts in the code are done in plotly with an asthetically pleasing template
                5. Make sure charts have a title and axis labels
                6. Make sure to save them files to the output directory as a HTML file
                7. When editing an existing chart, make sure to save the edited chart to the same file path and overwrite the old chart.
                8. Select which dataframes to use. Only use the dataframes provided to you below."""})

        # add the human message
        content.append({"type": "text", "text": f"""Previous Code:
                {starter_code}

                Available Data:
                {dataframes_info}

                User's Follow Up Question:
                {question}

                Provide only the updated code that addresses the requirements and fixes any errors. Do not include explanations or comments unless absolutely necessary."""})

        # Add image message if provided
        if image_input:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_input}"}
                }
            )

        # define the messages
        messages = [HumanMessage(content)]

        if manual_mode:
            # Direct invocation of LLM with messages
            response = self.llm_primary_engineer.stream(messages)
            return self._process_streamed_response(response)
        else:
            # Create agent with message-based prompt
            draft_started_code_agent = self.create_agent_wrapper(
                self.llm_primary_engineer,
                [self.dummy_tool],
                system_message=messages
            )
            draft_started_code_node = functools.partial(
                self.agent_node,
                agent=draft_started_code_agent,
                name="Editing Engineer"
            )
            return draft_started_code_node

    def create_follow_up_task_decision_node(self, question: str, previous_question: str, previous_code: str, manual_mode: bool = True):
        """define the follow up task"""

        # create the prompt
        follow_up_task_prompt: str = """
        Analyze if this question is a follow-up to the previous interaction. A question is a follow-up if ANY of these are true:
        1. The question is identical or very similar to the previous question
        2. The question asks for modifications to the previous code or analysis
        3. The question references or builds upon the previous results
        4. The question is asking for more details about the previous analysis
        5. The question is asking to modify visualization aspects of the previous code

        Current Question: {question}
        Previous Question: {previous_question}
        Previous Code: {previous_code}

        Output only YES or NO. No explanation needed.
        """

        if manual_mode:
            # call the llm directly and get the response
            response = self.follow_up_task_llm.invoke(follow_up_task_prompt.format(question=question, previous_question=previous_question, previous_code=previous_code))
            return response.content
        else:
            # create the agent
            follow_up_task_agent = self.create_agent_wrapper(self.follow_up_task_llm, [self.dummy_tool], system_message=follow_up_task_prompt.format(previous_code=previous_code, previous_question=previous_question, question=question))
            follow_up_task_node = functools.partial(self.agent_node, agent=follow_up_task_agent, name="Follow Up Task")

            return follow_up_task_node

    def create_summary_node_modified(self, question: str, output: dict, manual_mode: bool = True) -> str:
        """Explain the model output by providing a short summary of what the final code does."""

        # define the content
        content: list = []

        # add the system message first
        content.append({"type": "text", "text": """You are an AI assistant tasked with summarizing and interpreting the final code and its results that were generated to answer the user's question. Your goal is to provide a clear, concise, and insightful analysis that a non-technical audience can understand."""})

        # add the human message
        content.append({"type": "text", "text": self.system_msg_summary.format(question=question, code=output['code'], code_output=output['code_output'])})

        # define the messages
        messages = [HumanMessage(content)]

        if manual_mode:
            # call the llm directly and get the response
            response = self.hyde_llm.stream(messages)
            return self._process_streamed_response(response)
        else:
            # create the agent
            summary_node_agent = self.create_agent_wrapper(self.hyde_llm,
                                                        [self.dummy_tool],
                                                        system_message=self.system_msg_summary.format(question=question,
                                                                                        code=output['code'],
                                                                                        code_output=output['code_output']))

            summary_node = functools.partial(self.agent_node,
                                            agent=summary_node_agent,
                                            name="Summary Node Agent")

            return summary_node


class AgentConfig(ConstructAgentGraph):
    """class to configure the agent and provide runtime capabilities"""

    def __init__(self, specific_file_name: str = None, stream_url: str = 'http://5.78.113.143:8005/update_stream'):
        super().__init__(stream_url)

        # define the specific file name
        self.specific_file_name = specific_file_name

        # define the chat history
        self.chat_history: list = []

        # save the df info and relevant columns info
        self.dataframes_info: str = ''

        # define the streamed output for the agent
        self.streamed_output: str = ''

    def get_query_auth(self, **kwargs) -> list:
        """get the user id, model id, and chat id"""
        # Get the user_id, model_id, and chat_id
        user_id = kwargs.get('user_id', None)
        #model_id = kwargs.get('model_id', None)
        chat_id = kwargs.get('chat_id', None)

        # Check if any of the required IDs are missing
        if user_id is None:
            raise ValueError("Missing required parameter: user_id")
        if chat_id is None:
            raise ValueError("Missing required parameter: chat_id")

        return [user_id, chat_id]

    # Main execution function
    def invoke(self, query: str):
        # Ensure the coding directory exists
        os.makedirs("coding", exist_ok=True)

        # get the previous chat history
        previous_question, previous_code = self.get_previous_chat_history()

        agent_prompt, code_execution_result, task = self.create_first_draft_code_node(query)

        # create the agentic graph
        graph = self.define_graph(question=query, code_output=code_execution_result, previous_question=previous_question, starter_code=task, display_graph=True)
        print('the graph is defined now')
        self.run_graph(agent_prompt=agent_prompt, graph=graph)
        print('the graph is run now')

        # update the chat history
        self.chat_history.append((query, agent_prompt, code_execution_result))

        # Clean up the chat history
        #cleaned_chat_history = self.clean_chat_history(chat_result)

        # Explain the model output
        #explanation = self._explain_model_output(query, cleaned_chat_history)

        # Cleanup files after processing
        #self.cleanup_files(file_names)

        # Return both the agent histories and the explanation
        #return {
        #    'agent_histories': cleaned_chat_history,
        #    'explanation': explanation
        #}

    def conditional_invoke(self, question: str, **kwargs) -> list:
        """conditionally call the agent without using LangGraph in case of errors with package"""

        # make sure the streamed output is empty
        self.streamed_output = ''

        # check for image input
        image_input = kwargs.get('image_input', '')

        # authenticate the user id and chat id
        user_id, chat_id = self.get_query_auth(**kwargs)

        new_stream_url = f"http://5.78.113.143:8005/update_stream/{user_id}/{chat_id}"
        self.update_stream_url(new_stream_url)

        self.streamed_output += 'Verifying user credentials...'
        self.streamed_output += '---------------------------\n'

        # define the memory of the thought process
        internal_memory: list = []

        # get the last item in the chat history
        previous_question, previous_code = self.get_chat_history_from_management(user_id=user_id, model_id=self.model_id, chat_id=chat_id)
        self.streamed_output += f'Retrieved previous question and code from chat history...'
        self.streamed_output += '---------------------------\n'

        # call the follow up question router function
        follow_decision_response = self.create_follow_up_task_decision_node(question, previous_question, previous_code, manual_mode=True)
        internal_memory.append((question, follow_decision_response))
        self.streamed_output += f'Follow up decision response: {follow_decision_response}'
        self.streamed_output += '---------------------------\n'

        # run the agent router
        router_decision_tool = self.follow_up_task_router({'messages': internal_memory})
        self.streamed_output += f'Follow up task router decision: {router_decision_tool}'
        self.streamed_output += '---------------------------\n'

        # call the question enhancer
        question = self.create_enhanced_question_node(question)
        self.streamed_output += f'Enhanced question: {question}'
        self.streamed_output += '---------------------------\n'

        # do the follow up question routing
        router_decision_tool = 'call_draft_code' # hard coded for now
        if router_decision_tool == 'call_draft_code':
            # call the code draft
            code_generated, self.dataframes_info, file_paths = self.create_first_draft_code_node_modified(question, image_input=image_input)
        elif router_decision_tool == 'call_edit' or not len(self.dataframes_info):
            code_generated = self.create_followup_code_node(question, previous_code, self.dataframes_info, image_input=image_input)
        else:
            raise Exception("404: router error")

        # save the code response in short term memory
        internal_memory.append((question, code_generated))
        # call the code executor
        code_output = self.execute_code(code_generated)
        print(code_output)
        self.streamed_output += f'Code output: \n {code_output}'
        self.streamed_output += '---------------------------\n'
        # save in short term memory the code and its response
        internal_memory.append((code_generated, code_output))

        # call the critic node
        critic_response = self.create_critic_node(question, code_output, code_generated, self.dataframes_info)
        # save the critic response in the short-term memory
        internal_memory.append((code_generated, code_output, critic_response))
        self.streamed_output += f'Critic response: \n {critic_response}'
        self.streamed_output += '---------------------------\n'

        # call the critic router
        critic_router_response = self.critic_router({'messages': internal_memory})
        self.streamed_output += f'Critic router response: \n {critic_router_response}'
        self.streamed_output += '---------------------------\n'

        if critic_router_response == 'continue':
            # define the maximum number of iterations
            n_iter: int = 3
            current_counter: int = 0

            # run the while loop to improve the code generation
            while current_counter < n_iter and critic_router_response == 'continue':
                # get the engineer response
                engineer_code_generated = self.create_code_engineer_node(question, critic_response, code_output, code_generated, self.dataframes_info)
                self.streamed_output += f'Engineer response: \n {engineer_code_generated}'
                self.streamed_output += '---------------------------\n'
                # save the engineer response in the short-term memory
                internal_memory.append((critic_response, engineer_code_generated))

                # call the code executor
                code_output = self.execute_code(engineer_code_generated)
                # save in short term memory the code and its response
                internal_memory.append((code_generated, code_output))

                # call the critic again
                critic_response = self.create_critic_node(question, code_output, engineer_code_generated, self.dataframes_info)
                self.streamed_output += f'Critic response: \n {critic_response}'
                self.streamed_output += '---------------------------\n'
                # save the critic response in the short-term memory
                internal_memory.append((code_generated, code_output, critic_response))

                # overwrite the code_response with the engineer_code_response
                code_generated = engineer_code_generated

                # get the critic router response
                critic_router_response = self.critic_router({'messages': internal_memory})
                self.streamed_output += f'Critic router response: \n {critic_router_response}'
                self.streamed_output += '---------------------------\n'

                # update the counter
                current_counter += 1

            # define the summary node
            summary = self.create_summary_node_modified(question, {'code_output': code_output, 'code': code_generated})
            self.streamed_output += f'Summary: \n {summary}'
            self.streamed_output += '---------------------------\n'
        elif critic_router_response == 'summary_exit':
            summary = self.create_summary_node_modified(question, {'code_output': code_output, 'code': code_generated})
            self.streamed_output += f'Summary: \n {summary}'
            self.streamed_output += '---------------------------\n'
        else:
            raise Exception("404: critic router error")

        print('\n')
        print(code_output)
        print('\n')

        # remove duplicates from the csv files
        self.remove_duplicates_from_csvs(self.data_loader.download_path_model.charts_path)

        # clean the coding directory
        self.clean_coding_directory(file_paths)

        # save the chat history
        self.add_chat_history_to_management(user_id=user_id, model_id=self.model_id, chat_id=chat_id, question_answer_pair=(question, code_generated))

        # get the byte strings of the html files in the coding directory
        html_byte_strings = self.html_to_byte_strings_in_directory(self.data_loader.download_path_model.coding_path)

        # move the html files from the coding directory to the charts directory
        self.move_html_files_from_charts_to_reserves()

        return summary, html_byte_strings


agent_obj = AgentConfig()
user_id = 1
chat_id = 1

# Create an instance of MyStreamHandler
#stream_handler = MyStreamHandler()

# Set user_id and chat_id before starting
#stream_handler.set_user_and_chat(user_id=user_id, chat_id=chat_id)

#agent_obj.specialized_file_management('Please pull up the building information for a specific address?')
#agent_obj._create_prompt('Find out which agent had the most business in the last year?')
#agent_obj._create_prompt('find me the average minutes that were spent during a session? use /content/coding/kinkinterestpublic.csv')
#agent_obj.create_first_draft_code_node_modified('find out which agent had the most business in the last year?')
#starter_code, file_paths, agent_prompt = agent_obj._create_prompt('get me the average of the age column')
#task, file_names, agent_prompt, task_breakdown = agent_obj._create_prompt('get me the average of the age column')
summary, chart = agent_obj.conditional_invoke('find out which agent had the most business in the last year?', user_id=user_id, chat_id=chat_id)

