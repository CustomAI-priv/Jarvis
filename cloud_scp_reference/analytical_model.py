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
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker

try:
  from backend.setup import DataLoader, api_key, claude_api_key
  # from settings import Settings, CodingModelSpecs
  # from utils import format_file_name, get_file_extension
  from backend.text_model import ModelSpecifications, LanceTableRetriever, LanceRetriever
  from backend.chat_history_management import ChatHistoryManagement
except:
    print("using alternate import path")
    try:
        from setup import DataLoader, api_key, claude_api_key
        # from settings import Settings, CodingModelSpec
        # from utils import format_file_name, get_file_extension
        from text_model import ModelSpecifications, LanceTableRetriever, LanceRetriever
        from chat_history_management import ChatHistoryManagement
    except ImportError:
        pass
    pass

import autogen
from concurrent.futures import ThreadPoolExecutor, as_completed

import io
from datetime import datetime
from pathlib import Path
from typing import Iterable
from PIL import Image
import imgkit
import aide

from openai import AzureOpenAI
from openai.types import FileObject
from openai.types.beta.threads import Message, TextContentBlock, ImageFileContentBlock


from langchain.callbacks.base import BaseCallbackHandler
import requests


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


# restate the environment variables and api keys
os.environ['ANTHROPIC_API_KEY'] = claude_api_key
os.environ["OPENAI_API_KEY"] = api_key


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


class AgentLLMConfig(BaseModel):
    """define the language models to use for the agents"""

    # define the variables for primary engineer LLM
    llm_primary_engineer_temperature: float = 0
    llm_primary_engineer_model: str = 'gpt-4o'
    llm_primary_engineer_max_tokens: int = 1000

    # define the variables for the planner LLM
    llm_planner_temperature: float = 0
    llm_planner_model: str = 'gpt-4o'
    #llm_planner_max_tokens: int = 3000

    # define llm for the follow up task
    llm_follow_up_task_model: str = 'gpt-4o-mini'
    llm_follow_up_task_temperature: float = 0
    llm_follow_up_task_max_tokens: int = 50

    # path to save the charts 
    chart_save_path: str = '/root/BizEval/coding/'

class AnalyticalModelRetrieverProcessing():
    """all the retriever functionality for the analytical model"""

    # set the json mode to true
    json_mode: bool = False # True
    predownloaded_mode: bool = True

    def __init__(self, stream_url: str = 'http://127.0.0.1:8005/stream'):
        # create directory called coding and charting
        os.makedirs("coding", exist_ok=True)

        # set the stream url
        self.stream_url = stream_url

        # define the data loader 
        self.data_loader = DataLoader()

        # Set k to 2 for retrieving top 2 documents
        self.k = self.data_loader.settings.top_k_csv_files
    
        # initialize the agent llm config
        self.agent_llm_config = AgentLLMConfig()

        # create the lancedb retriever
        self.table_descriptions_retriever = self._create_table_descriptions_retriever()
        
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

    def _define_llms_arr(self):
        """define the llms for the analytical model"""

        # create the llm for the csv analysis
        llm = ChatOpenAI(model_name=self.data_loader.settings.csv_analysis_model,
                         temperature=self.data_loader.settings.summary_temp,
                         callbacks=[self.stream_handler])

        # define llm for the primary analysis
        llm_primary_engineer = ChatOpenAI(model_name=self.agent_llm_config.llm_primary_engineer_model,
                                               temperature=self.agent_llm_config.llm_primary_engineer_temperature,
                                               max_tokens=self.agent_llm_config.llm_primary_engineer_max_tokens,
                                               callbacks=[self.stream_handler])

        # check if the column header model is gpt-4o-mini so that we can use the same llm for the csv analysis and the column header model
        if self.data_loader.settings.csv_analysis_model == 'gpt-4o-mini':
            column_header_llm = llm
        else:
            # Initialize the LangChain ChatOpenAI model
            column_header_llm = ChatOpenAI(model_name='gpt-4o-mini',
                                           temperature=0,
                                           openai_api_key=api_key,
                                           callbacks=[self.stream_handler])

        # define the planner llm as different model because of max tokens settings
        planner_llm = ChatOpenAI(model_name=self.agent_llm_config.llm_planner_model,
                                temperature=self.agent_llm_config.llm_planner_temperature,
                                callbacks=[self.stream_handler])

        # define the follow up task llm
        follow_up_task_llm = ChatOpenAI(model_name=self.agent_llm_config.llm_follow_up_task_model,
                                        temperature=self.agent_llm_config.llm_follow_up_task_temperature,
                                        max_tokens=self.agent_llm_config.llm_follow_up_task_max_tokens,
                                        callbacks=[self.stream_handler])

        return llm, llm_primary_engineer, column_header_llm, planner_llm, follow_up_task_llm

    def update_stream_url(self, new_url: str) -> None:
        """Update the stream URL and reinitialize LLMs with new stream handler"""
        self.stream_url = new_url
        self.stream_handler.update_url(new_url)
        #self._define_llms_arr() 

    def _create_table_descriptions_retriever(self):
        """Create the retriever for the LanceDB database table descriptions"""

        # Define the reranker
        reranker = LinearCombinationReranker(
            weight=self.data_loader.settings.hybrid_search_ratio,
        )

        # Initialize the custom retriever object for the TABLE DESCRIPTIONS retriever
        table_retriever = LanceTableRetriever(
            table=self.data_loader.lancedb_client[3],
            reranker=reranker,
            k=self.data_loader.settings.k_top * self.data_loader.settings.k_tab_multiplier,  # Ensure k is set
            mode='fts'  # Ensure mode is set
        )

        return table_retriever

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
                file_path = os.path.join(self.data_loader.download_path_model.coding_path, file_name)
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

    # return pandas dataframes as a list of dictionaries
    def fetch_relevant_tables(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Fetch the relevant tables for the query"""

        # Use the custom table descriptions retriever and select to always use tabular data (True returns an empty list)
        relevant_docs = self.table_descriptions_retriever.get_relevant_documents(query, disable_tabular=False)

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
        print(self.streamed_response)
        return results

    def _define_charting_prompt(self, file_info, user_question: str) -> str:
        """define the charting prompt for the agent"""

        file_details = "\n".join([
            f"- File: {info['file_name']}"
            f"\n  Original headers: {', '.join(info['original_headers'])}"
            f"\n  Renamed headers: {', '.join(info['renamed_headers'].values())}"
            for info in file_info
        ])

        prompt = f"""
            Write a Python script to visualize data from the following CSV file(s):
            {file_details}

            Address the user's question: "{user_question}"

            Execute the script to generate an appropriate graph. Follow these guidelines:

            1. Use the following libraries: matplotlib with aquarel.
            2. Implement the 'artic_dark' theme from aquarel for a visually appealing graph.
            3. Read the CSV file(s) and determine the most suitable graph type based on both the data and the user's question.
            4. Create a clear and informative graph with proper labels, titles, and a legend (if applicable).
            5. Use the renamed headers for labels and legends, but refer to the original headers when reading the CSV file(s).
            6. Include error handling for file reading and data processing.
            7. Save the graph as a high-resolution PNG file with a descriptive name.

            Ensure the code is well-commented and follows Python best practices. After generating the graph, display it and explain how the visualization answers the user's question.
            """

        return prompt

    def _define_data_analysis_prompt(self, file_info, user_question: str) -> str:
        """Define the data analysis prompt for the agent tailored for autogen agent."""

        file_details = "\n".join([
            f"- File: {info['file_name']}"
            f"\n  Original headers: {', '.join(info['original_headers']) if info['original_headers'] else 'None'}"
            f"\n  Renamed headers: {', '.join(info['renamed_headers'].values()) if info['renamed_headers'] else 'None'}"
            for info in file_info
        ])

        prompt = f"""
            You are an expert data analyst and Python programmer acting as an autonomous agent.

            Your task is to write a **complete and executable Python script** to analyze data from the following CSV file(s):
            {file_details}

            **User's Question:**
            "{user_question}"

            **Instructions:**

            1. **Read the CSV file(s) using relative paths** within the `'coding'` directory.
            2. Use appropriate Python libraries for data manipulation and analysis (e.g., `pandas`, `numpy`).
            3. For visualizations, you may use `matplotlib` with `aquarel`, `plotly`, `seaborn`, or other suitable libraries.
            4. **Determine the most suitable output format** (tables, charts, computations) based on the data and the user's question.
            5. Include **error handling** for file reading and data processing.
            6. **Save all generated outputs** (graphs, tables, etc.) **within the `'coding'` directory** using descriptive filenames.
            7. Use the **renamed headers for any labels and legends**, but refer to the original headers when reading the CSV file(s).
            8. Write **clear and concise code with comments** explaining each step.
            9. **Do not include any code that requires user interaction or input**; the script should run autonomously.
            10. After **running the script**, provide a **summary of the findings** and explain how they answer the user's question within the code comments or as printed output.
            11. Ensure your code **does not access external resources or require internet connectivity**.
            12. All file paths should be relative and point to files within the `'coding'` directory.

            **Important Notes:**

            - **Ensure your code is fully executable and follows Python best practices**.
            - **Do not include any placeholder code**; provide the full implementation.
            - The code should be suitable for execution by an autogen agent with no modifications.
            - **Output all results within the script**; do not produce any separate explanatory text.

            **Output Format:**

            - Provide the **complete Python script** in a **single code block**.
            - Do not include any additional text or explanations outside the code block.
            - Start the code block with ```python and do not specify a file path.

            **Example Code Block Format:**

            \`\`\`python
            # Your Python script here
            \`\`\`

            Ensure your script meets all the above requirements and is ready to be executed as is.
            """

        return prompt

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

    def facade_chart_prompt(self, user_question: str) -> str:
        """Facade method to retrieve relevant tables and create the charting prompt"""

        # Step 1: Fetch relevant tables
        relevant_tables = self.fetch_relevant_tables(user_question)

        # Step 2: Generate the charting prompt
        charting_prompt = self._define_charting_prompt(relevant_tables, user_question)

        return charting_prompt

    def facade(self, user_question: str, context_mode: bool = True) -> str:
        """Facade method to retrieve relevant tables and create the data analysis prompt"""

        # If context mode is true, create the user queries
        if context_mode:
            # Create the user queries
            user_queries = self.create_user_queries(self.chat_history)
        else:
            # If context mode is false, use the user question
            user_queries = user_question

        # Fetch relevant tables based on the user queries
        relevant_tables = self.fetch_relevant_tables(user_queries)

        # Generate the data analysis prompt using the new method
        data_analysis_prompt = self._define_data_analysis_prompt(relevant_tables, user_queries)

        return data_analysis_prompt

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

    def move_html_files_from_coding_to_plots(self):
        """move the html files from the coding directory to the plots directory"""

        # get the list of all html files in the coding directory
        html_files = [f for f in os.listdir(self.data_loader.download_path_model.coding_path) if f.endswith('.html')]

        # move the html files to the plots directory
        for html_file in html_files:
            shutil.move(f'{self.data_loader.download_path_model.coding_path}/{html_file}', f'{self.data_loader.download_path_model.charts_path}/')

        return html_files


class AgentUtilities(AnalyticalModelRetrieverProcessing):
    """class to configure the agent and provide runtime capabilities"""

    def __init__(self, stream_url: str):
        super().__init__(stream_url)

        # define the system messages and engineer planner prompts
        self.system_msg_engineer, self.system_msg_planner = self._create_engineer_planner_prompts()

        # define the model selection
        self.config_list_gpt4o, self.config_list_gpt4o_mini, self.config_list_claude = self._llm_model_selection()

        # define the structured output
        self.code_output_model: CodeOutput = CodeOutput

        # number of columns after filtering
        self.number_columns_after_filtering = 15

        # select the language model from the configuration list for autogen usage (different formatting)
        self.selected_analytics_model = self.config_list_gpt4o[0]

        # path to save the file 
        self.chart_save_path = self.agent_llm_config.chart_save_path

        # define the chat history management
        self.chat_history_management = ChatHistoryManagement()

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
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def count_tokens(self, planner_template):
        token_count = self.num_tokens_from_string(str(planner_template), "cl100k_base")  # Use appropriate encoding
        print(f"Number of tokens: {token_count}")

    def _create_agents(self, question: str):
        """create the agents for the group chat"""

        # Create user proxy agent
        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin",
            code_execution_config=False,
            human_input_mode="NEVER",
            default_auto_reply="",
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        )

        # Create engineer agent
        engineer = autogen.AssistantAgent(
            name="Engineer",
            llm_config=self.selected_analytics_model,
            system_message='You are the engineer. Your job is to make edits to the code. Fix all the errors that the critic suggests. Implement the suggestions of the critic but do not change the rest of the code.',
        )

        # Create code executor agent
        code_executor = autogen.UserProxyAgent(
            name="Executor",
            # system_message="You are the Executor. Execute the code written by the engineer and report the result.",
            system_message="You are the Executor. Execute the code written by the engineer and report the result.",
            human_input_mode="NEVER",
            code_execution_config={
              "last_n_messages": 2,
              "work_dir": "coding",
              "use_docker": False
            },
        )

        # Create critic agent
        critic = autogen.AssistantAgent(
            name="Critic",
            system_message=f"""
            You are the Critic. Your task is to evaluate the code based on two criteria:
            1. Does the code run without any errors?
            2. Does the code directly address the user's question: "{question}"?

            If both criteria are met, respond with "TERMINATE".
            If either criterion is not met, respond with 1-2 sentences of what to change. If there are errors, restate the error message.
            Provide no other commentary or explanation.
            """,
            llm_config=self.selected_analytics_model,
        )

        # create the group chat
        groupchat = autogen.GroupChat(
            agents=[code_executor, critic, user_proxy, engineer],
            messages=[],
            max_round=2,
            speaker_selection_method="round_robin",
        )

        # create the manager
        manager = autogen.GroupChatManager(
            groupchat=groupchat,
            llm_config=self.selected_analytics_model
        )

        return manager, user_proxy

    @staticmethod
    def temp_file_cleanup(temp_dir: str):
        """clean up the temporary directory"""
        temp_dir.cleanup()

    @staticmethod
    def _create_engineer_planner_prompts():
        """create the engineer planner prompts"""

        # System messages
        system_msg_engineer = """
        Engineer. You write python/bash to retrieve relevant information. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
        Do not include code comments. Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
        If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
        Make sure to use plotly and make the graphs look very good with lots of detail and a modern template. Make sure to download the file after you are done in the /coding directory.
        """

        system_msg_planner = """
        You are the Planner. Your role is to decompose the user's question into specific subtasks and provide clear instructions for the coding agent to execute.

        **Instructions:**

        1. Restate the user's question.
        2. Break down the task into 3-5 subtasks.
        3. For each subtask, provide explicit instructions, including:
           - Data retrieval and cleaning steps
           - Calculations to be performed
           - Visualizations to be created (if applicable)
           - File paths for saving outputs
        4. Specify that all calculation results should be printed to the console.
        5. If visualizations are required, instruct to save all plots as HTML files.
        6. For complex questions, ensure all important numbers are printed and all charts are saved.

        **Output Format:**

        Question: [Restate the user's question]

        Subtasks:
        1. [Subtask 1]
           - Instructions: [Detailed steps]
           - Output: [Specify console output or file path]

        2. [Subtask 2]
           - Instructions: [Detailed steps]
           - Output: [Specify console output or file path]

        3. [Subtask 3]
           - Instructions: [Detailed steps]
           - Output: [Specify console output or file path]

        [Add more subtasks as needed]

        Additional Notes:
        - Use Plotly for all visualizations with modern templates.
        - Ensure proper labeling, titles, and legends for all charts.
        - Save all data files in the 'data/' directory.
        - Save all plot files in the 'plots/' directory.

        Your goal is to provide a clear, step-by-step plan that the coding agent can follow to effectively answer the user's question.
        """
        return system_msg_engineer, system_msg_planner

    @staticmethod
    def _llm_model_selection():
        """select the llm model"""
        config_list_gpt4 = [
            {
                "model": "gpt-4o",
                "api_key": api_key,
                "api_type": "openai",
            }
        ]
        config_list_gpt4_mini = [
            {
                "model": "gpt-4o-mini",
                "api_key": api_key,
                "api_type": "openai",
            }
        ]
        config_list_claude = [
            {
                "model": "claude-2",
                "api_key": claude_api_key,
                "api_type": "anthropic",
                "temperature": 0.1,
            }
        ]
        return config_list_gpt4, config_list_gpt4_mini, config_list_claude

    def define_task_template(self):
        """Create the task template for the planner agent, supporting multiple dataframes"""

        # define the types of charts that can be created
        plotly_charts = {
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

        # Define the task template
        planner_template = """
        You are the Planner. Your role is to decompose the user's question into specific subtasks and provide clear instructions for the coding agent to execute.

        **Instructions:**

        1. Restate the user's question.
        2. Break down the task into 4-6 subtasks, including data cleaning as a specific task.
        3. For each subtask, provide explicit instructions, including:
        - Data retrieval steps
        - Data cleaning and preprocessing steps
        - Articulate what to do with NaN values. Sometime we can drop them, sometimes we can fill them with the mean, median, or mode. You must make this decision based on the data provided and user question.
        - Calculations to be performed
        - Visualizations to be created (if applicable)
        - File paths for saving outputs
        4. Specify that all calculation results should be printed to the console.
        5. If visualizations are required, instruct to save all plots as HTML files.
        6. When you choose a chart, make sure you pay attention to the type of data you are working with. You need to make sure that when you plot the data it will look good on the chart and be informative.

        **Output Format:**

        Question: {question}

        Available Dataframes:
        {dataframes_info}

        Relevant Columns Across All Dataframes:
        {relevant_columns}

        Subtasks:
        1. Data Retrieval and Integration (if multiple dataframes)
        - Instructions: [Detailed steps for retrieving and potentially merging the necessary data]
        - Output: [Specify console output]

        2. Data Cleaning and Preprocessing
        - Instructions:
          - Perform all cleaning operations in-memory without saving any new files
          - Handle NaN values appropriately (e.g., drop, fill with mean/median/mode, or use a specific value)
          - Convert data types if necessary
          - Remove duplicates if any
          - Handle outliers if present
          - [Any other specific cleaning steps based on the data]
        - Output: Print a summary of cleaning actions taken to the console

        3. [Subtask 3]
        - Instructions: [Detailed steps]
        - Output: [Specify console output or file path for visualizations only]

        4. [Subtask 4]
        - Instructions: [Detailed steps]
        - Output: [Specify console output or file path for visualizations only]

        [Add more subtasks as needed]

        Additional Notes:
        - Use Plotly for all visualizations with modern templates.
        - Create two Plotly charts for the task, one of which can be a Plotly table. You can choose from the options below:
        {plotly_charts}
        - Ensure proper labeling, titles, and legends for all charts.
        - Do not save any intermediate data files. All data processing should be done in-memory.
        - Save only the final visualization plots as HTML files in the {output_directory} directory.

        Your goal is to provide a clear, step-by-step plan that the coding agent can follow to effectively answer the user's question, while ensuring all data cleaning and processing is done in-memory without creating new files.
        """
        return planner_template, plotly_charts

    def call_llm(self, df, question):
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

    def get_relevant_columns_with_lm_singular(self, question, file_path: str = None):
        # read the pandas dataframe from the file
        df = pd.read_csv(file_path)

        # add a condition if the dataframe is smaller than 50 columns then just use the original dataframe
        if df.shape[1] < 50:
            relevant_columns = df.columns.tolist()
            return relevant_columns, df, df.head(5), self._get_dataframe_info(df)

        # get the relevant columns
        relevant_columns, columns = self.call_llm(df, question)
        self.streamed_response += "Identified the following relevant columns: " + str(relevant_columns) + "\n"
        self.streamed_response +=  "-----------------------------------------\n"
        print(self.streamed_response)

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
        self.save_dataframe(subset_df, filename=file_path) # specify a file name in general

        # generate the dataframe info and description
        return valid_columns, subset_df, subset_df.head(10), str(subset_df.describe()) #self._get_dataframe_info(subset_df)

    def get_relevant_columns_with_lm(self, question, file_names: list):
        """get the relevant columns with the language model for multiple files"""

        # implement concurrent.futures to call the function for each file name
        with ThreadPoolExecutor(max_workers=len(file_names)) as executor:
            # map the function to the file names
            results = list(executor.map(self.get_relevant_columns_with_lm_singular, [question] * len(file_names), file_names))

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

    def _modify_file_paths_aide_code(self, code: str) -> str:
        """
        Modify the file paths in the aide code to use the correct path.

        Args:
            code (str): The original code string from AIDE.

        Returns:
            str: The modified code string with updated file paths.
        """
        # Define the regex pattern to match file paths
        pattern = r'(?:\.\/working\/|\.\/|\/working\/)([^\'"\s]+\.[^\'"\s]+)'

        # Define the replacement function
        def replace_path(match):
            filename = match.group(1)
            return f"'{self.data_loader.download_path_model.coding_path}/{filename}'"

        # Use regex to replace all matching file paths
        modified_code = re.sub(pattern, replace_path, code)

        return modified_code

    # Function to display images
    def display_images_from_directory(self, directory):
        # This function displays all images in the specified directory
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Full path to the image file
                img_path = os.path.join(directory, filename)
                # Display the image
                print(f"Displaying image: {img_path}")
                display(Image(filename=img_path))

    def get_agent_history(self, agent):
        """
        Retrieve the chat history for a specific agent.

        Args:
            agent (ConversableAgent): The agent whose chat history we want to retrieve.

        Returns:
            List[Dict]: A list of message dictionaries representing the chat history.
        """
        if not hasattr(agent, 'chat_messages'):
            return []

        # Create a copy of the chat messages to avoid modifying the original
        messages = agent.chat_messages

        # Flatten the nested dictionaries if necessary
        flattened_messages = []
        for recipient, message_list in messages.items():
            for message in message_list:
                flattened_message = {
                    'sender': agent.name,
                    'recipient': recipient,
                    'content': message['content'],
                    'role': message['role']
                }
                flattened_messages.append(flattened_message)

        return flattened_messages

    @staticmethod
    def cleanup_files(file_names: list):
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
            full_path = os.path.join('coding', file_name)

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

    def get_agent_history_from_history(self, agent: ConversableAgent, chat_history: List[Dict]) -> List[Dict]:
        """
        Retrieve the chat messages specific to a given agent from the provided chat history.

        This method filters the chat history to include only messages from the specified agent,
        excluding any empty or whitespace-only messages.

        Args:
            agent (ConversableAgent): The agent whose messages we want to extract.
            chat_history (List[Dict]): The cleaned chat history, containing messages from all agents.

        Returns:
            List[Dict]: A list of non-empty messages sent by the specified agent.
        """
        agent_messages = []  # Initialize an empty list to store the agent's messages

        # Iterate through each message in the chat history
        for message in chat_history:
            # Check if the message is from the specified agent
            if message.get('name') == agent.name:
                # Extract the content of the message, defaulting to an empty string if not present
                content = message.get('content', '').strip()

                # Only include non-empty messages
                if content:
                    agent_messages.append(message)

        # Return the list of messages from the specified agent
        return agent_messages


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
    run_task_decomposition: bool = True

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
    
    def _define_initial_code_generation_prompt(self, question: str, dataframes_info: str, relevant_columns: List[str]):
        """define the initial code generation prompt"""

        # define the types of charts that can be created
        plotly_charts = {
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

        # define the simplified task prompt if task decomposition is not run
        simplified_planner_template = f"""
            You are the Planner. Decompose the user's question into subtasks and provide instructions for execution.

            Question: {question}

            Dataframes Info: {dataframes_info}

            Relevant Columns: {relevant_columns}

            Choose the correct chart type from the following options:
            {plotly_charts}

            Instructions:
            1. Retrieve and integrate data.
            2. Clean data of NaN values, deal with outliers effectively, and handle missing values.
            3. Perform calculations and print all numerical results.
            4. Create Plotly visualizations with a modern template.
            5. Save all charts as HTML files in /root/BizEval/coding/.
            6. When you choose a chart, make sure you pay attention to the type of data you are working with. You need to make sure that when you plot the data it will look good on the chart and be informative.

            Use Plotly for all visualizations. Ensure proper labeling and legends. Read this prompt twice before proceeding.
            """
        return simplified_planner_template

    @staticmethod
    def display_graph(graph):
        """
        Display the agent graph as a Mermaid diagram.

        This method attempts to visualize the graph structure using Mermaid.
        It's designed to work in Jupyter notebook environments.

        Args:
            graph: The graph object to be displayed. Expected to have a get_graph() method
                   that returns an object with a draw_mermaid_png() method.
        """
        try:
            # Generate a PNG image of the Mermaid diagram
            mermaid_png = graph.get_graph().draw_mermaid_png()

            # Display the PNG image in the notebook
            # This requires IPython's display and Image functions to be available
            display(Image(mermaid_png))
        except Exception as e:
            # Catch any exceptions that occur during the process
            # This could include AttributeError if the graph methods are missing,
            # or other errors related to image generation or display
            print(f"Error displaying graph: {e}")

    @staticmethod
    def router(state):
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            # The previous agent is invoking a tool
            return "call_tool"
        if "TERMINATE" in last_message.content:
            # Any agent decided the work is done
            return END
        return "continue"

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
       
        # Prepare dataframes_info
        dataframes_info = ""
        for file_path, df_info, df_head in zip(file_paths, dataframes_infos, dataframes_heads):
            dataframes_info += f"File Path: {file_path}\n"
            dataframes_info += f"Dataframe Info:\n{df_info}\n\n"
            dataframes_info += f"Dataframe Head:\n{df_head.to_string()}\n\n"

        # Prepare relevant_columns_info
        relevant_columns_info = "\n".join([f"{file}: {', '.join(cols)}" for file, cols in zip(file_paths, relevant_columns)])

        # define the content messages container 
        content: list = []

        # Get the planner template
        planner_template, plotly_charts = self.define_task_template()

        # fill in the planner template 
        planner_template = planner_template.format(question=question, dataframes_info=dataframes_info, relevant_columns=relevant_columns_info, plotly_charts=plotly_charts, output_directory=self.chart_save_path)
        content.append({"type": "text", "text": planner_template})
        
        # define the human message
        messages = [HumanMessage(content)]

        print('input token count task decomposition:')
        self.count_tokens(planner_template)

        # Call the language model to generate the response
        planner_response = self.llm_primary_engineer.stream(messages)

        return self._process_streamed_response(planner_response), dataframes_info, relevant_columns_info

    def create_first_draft_code_node(self, question: str):
        """create the first draft code agent"""

        # Define the task
        task, file_names, agent_prompt, task_breakdown = self._create_prompt(question)

        # execute the code
        code_execution_result = self.execute_code(agent_prompt)
        return agent_prompt, code_execution_result, task

    def create_first_draft_code_node_modified(self, question: str, manual_mode: bool = True, image_input: str = ''):
        """create the first draft code agent"""
        starter_code, file_paths, agent_prompt, task, dataframes_info, relevant_columns_info = self._create_prompt(question, image_input)
        
        if manual_mode: 
            # call the llm directly and get the response
            return agent_prompt, dataframes_info
        else: 
            first_draft_agent = self.create_agent_wrapper(self.llm_primary_engineer,
                                                        [self.execute_code],
                                                        system_message=agent_prompt)

            first_draft_node = functools.partial(self.agent_node, agent=first_draft_agent, name="First Draft Agent")

            return first_draft_node

    def _create_prompt(self, query: str, image_input: str = ''):
        """create the LLM prompt - identify the correct file as well"""

        # get the correct file to use
        start_time = time.time()
        _, file_paths = self.specialized_file_management(query, file_name=self.specific_file_name)
        print('file paths:', file_paths)
        end_time = time.time()
        print(f"Time taken to fetch file paths from inside create_prompt: {end_time - start_time} seconds")

        # extract the relevant columns from the
        relevant_columns, _, subset_df_head_list, subset_df_info_list = self.get_relevant_columns_with_lm(query, file_names=file_paths)

        # Call _create_task_decomposition_call with correct arguments
        if self.run_task_decomposition:
            start_time = time.time()
            task, dataframes_info, relevant_columns_info = self._create_task_decomposition_call(
                question=query,
                dataframes_infos=subset_df_info_list,
                dataframes_heads=subset_df_head_list,
                relevant_columns=relevant_columns,  # This should be a list of lists, not a dict
                file_paths=file_paths
            )
            end_time = time.time()
            print(f"Time taken to create task: {end_time - start_time} seconds")
        else: 
            task = self._define_initial_code_generation_prompt(query, dataframes_info, relevant_columns)
            task = task.format(question=query, dataframes_info=dataframes_info, relevant_columns=relevant_columns_info)

        # call the starter code function
        start_time = time.time()
        starter_code = self._generate_starter_code_llm(task, image_input) # generate the code using the AIDE library: self._generate_starter_code_aide(prompt)
        end_time = time.time()
        print(f"Time taken to generate starter code: {end_time - start_time} seconds")

        # define the prompt to use
        agent_prompt = f"""
        User Question: {query}

        File(s) used: {', '.join(file_paths)}

        Starter Code in Python:
        {starter_code}
        """

        return starter_code, file_paths, agent_prompt, task, dataframes_info, relevant_columns_info

    def _generate_starter_code_llm(self, prompt: str, image_input: str = None):
        """
        Generate the starter code using the LLM with support for text and image inputs.
        
        Args:
            prompt (str): The text prompt for code generation
            image_input (str, optional): Base64 encoded image string
        """

        # define the content of the prompt
        content: list = []

        try:

            # Create the system message
            system_message = f"""You are a Python coding assistant. Follow this task.
            Generate code based on the given prompt and image if provided.
            Ensure the code is executable and syntax is correct.
            Make sure to use the same file paths from the task when importing data.
            Ensure any code you provide can be executed with all required imports and variables defined.
            Do not include code comments.
            Return only the python code at the end. No additional text or comments."""


            # add the system message to the content  
            content.append({"type": "text", "text": system_message})

            # add the prompt to the content  
            content.append({"type": "text", "text": prompt})

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

        except Exception as e:
            print(f"Error generating code: {str(e)}")
            return None

    def run_graph(self, agent_prompt, graph):
        """
        Run the constructed agent graph with the given prompt.

        Args:
            agent_prompt (str): The user's query or instruction to be processed by the graph.

        This method streams the events from the graph execution and prints each step.
        """
        # # Ensure the graph is compiled before running
        # if not hasattr(self, 'compiled_graph'):
        #     raise AttributeError("Graph has not been compiled. Call define_graph() first.")

        # Create the initial state with the user's prompt
        initial_state = {
            "messages": [
                HumanMessage(content=agent_prompt)
            ],
        }
        print('initial state is created: ', initial_state)
        # Stream the events from the graph execution
        events = graph.stream(
            initial_state,
            # Maximum number of steps to take in the graph
            {"recursion_limit": 150},
        )

        # Print each event in the stream
        print("\n\n")
        for event in events:
            print(event)
            print("----")

        # Note: You might want to return something here, like a summary or final state

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

    def _update_state_variables(self, question: str, code_output: str, critic_feedback: str, current_code: str) -> None:
        """update the state variables"""

        # update the state variables
        self.state_variables['code_execution_output'] = code_output
        self.state_variables['critic_feedback'] = critic_feedback
        self.state_variables['current_code'] = current_code
        self.state_variables['question'] = question
        return

    # Helper function to create a node for a given agent
    def agent_node(self, state, agent, name):
        result = agent.invoke(state)
        # We convert the agent output into a format that is suitable to append to the global state
        if isinstance(result, ToolMessage):
            pass
        else:
            result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
        return {
            "messages": [result],
            # Since we have a strict workflow, we can
            # track the sender so we know who to pass to next.
            "sender": name,
        }

    def create_agent_wrapper(self,llm, tools, system_message: str):
        """Create an agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with TERMINATE so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        print("tools\n", tools)
        # prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

        prompt = prompt.partial(tool_names=", ".join([getattr(tool, 'name', f"Tool{i}") for i, tool in enumerate(tools)]))

        # prompt = prompt.partial(tool_names="execute_code")

        # prompt = prompt.partial(tool_names="")
        return prompt | llm.bind_tools(tools)

    def _parse_code_execution_response(self, response: str):
        """parse the code execution response"""

        # Parse the response
        # content = response.get('content', '')
        content = response
        lines = content.split('\n')

        execution_status = ''
        code_output = ''
        error_message = ''

        for line in lines:
            if line.startswith('exitcode:'):
                execution_status = 'success' if 'execution succeeded' in line else 'failed'
            elif line.startswith('Code output:'):
                code_output = '\n'.join(lines[lines.index(line)+1:])
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

    @tool
    def code_executor_tool(self, starter_code: Annotated[str, "The starter code to execute"]):
        """tool to execute code"""
        return self.execute_code(starter_code)

    def dummy_tool(self):
        """dummy tool"""
        return None

    def create_critic_node(self, question: str, code_output: Annotated[str, "The code output to critique"], starter_code: Annotated[str, "The starter code to execute"], dataframes_info: Annotated[str, "The dataframes info"], manual_mode: bool = True):
        """create the critic agent"""

        critic_prompt: str = """
            You are the Critic. Your task is to evaluate the code based on two criteria:
            1. Does the code run without any errors?
            2. Are the charts in the code informative and useful? Do the numerical outputs make sense?
            3. Does the code directly address the user's question: "{question}"?
            4. Are all the charts and tables being saved to the output directory /root/BizEval/coding/

            For reference to the data, here is the dataframes info used to create the code:
            {dataframes_info}

            If all criteria are met, respond with "TERMINATE". You ignore warnings generated by the code. 
            If either criterion is not met, respond with 1-2 sentences of what to change. If there are errors, restate the error message.
            Provide no other commentary or explanation.

            Here is the code output:
            {code_output}

            Here is the code:
            {starter_code}
        """

        if manual_mode: 
            # define the content 
            content: list = []
            content.append({"type": "text", "text": critic_prompt.format(question=question, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info)})

            # define the messages
            messages = [HumanMessage(content)]

            # call the llm directly and get the response
            response = self.llm_primary_engineer.stream(messages)
            return self._process_streamed_response(response)
        else: 
            # create the agent
            critic_agent = self.create_agent_wrapper(self.llm_primary_engineer, [self.dummy_tool], system_message=critic_prompt.format(question=question, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info))
            critic_node = functools.partial(self.agent_node, agent=critic_agent, name="Critic")

            return critic_node

    def create_code_engineer_node(self, question: str,
                                   critic_feedback: Annotated[str, "The feedback from the critic"],
                                   code_output: Annotated[str, "The code output to critique"],
                                   starter_code: Annotated[str, "The starter code to execute"], 
                                   dataframes_info: Annotated[str, "The dataframes info"],
                                   manual_mode: bool = True):
        """create the code engineer agent"""

        # create the prompt
        code_engineer_prompt: str = """
        You are a Python Code Engineer. Your task is to fix bugs and implement changes suggested by the Critic. Focus on:

        1. Addressing the Critic's feedback directly.
        2. Fixing any errors or bugs in the code.
        3. Ensuring the code runs without errors and answers the user's question.

        Original Code:
        {starter_code}

        Critic's Feedback:
        {critic_feedback}

        Here is the info on the dataframes used to create the code:
        {dataframes_info}

        Code Output in Console:
        {code_output}

        User's Original Question:
        {question}

        Provide only the updated code that addresses the Critic's feedback and fixes any errors. Do not include explanations or comments unless absolutely necessary for understanding a complex change.

        Updated Code:
        ```python
        [Your updated code here]
        ```
        """

        if manual_mode: 
            # define the content 
            content: list = []
            content.append({"type": "text", "text": code_engineer_prompt.format(question=question, critic_feedback=critic_feedback, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info)})

            # define the messages   
            messages = [HumanMessage(content)]

            # call the llm directly and get the response
            response = self.llm_primary_engineer.stream(messages)
            return self._process_streamed_response(response)
        else: 
            # create the agent
            code_engineer_agent = self.create_agent_wrapper(self.llm_primary_engineer, [self.execute_code], system_message=code_engineer_prompt.format(question=question, critic_feedback=critic_feedback, code_output=code_output, starter_code=starter_code, dataframes_info=dataframes_info))
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

    def create_summary_node(self, question: str, output: dict) -> str:
        """Explain the model output by providing a short summary of what the final code does."""
        print(output['code_output'])
        # Create messages list with separated code and output
        messages = [
            SystemMessage(
                content="""You are an AI assistant tasked with summarizing and interpreting code results. Focus on:
                1. Citing specific numerical values and statistics from the output
                2. Explaining what these numbers mean in context
                3. Highlighting key patterns or relationships found
                4. Keeping explanations clear and concise (3-4 sentences)"""
            ),
            HumanMessage(
                content=f"""Question Asked: {question}

                Code Generated:
                {output.get('code', '')}

                Code Results:
                {output.get('code_output', '')}

                Please provide a summary that:
                1. References specific numbers from the analysis (e.g., "the correlation coefficient of 0.75 indicates...")
                2. Explains what the calculations show about the data
                3. Interprets any visualizations or charts created
                4. Highlights key insights using actual values from the results

                Keep your response focused on the numerical findings and their interpretation."""
            )
        ]

        # Get response from LLM
        response = self.hyde_llm.invoke(messages)
        return response.content

    def create_summary_node_modified(self, question: str, output: dict, manual_mode: bool = True) -> str:
        """Explain the model output by providing a short summary of what the final code does."""

        # Create the prompt
        prompt = """
                **User's Question:**
                {question}

                **Generated Code**
                {code}
             
                **Code Output**
                {code_output}

                Please provide:
                1. A brief summary of what the code does and how it addresses the user's question.
                2. An interpretation of the results, including what any numerical outputs (like 0-5 scales) might mean in context.
                3. A thoughtful analysis of the results, highlighting key insights or patterns.
                4. Any potential limitations or considerations about the analysis.
                5. Make sure to mention specific numerical results from the coding output

                In your response:
                - Use plain language and avoid technical jargon.
                - Emphasize the main functionalities and key findings.
                - Provide context for any scales or metrics used (e.g., what a score of 3 out of 5 might indicate).
                - Offer insights that go beyond just restating the numbers.
                - If applicable, suggest potential implications or next steps based on the results.

                Note: Your response needs to be maximum 3 sentences long and includes the most relevant information.

                Your explanation should be analytical and educational, helping the user understand the results and how it relates to their question. 
            """

        # define the content
        content: list = []

        # add the system message first 
        content.append({"type": "text", "text": """You are an AI assistant tasked with summarizing and interpreting the final code and its results that were generated to answer the user's question. Your goal is to provide a clear, concise, and insightful analysis that a non-technical audience can understand."""})

        # add the human message 
        content.append({"type": "text", "text": prompt.format(question=question, code=output['code'], code_output=output['code_output'])})

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
                                                        system_message=prompt.format(question=question,
                                                                                        code=output['code'],
                                                                                        code_output=output['code_output']))

            summary_node = functools.partial(self.agent_node,
                                            agent=summary_node_agent,
                                            name="Summary Node Agent")

            return summary_node

    def define_graph(self, question: str, code_output: str, starter_code: str, previous_question: str, display_graph: bool = True):
        """define the nodes for the graph"""

        # update the state variables
        self._update_state_variables(question=question, code_output=code_output, critic_feedback='', current_code=starter_code)

        # define the tool node
        tool_node = ToolNode([self.code_executor_tool])

        # define the critic node
        critic_node, engineer_node = self.create_critic_node(question=question, code_output=code_output, starter_code=starter_code), self.create_code_engineer_node(question=question,
                                                                                                                                                                    code_output=code_output,
                                                                                                                                                                    starter_code=starter_code,
                                                                                                                                                                    critic_feedback='')

        # define the follow up task node
        follow_up_task_node = self.create_follow_up_task_decision_node(question=question, previous_question=previous_question)

        # define the draft started code node
        followup_code_node = self.create_followup_code_node(question=question, starter_code=starter_code)

        # define the first draft code node
        first_draft_code_node = self.create_first_draft_code_node_fixed(question=question)

        # define the summary node
        summary_node = self.create_summary_node_fixed(question=question, output=code_output)

        # define the graph
        self.graph_builder.add_node("first_draft_code", first_draft_code_node)
        self.graph_builder.add_node("follow_up_task", follow_up_task_node)
        self.graph_builder.add_node("code_execution", tool_node)
        self.graph_builder.add_node("critic", critic_node)
        self.graph_builder.add_node("engineer", engineer_node)
        self.graph_builder.add_node("followup_code", followup_code_node)
        self.graph_builder.add_node("summary", summary_node)

        # Change: Start with follow_up_task node
        self.graph_builder.add_edge(START, 'follow_up_task')

        # define the edge from follow up task to draft started code
        self.graph_builder.add_conditional_edges(
            'follow_up_task',
            self.router,
            {'call_draft_code': 'first_draft_code', 'call_edit': 'followup_code'}
        )

        # define the edge from the call edit to the code execution
        self.graph_builder.add_conditional_edges(
            'followup_code',
            self.router,
            {'continue': 'critic', 'call_tool': 'code_execution', END: END}
        )

        # define the edge from first draft code to follow up code
        self.graph_builder.add_conditional_edges(
            'first_draft_code',
            self.router,
            {'continue': 'critic', END: END}
        )

        # define the starting edge from code execution to critic
        self.graph_builder.add_conditional_edges(
            'critic',
            self.critic_router,
            {'continue': 'engineer', 'call_tool': 'code_execution', 'summary_exit': 'summary', END: END} # this should not be using a tool
        )

        # define the edge from summary to end
        self.graph_builder.add_conditional_edges(
            'summary',
            self.router,
            {'continue': END, END: END}
        )

        # define the edge from engineer to code execution
        self.graph_builder.add_conditional_edges(
            'engineer',
            self.router,
            {'continue': 'critic', 'call_tool': 'code_execution', END: END}
        )

        # define the edge from code execution to critic
        self.graph_builder.add_conditional_edges(
            'code_execution',
            lambda x: x["sender"],
            {'critic': 'critic', 'engineer': 'engineer'}
        )

        # # add the starting edge to the graph
        # self.graph_builder.add_edge(START, 'critic')
        compiled_graph = self.graph_builder.compile()
        print('the graph is compiled')

        # display the graph if requested

        if True:
          from IPython.display import Image, display
          from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

          display(
              Image(
                  compiled_graph.get_graph().draw_mermaid_png(
                      draw_method=MermaidDrawMethod.API,
                  )
              )
          )

          # self.display_graph(compiled_graph)

        return compiled_graph


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

    def _generate_starter_code_aide(self, prompt: str, steps: int = 3):
        """get the code from the aide library"""
        exp = aide.Experiment(
            data_dir="coding",  # replace this with your own directory
            goal=prompt,  # replace with your own goal description
            eval="RMSLE"  # replace with your own evaluation metric
        )

        # run the experiment
        best_solution = exp.run(steps=steps)

        # correct file paths in the code
        best_solution.code = self._modify_file_paths_aide_code(best_solution.code)

        # output the code from the aide experiment
        return best_solution.code

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
    def invoke_aide(self, query: str):
        # Ensure the coding directory exists
        os.makedirs("coding", exist_ok=True)

        # create the experiment
        exp = aide.Experiment(
            data_dir="coding",  # replace this with your own directory
            goal=query,  # replace with your own goal description
            eval="RMSLE"  # replace with your own evaluation metric
        )

        best_solution = exp.run(steps=10)

        print(f"Best solution has validation metric: {best_solution.valid_metric}")
        print(f"Best solution code: {best_solution.code}")

        return {
            'best_solution': best_solution.code,
            'best_solution_metric_results': best_solution.valid_metric
        }

    # Main execution function
    # TODO: make the whole workflow agentic even the task decomposition and starter code generation
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

        # do the follow up question routing 
        if router_decision_tool == 'call_draft_code': 
            # call the code draft
            code_response, self.dataframes_info = self.create_first_draft_code_node_modified(question, image_input=image_input)
            self.streamed_output += f'Code response: \n {code_response}'
            self.streamed_output += '---------------------------\n'
        elif router_decision_tool == 'call_edit' or not len(self.dataframes_info):
            code_response = self.create_followup_code_node(question, previous_code, self.dataframes_info, image_input=image_input)
            self.streamed_output += f'Code response: \n {code_response}'
            self.streamed_output += '---------------------------\n'
        else:
            raise Exception("404: router error")

        # save the code response in short term memory 
        internal_memory.append((question, code_response))
        # call the code executor 
        code_output = self.execute_code(code_response)
        self.streamed_output += f'Code output: \n {code_output}'
        self.streamed_output += '---------------------------\n'
        # save in short term memory the code and its response 
        internal_memory.append((code_response, code_output))

        # call the critic node 
        critic_response = self.create_critic_node(question, code_output, code_response, self.dataframes_info)
        # save the critic response in the short-term memory 
        internal_memory.append((code_response, code_output, critic_response))
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
                engineer_code_response = self.create_code_engineer_node(question, critic_response, code_output, code_response, self.dataframes_info)
                self.streamed_output += f'Engineer response: \n {engineer_code_response}'
                self.streamed_output += '---------------------------\n'
                # save the engineer response in the short-term memory 
                internal_memory.append((critic_response, engineer_code_response))

                # call the code executor 
                code_output = self.execute_code(engineer_code_response)
                # save in short term memory the code and its response 
                internal_memory.append((code_response, code_output))

                # call the critic again 
                critic_response = self.create_critic_node(question, code_output, engineer_code_response, self.dataframes_info)
                self.streamed_output += f'Critic response: \n {critic_response}'
                self.streamed_output += '---------------------------\n'
                # save the critic response in the short-term memory 
                internal_memory.append((code_response, code_output, critic_response))

                # overwrite the code_response with the engineer_code_response 
                code_response = engineer_code_response

                # get the critic router response 
                critic_router_response = self.critic_router({'messages': internal_memory})
                self.streamed_output += f'Critic router response: \n {critic_router_response}'
                self.streamed_output += '---------------------------\n'

                # update the counter 
                current_counter += 1

            # define the summary node 
            summary = self.create_summary_node_modified(question, {'code_output': code_output, 'code': code_response})
            self.streamed_output += f'Summary: \n {summary}'
            self.streamed_output += '---------------------------\n'
        elif critic_router_response == 'summary_exit':
            summary = self.create_summary_node_modified(question, {'code_output': code_output, 'code': code_response})
            self.streamed_output += f'Summary: \n {summary}'
            self.streamed_output += '---------------------------\n'
        else:
            raise Exception("404: critic router error")

        print('\n')
        print(code_output)
        print('\n')

        # save the chat history 
        self.add_chat_history_to_management(user_id=user_id, model_id=self.model_id, chat_id=chat_id, question_answer_pair=(question, code_response))

        # get the byte strings of the html files in the coding directory
        html_byte_strings = self.html_to_byte_strings_in_directory(self.data_loader.download_path_model.coding_path)

        # move the html files from the coding directory to the charts directory
        self.move_html_files_from_coding_to_plots() 

        return summary, html_byte_strings
        

def old_prompts():
    """old prompts for the agents"""

    # System messages
    system_msg_engineer = """
    Engineer. You write python/bash to retrieve relevant information. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
    Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    Make sure to use plotly with nice graphics. Make sure to download the file after you are done.
    """
    system_msg_planner = """
    Planner. Given a task, please determine what information is needed to complete the task.
    Please note that the information will all be retrieved using Python code. Please only suggest information that can be retrieved using Python code.
    If required, make the graph visually appealing by incorporating the themes and styles from plotly with a modern template. Make sure to pick the write type of chart and stick to it.
    Include labels, titles, and a legend (if applicable).
    Ensure the graph is saved as a .png file (if applicable).
    """


#agent_obj = AgentConfig(specific_file_name='/mvp_vertafore_data_production.csv')
#user_id = 1
#chat_id = 1

# Create an instance of MyStreamHandler
#stream_handler = MyStreamHandler()

# Set user_id and chat_id before starting
#stream_handler.set_user_and_chat(user_id=user_id, chat_id=chat_id)

#agent_obj._create_prompt('find me the average age of the respondents. Make sure to clean the age column first')
#agent_obj._create_prompt('find me the average minutes that were spent during a session? use /content/coding/kinkinterestpublic.csv')
#agent_obj._generate_starter_code_llm('get me the average of the this coding file')
#starter_code, file_paths, agent_prompt = agent_obj._create_prompt('get me the average of the age column')
#task, file_names, agent_prompt, task_breakdown = agent_obj._create_prompt('get me the average of the age column')
#agent_obj.conditional_invoke('what is the most common type of transaction descrption', user_id=user_id, chat_id=chat_id)

