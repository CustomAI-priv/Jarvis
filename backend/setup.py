import spacy
import pandas as pd
from abc import ABCMeta, abstractmethod, ABC
import numpy as np
import json
import os
import shutil
import uuid
import requests
from io import BytesIO
import io
import base64
import bs4
import tabula
from functools import lru_cache
import time
import asyncio
from tqdm import tqdm
from typing import Any, Annotated, Callable, Dict, List, Tuple, Sequence, TypedDict, ClassVar, Union, cast
import re
from openparse import processing, DocumentParser
from time import sleep
from pprint import pprint
from locale import getpreferredencoding
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
import webbrowser
import cohere
import pyarrow as pa
import torch
import fitz
import Stemmer

from PIL import Image
import voyageai
import PIL 
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from pypdf import PdfReader as PdfReader2  # Renamed to avoid conflict

import dropbox
from dropbox.exceptions import ApiError
from dropbox.files import WriteMode, ListFolderResult, DeletedMetadata, FileMetadata

from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings as ChromaSettings

from langchain import hub
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.load import dumps, loads
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, LanceDB
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

#from langchain_mistralai import MistralAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI as AzureOpenAI_v1

from langgraph.graph import END, StateGraph

from ragatouille import RAGPretrainedModel
from operator import itemgetter
from flashrank import Ranker, RerankRequest

import lancedb
from lancedb.embeddings import get_registry, EmbeddingFunctionRegistry, TextEmbeddingFunction
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
from lancedb.embeddings.registry import register
from lancedb.util import attempt_import_or_raise

from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoProcessor

from colpali_engine.models.paligemma import ColPali, ColPaliProcessor
from vespa.package import (
    Schema, 
    Field, 
    FieldSet, 
    HNSW, 
    ApplicationPackage, 
    RankProfile, 
    Function, 
    FirstPhaseRanking, 
    SecondPhaseRanking,
    Document as VespaDocument  # Rename import to avoid conflicts
)
import pyarrow as pa
from vespa.io import VespaQueryResponse
from vespa.io import VespaResponse
from vespa.deployment import VespaDocker, VespaCloud


# import company and sector definitions
try:
    from sector_settings import SectorSettings
except ImportError:
    print('sector_settings.py not found')
    pass
try:
    from backend.sector_settings import SectorSettings
except ImportError:
    print('sector_settings.py not found')
    pass
print(SectorSettings)
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 encoding
getpreferredencoding = lambda: "UTF-8"
warnings.filterwarnings('ignore')

# Ensure UTF-8 encoding
getpreferredencoding = lambda: "UTF-8"
warnings.filterwarnings('ignore')

# import the api keys
from api_keys import api_key, claude_api_key, voyage_api_key, cohere_api_key, azure_api_key, azure_endpoint

# set the environment variables
os.environ['ANTHROPIC_API_KEY'] = claude_api_key
os.environ["OPENAI_API_KEY"] = api_key
os.environ["VOYAGE_API_KEY"] = voyage_api_key

# set vespa settings 
os.environ['TOKENIZERS_PARALLELISM'] = "false"

RAG_colbert_downloaded = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
nlp = spacy.load("en_core_web_sm")
hugging_face_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') #BAAI/bge-large-zh-v1.5
if "LANGCHAIN_TRACING_V2" in os.environ:
    del os.environ["LANGCHAIN_TRACING_V2"]
os.environ.pop("LANGCHAIN_TRACING_V2", None)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
#prompt = hub.pull("rlm/rag-prompt")
co = cohere.Client(cohere_api_key)
openai_embedding_key: str = 'text-embedding-3-small'
openai_embeddings: Any = OpenAIEmbeddings(model=openai_embedding_key,
                            openai_api_key=api_key)


class GeneralSettings(BaseModel):
    """settings pydantic model to establish configs"""
    # Embedding configuration
    embedding_selected: int = 4

    # Text splitting settings
    chunk_size: int = 600
    chunk_overlap: int = 50
    splitter_mech_type: str = 'recursive'
    markdown_headers_to_split_on: list = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # vector store ids
    vectordb_idx: dict = {
        1: 'db_text',
        2: 'db_tables',
        3: 'db_table_descriptions', 
        4: 'db_text_image_embeddings' # multimodal mode
    }
    text_databases: list = [1]
    documents_schema_index: int = 2 # check inside the VectorDB class for the specific index
    indexable_field: str = 'page_content'
    vectordb_type: str = 'lancedb' # or colbert
    database_path: str = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))), 'db/')
    colbert_params: dict = {"k": 10}
    chroma_params: dict = {"search_type": "mmr"}
    k_top: int = 20 # define the number of results to return from chroma vector database
    k_top_multimodal: int = 2
    k_tab_multiplier: int = 4
    hybrid_search_ratio: float = 0.7
    reranked_top: int = 5
    multi_modal_mode: bool = True

    # define the directories and their loading schemas
    dirs: list = ['pdf', 'tables', 'markdown']

    # first time run functionality
    initial_run: bool = False

    # define the number of threads of the files
    max_workers: int = 1
    embedding_batch_size: int = 20

    # CSV analyzer model settings
    csv_analysis_model: str = "gpt-4o-mini"
    summary_temp: float = 0.0  # Temperature for CSV summary generation
    top_k_csv_files: int = 2
    csv_chunk_size: int = 6000

    # define the uploading method for files  
    upload_method: str = 'auto' # or 'manual'
    retain_files_condition: bool = True
    disable_external_upload: bool = True


class DownloadPaths(BaseModel):
    """define the download paths for the files to sort them into their respective directories"""
    base_path: str = os.getcwd()  # Get the current working directory
    markdown_path: str = os.path.join(base_path, 'markdown')
    pdf_path: str = os.path.join(base_path, 'pdf')
    table_path: str = os.path.join(base_path, 'table')
    json_path: str = os.path.join(base_path, 'json')
    image_path: str = os.path.join(base_path, 'images')
    coding_path: str = os.path.join(base_path, 'coding')
    charts_path: str = os.path.join(base_path, 'charts')


class StorageSettings(BaseModel):
  """settings for where to store data from vector databases"""

  # define the paths (subject to change)
  chromadb_path: str = '/content/chroma_data/'
  colbert_path: str = '/content/.ragatouille/colbert/indexes/'

  # starting information for colbert system
  initial_data_path: str = '/content/initial_data.txt'


# define a text embedding for LanceDB custom made integrating Azure
@register("azure_embedding")
class SentenceTransformerEmbeddings(TextEmbeddingFunction):
    deployment_name: str = 'jarvis_embedding'
    # set more default instance vars like device, etc.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # define the number of dimensions with default None
        self._ndims = None

    def generate_embeddings(self, texts):
        """generate the embeddings for the texts"""
        # generate the embeddings
        results = self._embedding_model().embeddings.create(input = texts, model=self.deployment_name).data #[0].embedding
        
        # return the embeddings as a list structure 
        return [i.embedding for i in results]

    def ndims(self):
        """define the number of dimensions of the embedding"""

        # check if the number of dimensions is already defined
        if self._ndims is None:
            # define the number of dimensions
            self._ndims = len(self._embedding_model().embeddings.create(input = ['foo'], model=self.deployment_name).data[0].embedding)
        return self._ndims

    #@cached(cache={})
    @lru_cache(maxsize=None)
    def _embedding_model(self):
        """define the embedding model to use"""
        client = AzureOpenAI_v1(
            api_key = azure_api_key,
            api_version = "2024-02-01",
            azure_endpoint = azure_endpoint
          )
        return client


# define the custom embedding
registry = EmbeddingFunctionRegistry.get_instance()
lancedb_embeddings = registry.get("azure_embedding").create(max_retries=2) #get_registry().get("openai").create(name=openai_embedding_key, max_retries=2)


class DocumentsSchema(LanceModel):
    vector: Vector(lancedb_embeddings.ndims()) = lancedb_embeddings.VectorField()  # Adjust the dimension as needed
    page_content: str = lancedb_embeddings.SourceField()
    source: str
    page: int # impose the integer type here for metadata filtering purposes
    location_on_page: str
    additional_info: str


# define the multi modal schema for the table
multi_modal_schema = pa.schema([
    pa.field('vector', pa.list_(pa.float32(), 1024)),
    pa.field('page_image_bytes', pa.string()),
    pa.field('source', pa.string()),
    pa.field('page', pa.int32()),
])


class ResearchPapersSchema(LanceModel):
    vector: Vector(lancedb_embeddings.ndims()) = lancedb_embeddings.VectorField()  
    summary: str = lancedb_embeddings.SourceField()
    title: str
    authors: str
    updated: str
    published: str
    pdf_link: str
    affiliations: str


class Utilities:
    """class of utility functions for various common tasks"""

    def __init__(self):
        # embeddings
        self.embedding_idx: Dict[int, Any] = {
            1: GPT4AllEmbeddings(),
            2: openai_embeddings,
            3: lancedb_embeddings, 
            4: voyageai.Client() #VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model="voyage-3")
        }
        # define the settings model to be used globally
        self.settings = GeneralSettings()
        self.storage_settings = StorageSettings()
        self.sector_settings = SectorSettings()

        # define the path settings model
        self.download_path_model: Any = DownloadPaths()

        # define the dropbox parameters
        self.app_key: str = 'mjzyoqx7m3it8gb'
        self.app_secret: str = '01itl53fwmnnh1z'
        self.dbx_refresh_token: str = 'x5hflpVIknYAAAAAAAAAAdbvnEVbpPi713c-jUDoCJx3sheHytSDpf8hqcxA5kSj'
        self.dbx = dropbox.Dropbox(
                    app_key = self.app_key,
                    app_secret = self.app_secret,
                    oauth2_refresh_token = self.dbx_refresh_token
                )
        
        self._initialized = True

    # Function to check if the database is empty
    @staticmethod 
    def is_database_empty(db):
        table_names = db.table_names()
        if not table_names:
            # No tables in the database
            return True
        total_docs = 0
        for table_name in table_names:
            # Open the table
            table = db.open_table(table_name)
            # Count the number of documents
            doc_count = len(table.to_pandas())
            total_docs += doc_count
        return total_docs == 0
    
    @staticmethod
    def process_flashrank_responses(results: list) -> list:
        """post process the results from the flashrank reranking algorithm"""

        # define the output container
        container: list = []

        # iterate through each result and format it into a Langchain document
        for result in results:
            container.append(
                Document(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
            )
        return [container]

    @staticmethod
    def format_file_name(text: str):
        """format the file name for the vector database and keep it standard across all collections"""
        return re.sub(r'\s+', ' ', text).strip().replace(' ', '_')

    @staticmethod
    def group_files_by_type(file_list):
        grouped_files = {
            "tables": [],
            "pdf": [],
            "markdown": [],
            "json": [],
            "images": []
        }

        for file in file_list:
            ext = os.path.splitext(file)[1].lower()
            if ext == ".csv":
                grouped_files["tables"].append(file)
            elif ext == ".pdf":
                grouped_files["pdf"].append(file)
            elif ext == ".md":
                grouped_files["markdown"].append(file)
            elif ext == ".json":
                grouped_files["json"].append(file)
            elif ext in [".jpeg", ".jpg", ".png"]:
                grouped_files["images"].append(file)

        return grouped_files

    @staticmethod
    def generate_random_id():
        """generate a random ID for an object in a vector store or other database type"""
        return str(uuid.uuid4())

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def format_output(text: str):
        """format the output of the chain"""
        return text.split('Answer: ')[-1]

    def process_new_documents(self, documents: List[Any]) -> List[list]:
        """process new documents that are entering the collection"""

        # get the page contents
        doc_contents: list = [i.page_content for i in documents]

        # get the metadatas
        doc_metadatas: list = [i.metadata for i in documents]

        # get the document ids
        doc_ids: list = [self.generate_random_id() for _ in range(len(documents))]
        return [doc_contents, doc_metadatas, doc_ids]

    def get_all_files_dir(self, dir: str):
        """get all files in the directory"""
        files = [os.path.join(dir, file) for file in os.listdir(dir) if os.path.isfile(os.path.join(dir, file))]
        return files

    def get_all_dir_names(self, dir: str):
        """get all the directory names"""
        return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

    def format_docs_reranked(self, docs: list):
        """format the reranked documents"""
        try:
            return "\n\n".join(doc.page_content for doc in docs) #"\n\n".join(doc[0].page_content for doc in docs)
        except:
            return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_reranked_v2(self, docs: List[Document]) -> str:
        """
        Format reranked documents with selected metadata (source and page) for enhanced context.

        Args:
            docs (List[Document]): A list of reranked Document objects.

        Returns:
            str: A formatted string containing document contents with source and page metadata.
        """
        formatted_docs = ""
        for idx, doc in enumerate(docs, start=1):
            formatted_docs += f"### Document {idx}\n"

            # Extract only 'source' and 'page' from metadata with default values if missing
            source = doc.metadata.get('source', 'Unknown Source')
            page = doc.metadata.get('page', 'Unknown Page')

            formatted_docs += f"**Source**: {source}\n"
            formatted_docs += f"**Page**: {page}\n"
            formatted_docs += f"{doc.page_content}\n\n"
        return formatted_docs

    @staticmethod
    def delete_file(file_path: str):
        """
        Deletes the file at the given file_path.

        Parameters:
        file_path (str): The path of the file to delete.
        """
        try:
            # Check if file exists
            if os.path.exists(file_path):
                os.remove(file_path)
                #print(f"File {file_path} deleted successfully.")
            else:
                print(f"File {file_path} does not exist.")
        except Exception as e:
            print(f"Error: {e}")

    @staticmethod
    def location_on_page_to_string(location_on_page):
        # Convert the location_on_page dictionary to a formatted string
        x0 = location_on_page.get("x0", "None")
        y0 = location_on_page.get("y0", "None")
        x1 = location_on_page.get("x1", "None")
        y1 = location_on_page.get("y1", "None")

        return f"x0: {x0}, y0: {y0}, x1: {x1}, y1: {y1}"

    def standardize_openparse_to_langchain(self, parsed_document):
        """parse through the openparse results to standardize to langchain"""
        print('Length of Nodes: ')
        print(len(parsed_document.nodes))
        documents = []

        # Loop through each node in the parsed document
        for node in parsed_document.nodes:
            # Extract the page content (text) from the text elements
            page_content = "\n".join([element.embed_text for element in node.elements])

            # Extract metadata information
            page_number = node.elements[0].bbox.page if node.elements else None
            source = parsed_document.id_  # Assuming source can be the parsed document's ID

            # Convert location_on_page to a string
            location_on_page_dict = {
                "x0": node.elements[0].bbox.x0 if node.elements else None,
                "y0": node.elements[0].bbox.y0 if node.elements else None,
                "x1": node.elements[0].bbox.x1 if node.elements else None,
                "y1": node.elements[0].bbox.y1 if node.elements else None,
            }
            location_on_page_str = self.location_on_page_to_string(location_on_page_dict)

            # Build the metadata dictionary
            metadata = {
                "page": int(page_number),
                "source": source,
                "location_on_page": location_on_page_str, # Now a string
                "additional_info": ""
            }

            # Create the LangChain Document
            langchain_doc = Document(page_content=page_content, metadata=metadata)
            documents.append(langchain_doc)

        return documents

    @staticmethod
    def sanitize_file_path(file_path: str) -> str:
        """
        Sanitize file path by keeping only the filename and replacing any remaining
        slashes with underscores for SQL compatibility.

        Args:
            file_path (str): The original file path.

        Returns:
            str: The sanitized file path.
        """
        # Extract the basename (filename with extension)
        base_name = os.path.basename(file_path)
        
        # Replace any remaining slashes with underscores (if any)
        sanitized_path = base_name.replace('/', '_').replace('\\', '_')

        return sanitized_path

    @staticmethod
    def get_base64_image(image, add_url_prefix=True):
        """
        Converts a PIL Image to a base64 encoded string.

        Args:
            image (PIL.Image.Image): The image to encode.
            add_url_prefix (bool): If True, adds 'data:image/png;base64,' prefix to the string.

        Returns:
            str: The base64 encoded string of the image.
        """
        # Create a byte buffer to store the image data
        buffered = BytesIO()
        
        # Save the image to the buffer in PNG format
        image.save(buffered, format="PNG")
        
        # Encode the image data to base64 and convert to a UTF-8 string
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Optionally add the data URL prefix
        if add_url_prefix:
            return f'data:image/png;base64,{img_str}'
        else:
            return img_str

    @staticmethod
    def scale_image(image, width):
        """
        Scale the image to the given width while maintaining the aspect ratio.

        Args:
            image (PIL.Image.Image): The input image to be scaled.
            width (int): The desired width in pixels.

        Returns:
            PIL.Image.Image: The scaled image.
        """
        # Get original dimensions
        original_width, original_height = image.size

        # Calculate the new height to maintain aspect ratio
        aspect_ratio = original_height / original_width
        new_height = int(width * aspect_ratio)

        # Resize the image using the LANCZOS filter
        resized_image = image.resize((width, new_height), Image.LANCZOS)
        return resized_image


class FileManagement(Utilities):
    """manage the file management"""

    def __init__(self):
        super().__init__()

    # This function manually downloads a file from the vector database and organizes it into appropriate folders.
    def manual_file_download(self, file_path: str, **kwargs) -> None:
        """
        Manually download a file from manual_upload directory and organize it into appropriate folders.
        
        Args:
            file_path (str): Name of the file to be moved from manual_upload directory
        """
        # Get the file name from the path
        file_name = os.path.basename(file_path)
        
        # Define source path (manual upload directory)
        manual_upload_path = os.path.join(self.download_path_model.base_path, 'manual_upload')
        source_path = os.path.join(manual_upload_path, file_name)

        # Determine the destination folder based on file extension
        if file_name.endswith('.md'):
            destination = os.path.join(self.download_path_model.markdown_path, file_name)
        elif file_name.endswith('.pdf'):
            destination = os.path.join(self.download_path_model.pdf_path, file_name)
        elif file_name.endswith('.json'):
            destination = os.path.join(self.download_path_model.json_path, file_name)
        elif file_name.endswith(('.csv', '.xls', '.xlsx')):
            destination = os.path.join(self.download_path_model.table_path, file_name)
        elif file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            destination = os.path.join(self.download_path_model.image_path, file_name)
        else:
            print(f"Unsupported file type for {file_name}. Please upload a markdown, pdf, json, table, or image file.")
            return

        try:
            # Move the file to the appropriate folder with relative paths
            shutil.move(os.path.normpath('/' + source_path), os.path.normpath('/' + destination))
            print(f"Successfully moved {file_name} to {destination}")
        except FileNotFoundError:
            print(f"WARNING: File {file_name} not found in manual_upload directory, likely was already moved")
        except Exception as e:
            print(f"ERROR: Failed to move {file_name}: {str(e)}")

    # This function moves a file manually to the appropriate folder.
    def move_file_manually(self, file_name: str, **kwargs):
        """Move a file manually from the manual_upload directory to the specified destination folder."""

        # Get the local directory from the kwargs
        local_dir: str = kwargs.get('local_dir', None)

        # Get the source directory from the kwargs
        source_dir: str = kwargs.get('source_dir', None)
        
        # Define the source path from the manual_upload directory
        source_path = source_dir + file_name

        # Define the destination path
        destination_path = local_dir + file_name

        # Move the file from the manual_upload directory to the specified local directory
        try:    
            shutil.move(os.path.normpath(source_path), os.path.normpath(destination_path))
            #print(f"Successfully moved {file_name} from {source_path} to {destination_path}.")
        except Exception as e:
            pass 
    
    # This function manages the manual file download and upload from either the manual upload directory or the dropbox.
    def manual_file_manager(self, file_path: str, **kwargs) -> None:
        """manage the manual file download and upload"""
        
        # download the files
        if kwargs.get('download', False):
            self.manual_file_download(file_path, **kwargs)

        # move the files
        elif kwargs.get('move', False):
            self.move_file_manually(file_path, **kwargs)
        
        return 

    def manual_move_file_back(self, file_path: str, **kwargs):
        """move a file back to the manual upload directory"""
        
        # get the local directory from the kwargs
        local_dir: str = kwargs.get('local_dir', '/')

        # get the destination directory from the kwargs
        destination_dir: str = kwargs.get('destination_dir', '/')

        # define the source path
        source_path = local_dir + file_path

        # define the destination path
        destination_path = destination_dir + file_path

        # move the file from the local directory to the manual upload directory
        try:
           shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Error moving file: {e}")

    # This function retrieves recent files from the manual upload directory based on the specified time frame.
    def get_recent_manual_files(self) -> list:
        """Get all files uploaded in the last 1 day from the manual_upload directory."""
        recent_files = []
        one_day_ago = datetime.now() - timedelta(days=1)
        
        # Define the path to the manual upload directory
        manual_upload_path: str = os.path.join(self.download_path_model.base_path, 'manual_upload/')
        
        # Get a list of all files in the manual_upload directory
        for file_name in os.listdir(manual_upload_path):

            # get the file path
            file_path = os.path.join(manual_upload_path, file_name)

            # get the file modification time
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(file_name, file_mod_time)

            # Check if the file was modified in the last day
            if os.path.isfile(file_path) and file_mod_time > one_day_ago:
                recent_files.append(file_name)

        # save the current files to a text file
        with open('current_files.txt', 'w') as f:
            for file in recent_files:
                f.write(f"{os.path.basename(file)}\n")

        return recent_files

    # This function retrieves recent files from Dropbox based on the specified time frame.
    @staticmethod
    def get_recent_dropbox_files(dbx, folder_path='/', minutes=6000):
        recent_files = []
        current_time = datetime.now(timezone.utc)
        time_delta = timedelta(minutes=minutes)
        # List all files in the specified folder
        result = dbx.files_list_folder('')

        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                # Ensure server_modified is timezone-aware
                server_modified = entry.server_modified.replace(tzinfo=timezone.utc)
                # Check if the file was modified within the last `minutes` minutes
                if current_time - server_modified <= time_delta:
                    recent_files.append(entry.path_display)

        return recent_files

    # This function retrieves recent files based on the specified upload method (manual or automatic).
    def get_recent_files(self, **kwargs) -> list:
        """get the recent files from the manual upload and the dropbox"""
        
        # Check if the upload method is set to 'manual'
        if self.settings.upload_method == 'manual':
            # If manual, retrieve recent files from the manual upload directory
            return self.get_recent_manual_files()
        # If the upload method is set to 'auto'
        elif self.settings.upload_method == 'auto':
            # Retrieve the Dropbox client from the provided keyword arguments
            dbx = kwargs.get('dbx', None)
            # Raise an error if the Dropbox client is not provided
            if dbx is None:
                raise ValueError("dbx client is not provided")
            # If the Dropbox client is available, retrieve recent files from Dropbox
            return self.get_recent_dropbox_files(dbx)

    # This function downloads a file from Dropbox to a local directory.
    def download_from_dropbox(self, dbx, dropbox_path: str, local_dir: str = 'downloads') -> None:
        """
        Download a file from Dropbox to a local directory.

        Args:
            dbx: The Dropbox client object.
            dropbox_path (str): The path of the file in Dropbox.
            local_dir (str): The local directory to save the file to. Defaults to 'downloads'.

        Returns:
            Optional[str]: The path where the file was saved locally, or None if download failed.
        """
        # Ensure the dropbox_path starts with a forward slash
        dropbox_path = f"/{dropbox_path.lstrip('/')}"

        # Create the local directory if it doesn't exist
        full_local_dir = os.path.join(os.getcwd(), local_dir)
        os.makedirs(full_local_dir, exist_ok=True)

        # Get the filename from the Dropbox path
        file_name = os.path.basename(dropbox_path)

        # Construct the full local path
        local_path = os.path.join(full_local_dir, file_name)

        # Download the file
        try:
            dbx.files_download_to_file(local_path, dropbox_path)
            #print(f"File downloaded to: {local_path}")
            return
        except Exception as e:
            print(f"Error downloading file: {e}")
            return

    # This function handles deleted files from Dropbox.
    @staticmethod
    def handle_deleted_files(result: ListFolderResult, minutes=150):
        # Initialize an empty list to store paths of deleted files.
        container: list = []
        # Iterate through each entry in the result's entries.
        for entry in result.entries:
            # Check if the entry is a deleted file.
            if isinstance(entry, dropbox.files.DeletedMetadata):
                # If it is deleted, append its path to the container list.
                container.append(entry.path_display)
        # Return the list of deleted file paths.
        return container
    
    # This function lists deleted files from Dropbox.
    def list_deleted_dropbox_files(self, dbx, path=""):
        container: list = []
        try:
            result = dbx.files_list_folder('', include_deleted=True)
            container.extend(self.handle_deleted_files(result))

            # Handle pagination if there are more entries
            while result.has_more:
                result = dbx.files_list_folder_continue(result.cursor)
                container.extend(self.handle_deleted_files(result))

        except dropbox.exceptions.ApiError as err:
            print(f"API error: {err}")
            pass
        return container
    
    # This function lists deleted files in the manual upload directory.
    def list_deleted_manual_upload_files(self) -> list:
        """
        List all files that have been deleted by comparing current server files 
        with the previous state stored in current_files.txt
        """
        # define a container to fill with the deleted files
        deleted_files: list = []
        
        # define a container to fill with the current files
        current_server_files: set = set()

        # define the directories to search for files
        directories: dict = {
            'pdf': self.download_path_model.pdf_path,
            'images': self.download_path_model.image_path,
            'json': self.download_path_model.json_path,
            'tables': self.download_path_model.table_path,
            'markdown': self.download_path_model.markdown_path,
            'manual_upload': os.path.join(self.download_path_model.base_path, 'manual_upload/')
        }
        
        # Get all current files from all directories
        for dir_type, dir_path in directories.items():
            if os.path.exists(dir_path):
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):  # Make sure it's a file
                        current_server_files.add(os.path.basename(file_name))
                        print(f"Found file in {dir_type}: {file_name}")  # Debug print
        
        # Read the previous state from current_files.txt
        try:
            with open('current_files.txt', 'r') as f:
                previous_files: set = set(line.strip() for line in f.readlines())
        except FileNotFoundError:
            previous_files: set = set()
        
        # Find files that were in previous_files but are not in current_server_files
        deleted_files = list(previous_files - current_server_files)
        
        return deleted_files

    # This function lists deleted files from both the manual upload and the dropbox.
    def list_deleted_files(self, **kwargs):
        """list the deleted files from both the manual upload and the dropbox"""
        
        # Check the upload method to determine how to list deleted files
        if self.settings.upload_method == 'manual':
            print("in manual mode")
            # If the upload method is manual, call the function to list deleted manual upload files
            return self.list_deleted_manual_upload_files()
        elif self.settings.upload_method == 'auto':
            # If the upload method is automatic, retrieve the Dropbox client from kwargs
            dbx = kwargs.get('dbx', None)
            # Check if the dbx client is provided; raise an error if not
            if dbx is None:
                raise ValueError("dbx client is not provided")
            # Call the function to list deleted files from Dropbox
            return self.list_deleted_dropbox_files(dbx=dbx)
        
    # This function uploads a file to Dropbox, replacing the old file if it exists.
    def upload_to_dropbox(self, file_path: str) -> None:
        """
        Upload a file to Dropbox, replacing the old file if it exists.

        Args:
            file_path (str): The path to the file to be uploaded.
        """
        dropbox_path = f'/{os.path.basename(file_path)}'
      
        try:
            # Check if file already exists
            try:
                self.dbx.files_get_metadata(dropbox_path)
                print(f"File {dropbox_path} already exists in Dropbox. It will be overwritten.")
            except ApiError as e:
                if e.error.is_path() and e.error.get_path().is_not_found():
                    print(f"File {dropbox_path} does not exist in Dropbox. A new file will be created.")
                else:
                    raise

            # Upload the file
            with open(file_path, 'rb') as file:
                file_size = os.path.getsize(file_path)
                if file_size <= 150 * 1024 * 1024:  # 150 MB
                    self.dbx.files_upload(
                        file.read(),
                        dropbox_path,
                        mode=WriteMode.overwrite
                    )
                else:
                    # For larger files, use upload session
                    upload_session_start_result = self.dbx.files_upload_session_start(file.read(1024 * 1024))
                    cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                                offset=file.tell())
                    commit = dropbox.files.CommitInfo(path=dropbox_path, mode=WriteMode.overwrite)

                    while file.tell() < file_size:
                        if (file_size - file.tell()) <= 1024 * 1024:
                            self.dbx.files_upload_session_finish(file.read(1024 * 1024),
                                                                  cursor,
                                                                  commit)
                        else:
                            self.dbx.files_upload_session_append(file.read(1024 * 1024),
                                                                 cursor.session_id,
                                                                 cursor.offset)
                            cursor.offset = file.tell()

            print(f"Successfully uploaded {os.path.basename(file_path)} to Dropbox.")
        except ApiError as e:
            print(f"Dropbox API error: {e}")
        except Exception as e:
            print(f"Error uploading file to Dropbox: {str(e)}")


class VectorDB(FileManagement):
    """
    This class manages the interaction with the vector database, facilitating:

    - **Creation**: Enables the creation of vector embeddings for various document types.
    - **Retrieval**: Provides methods to retrieve stored embeddings efficiently.
    - **Manipulation**: Allows for the manipulation of vector embeddings as needed.

    Key functionalities include:

    - **Database Initialization**: Initializes the database connections.
    - **Schema Handling**: Manages schema definitions for different document types.
    - **Document Management**: Offers methods for loading and deleting documents.

    This ensures efficient data management for retrieval-augmented generation (RAG) systems.
    """

    def __init__(self):
        super().__init__()

        # define the selected embedding model
        self.embedding = self.embedding_idx[self.settings.embedding_selected]

        # define the mapping of database schemas 
        self.vectordb_schemas = {1: ResearchPapersSchema, 2: DocumentsSchema}

        # initialize databases if it is the first run of the code
        #if self.settings.initial_run:
        #  self.create_dbs()

        # fetch all the clients/DBs for the RAG system
        #self.chroma_client = self.fetch_cols_client()
        self.lancedb_client = self.fetch_lancedb_client()
        self.chroma_client = None

        # organize the retrievers by text and tables
        #self.collections_idx = {
        #  "chroma": {1: self.chroma_text_col, 2: self.chroma_table_col},
        #  "colbert": {1: self.colbert_text_col, 2: self.colbert_table_col}
        #}
        self._initialized = True

    def create_dbs(self):
        """create the databases if they do not already exist"""
        # create the vector store clients
        if self.settings.vectordb_type == 'chroma':
            self.pdf_col, self.tab_col = self.initialize_chromadb()
        else:
            # check if the database already exists
            self.pdf_col, self.tab_col = self.initialize_colbert()

    def fetch_cols_client(self):
        """fetch all the databases for chroma and for colbert"""

        # for chromadb
        chroma_client = chromadb.PersistentClient(path=self.storage_settings.chromadb_path)
        #chroma_text_col, chroma_table_col = chroma_client.get_or_create_collection(self.settings.vectordb_idx[1]), chroma_client.get_or_create_collection(self.settings.vectordb_idx[2])

        # for colbert preload this and then make sure that it does not get reloaded again after
        #text_path_colbert, table_path_colbert = self.storage_settings.colbert_path + 'db_text', self.storage_settings.colbert_path + 'db_tables'
        #colbert_text_col, colbert_table_col = RAGPretrainedModel.from_index(text_path_colbert), RAGPretrainedModel.from_index(table_path_colbert)

        #colbert_text_col, colbert_table_col = None, None

        return chroma_client

    def fetch_lancedb_client(self) -> dict:
        """Create or open the LanceDB client and tables, overwriting only if the database is empty."""

        # Output container for the tables
        container: dict = {}

        # Create the database connection
        db = lancedb.connect(self.settings.database_path)

        # Check if the database is empty
        database_empty = self.is_database_empty(db)

        # Get the list of existing table names
        existing_tables = db.table_names()

        # For each table we need, check if it exists
        for key, val in self.settings.vectordb_idx.items():
            # define the schema 
            schema = self.vectordb_schemas[self.settings.documents_schema_index] if key != 4 else multi_modal_schema
            if database_empty:
                # Database is empty, create tables with overwrite mode
                container[key] = db.create_table(
                    name=val,
                    mode="overwrite",
                    schema=schema
                )
                print(f"Created new table with overwrite: {val}")
            else:
                if val in existing_tables:
                    # Open existing table
                    container[key] = db.open_table(name=val)
                    print(f"Opened existing table: {val}")
                else:
                    # Table doesn't exist, create it without overwriting existing tables
                    container[key] = db.create_table(
                        name=val,
                        schema=schema
                    )
                    print(f"Created new table: {val}")

        return container

    def initialize_chromadb(self) -> list:
        """initialize the chromadb collection"""

        # create container for collections
        collections: list = []

        # create the chromadb client
        client = chromadb.PersistentClient(path=self.storage_settings.chromadb_path)
        # create the collections
        for collection_name in self.settings.vectordb_idx.values():
            # define the client and open the collection
            client.get_or_create_collection(collection_name)

        # create the collection in chromadb
        collection = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embedding,
        )
        collections.append(collection)

        return collections

    def initialize_colbert(self) -> list:
        """initialize the ragatouille database with its settings"""

        # client container
        collections: list = []

        # define the colbert model
        RAG = None #RAG_colbert_downloaded

        # read in the initial data
        with open(self.storage_settings.initial_data_path, 'r') as file:
            file_content = self.clean_text(file.read())

        # create the index path for ragatouille colbert
        for collection_name in self.settings.vectordb_idx.values():
            collections.append(
                RAG.index(index_name=collection_name,
                            collection=[file_content],
                            document_metadatas=[{'source': self.storage_settings.initial_data_path, "page": 0}],
                            document_ids=[self.generate_random_id()],
                            split_documents=True,
                            max_document_length=self.settings.chunk_size,
                        )
            )

        return collections

    def clean_text(self, text: str):
        """preprocess any text for special characters and any strange notation"""
        # Normalize Unicode characters to the closest ASCII representation
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove all single and double quotes
        text = re.sub(r"[\"']", '', text)

        # Replace all non-word characters (except for spaces) with a space
        text = re.sub(r'[^\w\s-]', ' ', text)

        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Strip whitespace at the beginning and end of the line
        text = text.strip()
        return text

    def standard_vector_store(self, document_splits: list, **kwargs):
        """standard vector store"""
        # define the vector store
        vectorstore: Any = Chroma.from_documents(
        documents=document_splits,
        collection_name=self.settings.vectordb_idx[kwargs['db']],
        embedding=self.embedding
        )

        # define the document retriever
        retriever = vectorstore.as_retriever()
        return retriever

    def delete_documents_general(self, filename: str) -> None:
        """Check if a file already exists in the vector database and delete it if found."""

        sanitized_filename = self.sanitize_file_path(filename)  # Sanitize filename

        # Check if the file is already in the dataset by checking the length of the results
        for collection_name in self.settings.vectordb_idx.values():
            # Get the collection
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
            except ValueError as e:
                print(f"Error retrieving collection '{collection_name}': {e}")
                continue

            # Do the search for metadata
            results = collection.get(where={'source': sanitized_filename})

            if len(results['ids']) > 0:
                #print(f'Deleting filename: {sanitized_filename} from collection: {collection_name}')
                collection.delete(ids=results['ids'])
            else:
                print(f'No records found for filename: {filename} in collection: {collection_name}')

    def delete_documents_lancedb(self, filename: str) -> None:
        """Check if a file already exists in the LanceDB database and delete it if found."""

        sanitized_filename = self.sanitize_file_path(filename)  # Sanitize filename

        # Run a loop through the collections in the database
        for key, table in self.lancedb_client.items():
            # Delete the document based on the source keyword using SQL statement
            try:
                # Ensure proper SQL formatting, especially with strings
                table.delete(f"source = '{sanitized_filename}'")
                #print(f'Deleted document with source: {sanitized_filename} from table: {key}')
            except Exception as e:
                print(f'Failed to delete document: {sanitized_filename} from table: {key}. Error: {e}')

    def load_documents_chroma(self, documents: List[Any], document_type: int = 1) -> None:
        """add documents to the vector database
        -- 1 means text and 2 means table mode
        """

        # select the collection to use
        collection_name: str = self.settings.vectordb_idx[document_type]

        # load into chroma
        Chroma.from_documents(embedding=self.embedding_idx[self.settings.embedding_selected],
                                client=self.chroma_client, documents=documents,
                        collection_name=collection_name)

    def load_documents(self, documents: List[Any], document_type: int = 1, batch_size: int = 100) -> None:
        """add documents to the vector database
        -- 1 means text and 2 means table mode
        """

        # select the collection to use from lancedb client
        collection = self.lancedb_client[document_type]

        if document_type in self.settings.text_databases: # text mode
            # add the data to the collection
            for i in tqdm(range(0, len(documents), batch_size)):
                # define the batch of documents to add 
                batch = documents[i:i+batch_size]

                # add the batch to the collection
                collection.add(batch)
        elif document_type == 4: # multimodal mode
            # add the data to the collection
            for i in tqdm(range(0, len(documents), batch_size)):
                # define the batch of documents to add 
                batch = documents[i:i+batch_size]

                # add the batch to the collection
                collection.add(batch)
        else: # all other modes that do not require batching 
            collection.add(documents)

    def create_lancedb_fts_index(self):
        """create the lancedb fts index"""

        # create the index on each collection
        for collection, key in zip(self.lancedb_client.values(), self.settings.vectordb_idx.keys()):
            if key != 4:
                collection.create_fts_index([self.settings.indexable_field], replace=True)


# Abstract Base Class using a metaclass
class DataFrameProcessor(metaclass=ABCMeta):
    @abstractmethod
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process the DataFrame in a specific way."""
        pass


# Class for the first functionality: returning the original DataFrame
class ReturnDataFrameProcessor(DataFrameProcessor):
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Returns the original DataFrame."""
        return df


# Class for the second functionality: modifying the DataFrame based on the filename
class ModifyDataFrameProcessor(DataFrameProcessor):
    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Modifies the DataFrame based on the filename:
        
        1. If the filename is 'Egodeath public', drops specified columns.
        2. If the filename is 'ConsentResultsPublic', removes substrings in headers matching 'This is: (...)' or '(...)' along with any trailing spaces.
        3. If the filename is 'Relationships Raw Data Public', removes all '(...)' substrings from the headers along with any trailing spaces.
        4. If the filename is 'Chaos Survey Public', removes all columns containing the substring 'Randomize' and then removes all '(...)' substrings from the remaining headers.
        5. If the filename is 'correlations KINKSURVEYDONE2', removes all headers that have blank entries in all of their columns.
        6. If the filename is 'How Rationalist Are You raw data public', removes all '(...)' substrings from headers and keeps only specific columns starting from the 15th header.
        7. If the filename is 'Gender And Valence Avg Ratings', sets the first row as column headers, drops the 'thing?' column, and replaces 'x' with 1 and blanks with 0 in specific columns.
        8. If the filename is 'taboo-ratings', removes all '(...)' from column headers and keeps only odd-numbered columns starting from the 21st header.
        9. If the filename is 'Raw Gender And Valence Data', removes all columns containing the substring 'Randomize', removes all '(...)' substrings from headers starting from the 7th header, keeps the first 8 columns, and then keeps only every other column starting from the 9th.
        10. If the filename is 'Correlation Chart for the Childhood Survey (Public)', removes all '(...)' substrings from headers and sets the first column as the index.

        Args:
            df (pd.DataFrame): The original DataFrame to be processed.
            **kwargs: Additional keyword arguments. Expects 'filename' key.

        Returns:
            pd.DataFrame: The modified DataFrame with specified changes applied.
        """

        # Extract just the filename from the path and convert to lowercase
        filename = os.path.basename(kwargs.get('filename', '')).lower()
        #print('this is the filename: ', filename)

        # Check if the filename matches 'Egodeath public'
        if filename == 'egodeath_public.csv':
            # List of columns to exclude
            excluded_columns = [
                'fbclid',
                'Randomize (bxu0awp)',
                'surrealdalifacefixed',
                'Randomize (jzmxrr)',
                'dissolvefireflydarkwoods',
                's',
                'utm_source',
                'utm_medium',
                'Randomize (h3axmqx)'
            ]
            # Drop the excluded columns if they exist in the DataFrame
            df = df.drop(columns=[col for col in excluded_columns if col in df.columns])
            #print(f"Dropped columns {excluded_columns} from DataFrame for file '{filename}'.")

            # Remove '(...)' from header names
            pattern = r'\s*\([^)]*\)'  # Matches '(...)' and any preceding whitespace
            df.columns = df.columns.str.replace(pattern, '', regex=True)
            #print("Removed '(...)' from all column headers.")

            # Optionally, clean up any resulting double spaces and strip whitespace
            df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
            #print("Cleaned up spaces in column headers.")

        # Check if the filename matches 'ConsentResultsPublic'
        elif filename == 'consentresultspublic.csv':
            # Define the regex pattern to match:
            # 1. 'This is: (...)' with any trailing spaces
            # 2. '(...)' with any trailing spaces
            # The pattern uses a non-capturing group for the optional 'This is:' part
            pattern = r'(?:This is:\s*)?\(.*?\)\s*'

            # Apply the static method to clean all column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed substrings matching 'This is: (...)' or '(...)' from column headers for file '{filename}'.")

        # Check if the filename matches 'Relationships Raw Data Public'
        elif filename == 'relationships_raw data_public.csv':
            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Apply the static method to clean all column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from column headers for file '{filename}'.")

        # Check if the filename matches 'Chaos Survey Public'
        elif filename == 'chaos_survey_public.csv':
            #print("Length of the dataframe: ", len(df))
            df = df.iloc[:952]  
            #print(f"Dropped end rows. DataFrame now has {len(df)} rows.")

            # Remove all columns that contain the substring 'Randomize'
            columns_to_remove = [col for col in df.columns if 'Randomize' in col]
            df = df.drop(columns=columns_to_remove)
            #print(f"Removed columns containing 'Randomize': {columns_to_remove} from DataFrame for file '{filename}'.")

            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Apply the static method to clean all remaining column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from column headers for file '{filename}'.")

        # Check if the filename matches 'correlations KINKSURVEYDONE2'
        elif filename == 'correlations_kink_survey_females.csv' or filename == 'correlations_kink_survey_males.csv':
            # Remove all columns that have all blank entries
            # Define what constitutes a "blank" entry (e.g., NaN, empty string, etc.)
            # Here, we'll consider NaN and empty strings as blank
            # First, replace empty strings with NaN
            df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

            # Identify columns where all entries are NaN
            columns_with_all_blanks = df.columns[df.isna().all()].tolist()

            # Drop these columns
            df = df.drop(columns=columns_with_all_blanks)
            #print(f"Removed columns with all blank entries: {columns_with_all_blanks} from DataFrame for file '{filename}'.")

        # Check if the filename matches 'How Rationalist Are You raw data public'
        elif filename == 'rationalist_raw_data_public.csv':
            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Apply the static method to clean all column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from column headers for file '{filename}'.")

            # Keep the first 14 columns as they are
            columns_to_keep = list(df.columns[:14])

            # For columns starting from the 15th, keep only every other column (15, 17, 19, etc.)
            columns_to_keep.extend(df.columns[14::2])

            # Keep only the selected columns
            df = df[columns_to_keep]
            #print(f"Kept only specific columns starting from the 15th header for file '{filename}'.")

        # Check if the filename matches 'Gender And Valence Avg Ratings'
        elif filename == 'gender_and_valence_average_ratings.csv':
            # Set the first row as the column headers
            df.columns = df.iloc[0]
            # Remove the first row and reset the index
            df = df.iloc[1:].reset_index(drop=True)
            #print("Set the first row as column headers and removed it from the data.")

            # Remove the 'thing?' column if it exists
            if 'thing?' in df.columns:
                df = df.drop(columns=['thing?'])
                #print("Removed the 'thing?' column.")

            # List of columns to process
            columns_to_process = ['trait', 'sex, gender, and bodies', 'people', 'place', 'politics', 'concept/action/state?']

            # Replace 'x' with 1 and blanks with 0 in specified columns
            for col in columns_to_process:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: 1 if x == 'x' else (0 if pd.isna(x) or x == '' else x))
            #print(f"Replaced 'x' with 1 and blanks with 0 in columns: {columns_to_process}")

        # Check if the filename matches 'taboo-ratings'
        elif filename == 'taboo-ratings-v3.csv':
            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Remove the 'Position' and 's' columns
            columns_to_remove = ['Position', 's']
            df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
            #print(f"Removed columns: {columns_to_remove}")

            # Apply the static method to clean all column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from column headers for file '{filename}'.")

            # Drop 'arousalScale' and any columns containing 'Randomize'
            columns_to_drop = [col for col in df.columns if 'Randomize' in col or col == 'arousalScale']
            df = df.drop(columns=columns_to_drop)
            #print(f"Dropped 'arousalScale' and columns containing 'Randomize': {columns_to_drop}")

            # Keep the first 20 columns
            columns_to_keep = list(df.columns[:20])

            # For columns starting from the 21st, keep only every other column (21, 23, 25, etc.)
            columns_to_keep.extend([df.columns[i] for i in range(20, len(df.columns), 2)])

            # Keep only the selected columns
            df = df[columns_to_keep]
            #print(f"Kept first 20 columns and every other column starting from the 21st for file '{filename}'.")

            # Print the final column names for verification
            #print(f"Final columns: {df.columns.tolist()}")

        # Check if the filename matches 'Raw Gender And Valence Data'
        elif filename == 'raw_gender_and_valence_data_masculinity_femininity_ratings.csv':
            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Remove all columns that contain the substring 'Randomize'
            columns_to_remove = [col for col in df.columns if 'Randomize' in col]
            df = df.drop(columns=columns_to_remove)
            #print(f"Removed columns containing 'Randomize': {columns_to_remove}")

            # Clean the second header specifically
            df.columns.values[1] = self.clean_column(df.columns[1], pattern)
            #print(f"Removed '(...)' from the second header: {df.columns[1]}")

            # Clean headers starting from the 7th one
            new_columns = list(df.columns[:6])  # Keep first 6 headers unchanged (except the 2nd one which we just cleaned)
            new_columns.extend([self.clean_column(col, pattern) for col in df.columns[6:]])
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from headers starting from the 7th header for file '{filename}'.")

            # Keep the first 8 columns as they are
            columns_to_keep = list(df.columns[:8])

            # For columns starting from the 9th, keep only every other column (9, 11, 13, etc.)
            columns_to_keep.extend(df.columns[8::2])

            # Keep only the selected columns
            df = df[columns_to_keep]
            #print(f"Kept only specific columns starting from the 9th header for file '{filename}'.")

        # Check if the filename matches 'Correlation Chart for the Childhood Survey (Public)'
        elif filename == 'correlation_chart_for_childhood_survey_public.csv':
            # Define the regex pattern to match '(...)' with any trailing spaces
            pattern = r'\(.*?\)\s*'

            # Apply the static method to clean all column names
            new_columns = [self.clean_column(col, pattern) for col in df.columns]
            df.columns = new_columns
            #print(f"Removed all '(...)' substrings from column headers for file '{filename}'.")

            # Set the first column as the index
            #df.set_index(df.columns[0], inplace=True)
            #print(f"Set the first column as the index for file '{filename}'.")

        else:
            #print(f"No modifications applied. Filename '{filename}' does not match any criteria.")
            pass 

        # Apply general cleaning operations
        df = self.apply_general_cleaning(df)

        # Handle duplicate column names
        df = self.handle_duplicate_columns(df)

        return df

    @staticmethod
    def clean_column(col_name: str, pattern: str) -> str:
        """
        Removes substrings matching the given pattern from a column name,
        along with any trailing spaces.

        Args:
            col_name (str): The original column name.
            pattern (str): The regex pattern to identify substrings to remove.

        Returns:
            str: The cleaned column name.
        """
        # Remove the matched pattern
        cleaned_name = re.sub(pattern, '', col_name)
        # Strip leading/trailing spaces
        cleaned_name = cleaned_name.strip()
        return cleaned_name

    def apply_general_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply general cleaning operations to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame with reset index.
        """
        # Replace all spaces with underscores in column headers
        #df.columns = df.columns.str.replace(' ', '_')
        #print("Replaced all spaces with underscores in column headers.")

        # Remove columns with blank or NaN headers
        df = df.loc[:, df.columns.notna()]  # Remove columns with NaN headers
        df = df.loc[:, df.columns != '']    # Remove columns with empty string headers
        #print("Removed columns with blank or NaN headers.")

        # Remove columns with 'unnamed' in the header
        unnamed_columns = [col for col in df.columns if 'unnamed' in col.lower()]
        df = df.drop(columns=unnamed_columns)
        #print(f"Removed columns with 'unnamed' in the header: {unnamed_columns}")

        # Remove rows that are entirely blank or NaN
        df = df.dropna(how='all')
        #print(f"Removed {df.shape[0]} rows that were entirely blank or NaN.")

        # Remove rows where all entries are empty strings
        df = df[(df.applymap(lambda x: str(x).strip() != '') != False).any(axis=1)]
        #print(f"Removed rows where all entries were empty strings. DataFrame now has {df.shape[0]} rows.")

        # Reset the index
        df = df.reset_index(drop=True)
        #print(f"Reset DataFrame index. DataFrame now has {df.shape[0]} rows.")
        return df

    def handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate column names by appending a suffix.

        Args:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with unique column names.
        """
        # Get a list of all columns
        cols = pd.Series(df.columns)

        # For each duplicate column name
        for dup in cols[cols.duplicated()].unique():
            # Find all occurrences of the duplicate
            mask = cols == dup
            # Enumerate the duplicates
            cols[mask] = [f'{dup}_{i}' if i != 0 else dup for i in range(sum(mask))]

        # Assign new column names to the DataFrame
        df.columns = cols

        return df


class CSVAnalyzer(Utilities):
    """analyze the csv file and generate a description based on the headers - also rename illusive header names to make it easier for the model to understand what to do"""
    def __init__(self, settings, sector_settings):
        super().__init__()
        self.settings = settings

        # define the strategy for processing the csv files
        self.processor = ReturnDataFrameProcessor() # this can be changed depending on the task

        # define the sector settings
        self.sector_settings = sector_settings

        # define the model to be used for summarization
        self.model = ChatOpenAI(
            model_name=self.settings.csv_analysis_model,
            temperature=self.settings.summary_temp
        )
        self.prompt_template = ChatPromptTemplate.from_template("""
        Analyze CSV file: {num_rows} rows, {num_cols} columns.
        Stats: {column_stats}
        Headers and sample data:
        {sample_data}

        Provide concise description:
        1. Content summary
        2. Column purposes, data types
        3. Use cases, potential analyses
        4. Data quality issues

        Keywords: CSV, data analysis, column statistics, data types, use cases, data quality

        Response: Single paragraph, concise, information-dense. Include specific column names, data types, and key statistics where relevant.
        """)

        self.acronym_template_baseline = ChatPromptTemplate.from_template("""
        In the context of the {industry} industry, is the following column header an acronym: {header}?
        If it is an acronym, provide its full meaning. If it's not an acronym, respond with "Not an acronym".

        Format your response as follows:
        Is acronym: Yes/No
        If yes, full meaning: (Original Header) Full Meaning

        For example:
        For "SKU":
        Is acronym: Yes
        Full meaning: (SKU) Stock Keeping Unit

        For "Product Name":
        Is acronym: No

        Response: Determine if the header is an acronym and provide the full meaning if it is.
        """)

        self.acronym_template = ChatPromptTemplate.from_template("""
        In the context of the {industry} industry, analyze the following column header: {header}

        1. Correct any typos.
        2. Format it nicely (e.g., split any words that have become one word, capitalize appropriately).
        3. Do not change the meaning of the header.

        Format your response as follows:
        Processed header: [Your processed version or original if no changes are needed]

        Examples:
        1. Input: "revenueTotal"
           Processed header: "Revenue Total"

        2. Input: "custmer_satsfacton"
           Processed header: "Customer Satisfaction"

        Now, please analyze this header: {header}
        """)

    def manual_file_upload(self) -> Tuple[str, Dict[str, str]]:
        """
        Manually upload a CSV file for analysis.
        """
        pass

    def process_header(self, header: str, industry: str) -> str:
        """
        Process a single header to determine if it's an acronym and clarify if necessary.

        Args:
            header (str): The column header to process.
            industry (str): The industry sector for context.

        Returns:
            str: The original header if not an acronym, or the clarified version if it is.
        """
        messages = self.acronym_template.format_messages(
            industry=industry,
            header=header
        )

        try:
            output = self.model(messages)
            response = output.content.strip().lower()

            if "is acronym: yes" in response:
                # Look for the full meaning
                full_meaning_match = re.search(r'full meaning:(.+)', response, re.IGNORECASE)
                if full_meaning_match:
                    return full_meaning_match.group(1).strip()

            # If we can't find a full meaning or it's not an acronym, return the original header
            return header
        except Exception as e:
            print(f"Error processing header '{header}': {str(e)}")
            return header

    def generate_renamed_headers_metadata(self, df: pd.DataFrame, industry: str = "General") -> Dict[str, str]:
        """
        Generate metadata with renamed headers.

        Args:
            df (pd.DataFrame): The DataFrame containing the CSV data.
            industry (str): The industry sector for context.

        Returns:
            Dict[str, str]: A dictionary mapping original headers to renamed headers.
        """
        headers = df.columns.tolist()
        renamed_headers = {}
        #print(headers)
        for header in tqdm(headers):
            #print(f'processing the header: {header}')
            clarified_header = self.process_header(header, industry)
            renamed_headers[header] = clarified_header

        return renamed_headers

    def _read_and_process_csv(self, csv_file_path: str) -> pd.DataFrame:
        read_start = time.time()
        df_full = pd.read_csv(csv_file_path, nrows=20000)

        # update the dataframe with the cleaning algorithms under the processor abstraction
        df_full = self.processor.process(df_full, filename=csv_file_path)
        read_end = time.time()
       # print(f"Time to read and process CSV: {read_end - read_start:.2f} seconds")
        return df_full

    def _generate_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        stats_start = time.time()
        column_stats = self._get_column_stats(df)
        stats_end = time.time()
        #print(f"Time to generate column stats: {stats_end - stats_start:.2f} seconds")
        return column_stats

    def _create_sample_data(self, df: pd.DataFrame) -> str:
        df_sample = df.head()
        return df_sample.to_string(index=False)

    def _generate_description(self, num_rows: int, num_cols: int, column_stats: Dict[str, Dict[str, Any]], sample_data: str) -> str:
        prompt_start = time.time()
        messages = self.prompt_template.format_messages(
            num_rows=num_rows,
            num_cols=num_cols,
            column_stats=column_stats,
            sample_data=sample_data
        )
        prompt_end = time.time()
        #print(f"Time to format prompt: {prompt_end - prompt_start:.2f} seconds")

        model_start = time.time()
        try:
            output = self.model(messages)
            model_end = time.time()
            #print(f"Time for model to generate description: {model_end - model_start:.2f} seconds")
            return output.content.strip()
        except Exception as e:
            #print(f"Error in model generation: {str(e)}")
            return f"Error generating description: {str(e)}"

    def _handle_analysis_error(self, e: Exception, csv_file_path: str, start_time: float) -> Tuple[str, Dict[str, str]]:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Error occurred. Total time for {self.sanitize_file_path(csv_file_path)}: {total_time:.2f} seconds")
        return f"Error analyzing CSV file: {str(e)}", {}

    def _save_revised_csv_file(self, df: pd.DataFrame, csv_file_path: str) -> None:
        """
        Save the revised CSV file to the specified path.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            csv_file_path (str): The path to save the CSV file.
        """

        # save the dataframe to a csv file
        df.to_csv(csv_file_path, index=False)

    def generate_csv_description(self, csv_file_path: str) -> Tuple[str, Dict[str, str]]:
        """
        Generate a concise description of a CSV file using GPT model.

        Args:
            csv_file_path (str): Path to the CSV file to be analyzed.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing the description and the renamed headers metadata.
        """
        start_time = time.time()
        try:
            # read the dataframe into pandas and then clean the column names and unnecessary rows/columns
            df_full = self._read_and_process_csv(csv_file_path)

            # generate statistics about the dataframe 
            num_rows, num_cols = df_full.shape
            column_stats = self._generate_column_stats(df_full)

            # create a sample based on the top 5 items in the dataset
            sample_data = self._create_sample_data(df_full)

            # generate a description of the dataset based on the sample just created
            description = self._generate_description(num_rows, num_cols, column_stats, sample_data)

            # get the headers that are renamed based on the corrected dataframe
            renamed_headers_metadata = self.generate_renamed_headers_metadata(df_full)

            # Upload the processed CSV file to Dropbox
            self._save_revised_csv_file(df_full, csv_file_path) # saved to the original file path

            # upload the file to dropbox if the setting is enabled - KEEP DISABLED IF CLIENT DOES NOT USE DROPBOX 
            if not self.settings.disable_external_upload:
                self._upload_to_dropbox(csv_file_path)

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time for {self.sanitize_file_path(csv_file_path)}: {total_time:.2f} seconds")
            return description, renamed_headers_metadata
        except Exception as e:
            return self._handle_analysis_error(e, csv_file_path, start_time)

    def _get_column_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate detailed statistics for each column in the DataFrame.

        Args:
            df (pd.DataFrame): The full DataFrame to analyze.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing statistics for each column.
        """
        stats = {}
        for col in df.columns:
            #print('column: ', col)
            try:
                # Basic statistics for all column types
                col_stats = {
                    "dtype": str(df[col].dtype),
                    "unique_count": df[col].nunique(),
                    "null_count": df[col].isnull().sum(),
                    "sample_values": df[col].dropna().sample(min(3, df[col].nunique())).tolist() if not df[col].empty else []
                }

                # Additional statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
                    col_stats.update({
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": df[col].mean(),
                    })
                stats[col] = col_stats
            except Exception as e:
                print(f"Error processing column '{col}': {str(e)}")
                #print(df[col])
                stats[col] = {"error": str(e)}
        return stats


class FileDownloader(ABC):
    """Abstract base class for file uploaders."""

    @abstractmethod
    def download(self, file_path: str, **kwargs) -> None:
        """Download a file from the specified destination."""
        pass


class DropboxDownloader(FileDownloader):
    """Concrete implementation of FileDownloader for Dropbox."""

    def __init__(self, tool: Any, download_path_model: Any, dropbox_downloader_func: Callable):
        self.tool = tool
        self.download_path_model = download_path_model
        self.dropbox_downloader_func = dropbox_downloader_func

    def download(self, file_path: str, **kwargs) -> None:
        # get the local directory from the kwargs
        local_dir: str = kwargs.get('local_dir', None)

        if local_dir is None:
            # Check the file extension to determine the local directory for download
            if file_path.endswith('.pdf'):
                local_dir = self.download_path_model.pdf_path  # Set directory for PDF files
            elif file_path.endswith('.md'):
                local_dir = self.download_path_model.markdown_path  # Set directory for Markdown files
            elif file_path.endswith('.csv') or file_path.endswith('.xls') or file_path.endswith('.xlsx'):
                local_dir = self.download_path_model.table_path  # Set directory for table files (CSV, Excel)
            elif file_path.endswith('.json'):
                local_dir = self.download_path_model.json_path  # Set directory for JSON files
            elif file_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                local_dir = self.download_path_model.image_path  # Set directory for image files
            else:
                print("Unsupported file type. Please upload a valid file.")  # Inform user of unsupported file type
                return

        # Download the file from Dropbox to the specified local directory
        self.dropbox_downloader_func(self.tool, file_path, local_dir=local_dir)


class ManualDownloader(FileDownloader):
    """Concrete implementation of FileDownloader for manual downloads."""
    
    def __init__(self, tool: Any):
        self.tool = tool

    def download(self, file_path: str = '', **kwargs) -> None:
        self.tool(file_path, **kwargs)


class MultiModalDocumentProcessor(VectorDB):
    """
    This class is responsible for processing documents for multi-modal vector databases.
    Handles batch processing of images and their embeddings.
    """

    def __init__(self):
        super().__init__()

    def format_processed_batch(self, embeddings_batch: list, base64_images: list, file_path: str) -> list:
        """
        Process a batch of images using Voyage AI's multimodal embedding model
        
        Args:
            batch_data: List of tuples containing (idx, (image, base64_image))
            file_path: Source file path
            
        Returns:
            list: Processed batch data with embeddings
        """
        # define the container for the formatted batch
        formatted_batch = []

        # iterate through the embeddings batch and format the data
        for page_number, embedding in enumerate(embeddings_batch):
            # convert the embedding to a numpy array of floats
            embedding = np.array(embedding[0], dtype=np.float32)
            
            # follow the schema of the document database
            formatted_batch.append(
                {
                'vector': embedding,
                'page_image_bytes': base64_images[page_number],  # Assuming base64_images is indexed by page number
                'source': file_path,
                'page': page_number + 1
                }
            )
        return formatted_batch

    @staticmethod
    def render_image_from_bytes(byte_string: bytes) -> Image:
        """
        Render an image from a byte string.

        Args:
            byte_string (bytes): The byte string representing the image.

        Returns:
            Image: A PIL Image object.
        """
        # open the image from the byte string
        image = Image.open(io.BytesIO(byte_string))
        return image

    # Helper function to process a batch of images through the multimodal embedding model
    def create_embeddings(self, batch):
        """
        Create embeddings for a batch of images.

        Args:
            batch (list): A list of tuples, where each tuple contains:
                - page_number (int): The page number of the image
                - image (PIL Image): The image to be processed

        Returns:
            list: A list of tuples containing (embedding, page_number)
        """
        # Extract images from batch - no need for nested lists
        images = [[image[1]] for image in batch]  # Just get the images
        pages = [image[0] for image in batch]   # Get the page numbers

        # Get embeddings using Voyage AI's multimodal model
        # The model expects a list of images, not a list of lists
        result = self.embedding.multimodal_embed(
            inputs=images,  # Pass images directly, not as nested lists
            model="voyage-multimodal-3"
        )

        # Pair embeddings with their page numbers
        return list(zip(result.embeddings, pages))
    
    def generate_multi_modal_embeddings(self, batches: list, base64_images: list, file_path: str) -> list:
        """
        Generate multi modal documents using concurrent batch processing.
        
        Args:
            images: List of PIL images
            base64_images: List of base64 encoded images
            file_path: Source file path
            
        Returns:
            list: Document dictionaries with embeddings
        """
       
        # define the container for the embeddings
        all_embeddings = []

        # Use ThreadPoolExecutor to run tasks concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Map the process_inputs function to the input_batches
            results = list(executor.map(self.create_embeddings, batches))

        # Combine embeddings from all batches
        all_embeddings = []
        for embeddings in results:
            all_embeddings.extend(embeddings)

        # format the batch
        formatted_batch = self.format_processed_batch(all_embeddings, base64_images, file_path)

        return formatted_batch

    def encode_single_image(self, idx_and_image):
        """
        Encode a single image and return the base64 string.
        """
        idx, image = idx_and_image
        try:
            # Create a copy of the image to prevent concurrent access issues
            image_copy = image.copy()
            base64_str = self.get_base64_image(image_copy)
            
            # Debug print to check uniqueness
            #print(f"Image {idx} hash: {hash(base64_str[:100])}")
            
            return idx, base64_str
        except Exception as e:
            #print(f"Error encoding image {idx}: {str(e)}")
            return idx, None

    def create_batch_base64_images(self, images: list) -> list:
        """
        Create base64 encodings for a batch of images using concurrent processing.
        
        Args:
            images: List of PIL images to encode.
            
        Returns:
            list: List of base64 encoded images.
        """
        base64_images = [None] * len(images)  # Pre-allocate list with correct size

        # Create list of (index, image) pairs
        indexed_images = list(enumerate(images))

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=50) as executor:
            # Submit tasks for each image
            futures = [executor.submit(self.encode_single_image, (idx, img)) 
                      for idx, img in indexed_images]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    idx, base64_str = future.result()
                    if base64_str is not None:
                        base64_images[idx] = base64_str
                except Exception as e:
                    print(f"Error processing future: {str(e)}")

        # Remove any None values from failed encodings
        base64_images = [img for img in base64_images if img is not None]
        
        # Verify uniqueness of results
        unique_hashes = len(set(hash(b[:100]) for b in base64_images))
        print(f"Number of unique encodings: {unique_hashes} out of {len(base64_images)}")

        return base64_images

    def document_pdf_image_embeddings(self, file_path: str, scaling: bool = False) -> list:
        """
        Process PDF files, treating each page as a separate document and image.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            list: Processed documents with embeddings.
        """
        # Construct full file path using the download path model
        full_path = self.download_path_model.pdf_path + file_path
        print('Processing file:', full_path)

        # Convert each page of the PDF to images
        images = convert_from_path(full_path, fmt='jpg', thread_count=8)

        # Scale the images to a specific width
        if scaling:
            images = [self.scale_image(image, 1024) for image in images]

        # Get the base64 encoding of the images using concurrent processing
        base64_images = self.create_batch_base64_images(images)

        # create the batches of images with their indices
        batches = [[(j+1, img) for j, img in enumerate(images[i:i + self.settings.embedding_batch_size], start=i)] for i in range(0, len(images), self.settings.embedding_batch_size)]
     
        # generate the embeddings
        return self.generate_multi_modal_embeddings(batches, base64_images, full_path)


class DataLoader(MultiModalDocumentProcessor):
    """
    Load the data from a specific folder.

    This class is responsible for managing the loading of various document types, including:
    - **CSV Files**: Analyzed and processed for data extraction.
    - **PDF Files**: Each page is treated as a separate document for detailed processing.
    - **Markdown Files**: Loaded and split for further analysis.

    Key functionalities include:
    - **Document Processing**: Supports multiple vector database types (e.g., LanceDB, Chroma).
    - **File Management**: Handles file downloads from Dropbox and organizes them appropriately.
    - **Error Handling**: Robust mechanisms to manage exceptions during file processing.

    Ensure to configure the settings for optimal performance based on the document types being processed.
    """

    def __init__(self):
        super().__init__()

        # define the csv analyzer
        self.csv_analyzer = CSVAnalyzer(self.settings, self.sector_settings)

        # create the setting for the document processor
        self.document_processor: dict = {
            'lancedb': [self._process_documents_specialized, self._process_csv_chunking_document],
            'chroma': [self._process_documents_generic],
        }
        self._process_documents = self.document_processor[self.settings.vectordb_type]

        # create the mapping for document deletion from the databases
        self.document_deletion: dict = {
            'lancedb': self.delete_documents_lancedb,
            'chroma': self.delete_documents_general
        }
        self.delete_documents = self.document_deletion[self.settings.vectordb_type]

        # define the loading schemas of each type of file
        try:
            # define the pipeline index
            self.pipeline_idx: dict = dict(zip(
                self.settings.dirs, [self._table_pipeline, self._markdown_pipeline, self._text_pipeline_multimodal]
            ))
        except (KeyError, ValueError):
            raise Exception("Length of pipelines index does not match directory index")
        
        # define the uploader
        self.download_file_idx = {
            'auto': DropboxDownloader(self.dbx, self.download_path_model, self.download_from_dropbox),
            'manual': ManualDownloader(self.manual_file_manager)
        }
        # define the downloader based on the upload method and settings configuration
        self.downloader = self.download_file_idx[self.settings.upload_method]

    def process_pdf(self, file_path):
        # Process each recent file
        if '.pdf' not in file_path:
            return
        print(file_path)
        # Download the file from Dropbox
        metadata, res = self.dbx.files_download(file_path)

        # Convert the content to a byte stream
        pdf_stream = io.BytesIO(res.content)

        # Load the PDF content directly from the byte stream using PyMuPDF
        document = fitz.open(stream=pdf_stream, filetype="pdf")

        # Create a list to hold the Document objects
        documents = []
        #print(document.page_count)
        # Extract text from the document and create Document objects
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text = page.get_text('text').replace('\n', ' ')
            doc = Document(
                page_content=text,
                metadata={
                    "page": int(page_num + 1), # impose the integer type on the page
                    "source": self.sanitize_file_path(file_path) #self.format_file_name(file_path)
                }
            )
            documents.append(doc)
        #print(len(documents))
        return documents

    def process_markdown(self, file_path):
        print(file_path)
        # Check if the file is a Markdown file
        if not file_path.endswith('.md'):
            return

        # Download the file from Dropbox
        metadata, res = self.dbx.files_download(file_path)

        # Convert the content to a byte stream
        file_content = res.content.decode('utf-8')

        return [Document(
            page_content=str(file_content),
            metadata={
                'source': self.sanitize_file_path(file_path), #self.format_file_name(file_path)
                'page': 1, # impose the integer type on the page
            }
        )]

    def process_csv(self, file_path):
        """load in a csv file from dropbox"""

        # check file extension if it fits the requirements
        print(file_path)
        if not file_path.endswith('.csv'):
            return

        metadata, res = self.dbx.files_download(file_path)

        # Use io.BytesIO to create a file-like object
        file_like_object = io.BytesIO(res.content)

        # Read the content of the CSV into a pandas DataFrame
        df = pd.read_csv(file_like_object)

        # Save the DataFrame to a CSV file-like object for CSVLoader
        df.to_csv(file_path, index=False)

        # Use the custom CSV loader to load data
        loader = CSVLoader(file_path)
        data = loader.load()
        os.remove(file_path)
        return data

    def _create_ids_arr(self, length: int = 0):
        """define an array of unique ids to be used in database indexing"""
        return [str(self.generate_random_id()) for _ in range(length)]

    def create_splitter(self) -> Union[Callable, dict]:
        """create the splitting mechanism"""

        # define the container with types of splitters
        splitter_idx: Dict[str, Any] = {
            'recursive': {'pdf': RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                                chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap),
                        'table': None,
                        'markdown': MarkdownHeaderTextSplitter(headers_to_split_on=self.settings.markdown_headers_to_split_on)}, # find a splitter in the future
            'contextual': SemanticChunker(self.embedding)}
        return splitter_idx[self.settings.splitter_mech_type]

    def _process_documents_generic(self, docs: List[Any], mode: int = 0) -> List[list]:
        """split open the document where mode is either table or text"""

        # define the new splits
        new_splits: list = []
        new_metadata: list = []
        new_ids: list = self._create_ids_arr(len(docs))

        # iterate through each raw document and clean it out
        for document_raw in docs:
            try:
                document_raw.page_content = str(self.clean_text(str(document_raw.page_content)))
                document_raw.metadata = {'souce': self.sanitize_file_path(document_raw.metadata['source']), 'page': document_raw .metadata['page']} # clean the source file path to standardized
                new_splits.append(document_raw)
                #new_splits.append(str(self.clean_text(str(document_raw.page_content))))
                #new_metadata.append(document_raw.metadata)
            except AttributeError:
                print("document missing page content during processing")

        return new_splits

    def _process_documents_specialized(self, docs: List[Any], mode: int = 0) -> List[dict]:
        """process the documents for lancedb purposes..."""

        # process the documents for ingest into LanceDB and specialized vector DBs
        new_splits: list = []
        new_ids: list = self._create_ids_arr(len(docs))

        # Iterate through each raw document and clean it out
        for document_raw in docs:
            #print(document_raw.keys())
            try:
                # Determine whether to clean the text based on the mode
                if mode == 0:  # For text documents (PDFs, etc.)
                    page_content = str(self.clean_text(str(document_raw.page_content)))
                else:  # For tables or markdown (mode == 1)
                    try:
                        page_content = str(document_raw.page_content)
                    except AttributeError:
                        page_content = str(document_raw['page_content'])

                # Create the dictionary with the required keys
                new_splits.append({
                    'page_content': str(page_content),
                    'page': int(document_raw.metadata.get('page', 1)),  # Default to 1 if not found and impose the integer type
                    'source': self.sanitize_file_path(document_raw.metadata.get('source', 'unknown')),  # Default to 'unknown' if not found
                    'location_on_page': document_raw.metadata.get('location_on_page', 'full'),  # Default to 'full' if not found
                    'additional_info': document_raw.metadata.get('additional_info', str({}))  # Default to empty JSON string if not found
                })

            except AttributeError:
                print("Document missing page content during processing")

        return new_splits

    def _process_csv_chunking_document(self, docs: List[Any], mode=1) -> List[dict]:
        """Process CSV documents for LanceDB purposes, using CSV chunks as the 'page' content"""

        new_splits: list = []
        new_ids: list = self._create_ids_arr(len(docs))

        for document_raw in docs:
            try:
                # Create the dictionary with the required keys
                new_splits.append({
                    'page_content': str(document_raw.metadata.get('page', 'none given')),  # Placeholder for page_content
                    'page': int(document_raw.metadata.get('page', 100000000)), #document_raw.page_content,  # CSV data as JSON string
                    'source': self.sanitize_file_path(document_raw.metadata.get('source', 'unknown')),
                    'location_on_page': document_raw.metadata.get('location_on_page', 'full'),
                    'additional_info': document_raw.page_content #document_raw.metadata.get('additional_info', str({}))
                })

            except AttributeError:
                print("Document missing required attributes during CSV chunk processing")

        return new_splits
    
    def process_table_descriptions(self, description: str, file_path: str, renamed_headers: Dict[str, str]) -> List[dict]:
        """Process table descriptions for input into LanceDB"""

        # Read the CSV file to get additional information
        try: 
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f'File not found: {file_path}, returning empty list')
            return []

        # Create the document with additional information
        document = {
            'page_content': str(description),
            'source': self.sanitize_file_path(file_path),  # Use the full original file path
            'page': 1,  # Assuming it's always page 1 for CSV files and impose the integer type
            'location_on_page': 'full',  # Indicating it covers the full page
            'additional_info': json.dumps({
                'renamed_headers': renamed_headers,
                'original_headers': df.columns.tolist(),
                'row_count': len(df),
                'column_count': len(df.columns), 
                'full_dataframe': df.to_dict(orient='records')
            }
        )
        }
        return [document]

    def _embedded_table_reader(self, file_path: str) -> list:
        """process the tables that are part of the pipeline"""

        # read in the pdf file to analyze the tables
        dfs: list = tabula.read_pdf(file_path, pages='all')

        # define the metadatas
        metadatas: list = [{'source': self.sanitize_file_path(file_path)} for _ in range(len(dfs))]

        # define the ids
        ids: list = self._create_ids_arr(len(dfs))
        return dfs

    def _documents_pdf_reader(self, file_path: str = "", ocr_mode: bool = False):
        """read the documents in a particular PDF file"""

        # check if ocr mode is on 
        if not ocr_mode:
            try: 
                # define the loader from langchain default 
                loader = PyPDFLoader(f'{self.download_path_model.pdf_path}{file_path}')

                # load the documents
                docs = loader.load()
            except FileNotFoundError:
                print(f'File not found: {self.download_path_model.pdf_path}{file_path}, returning empty list')
                docs: list = []
        else:
            # define the pdf parser
            parser = DocumentParser(
                table_args={
                    "parsing_algorithm": "pymupdf",
                    "table_output_format": "markdown"
                }
            )

            # extract the documents from pdf page with the parser and standardize to langchain
            docs: list = self.standardize_openparse_to_langchain(parser.parse(f'{self.download_path_model.pdf_path}{file_path}'))

        return docs

    def _markdown_reader(self, file_path: str = ""):
        """process the markdown documents before entering the rest of the pipeline"""

        # define the document loader
        #docs = self.process_markdown(file_path) #UnstructuredMarkdownLoader(file_path)
        docs = UnstructuredMarkdownLoader(f'{self.download_path_model.markdown_path}{file_path}')

        # define the documents from the splitter
        #docs = loader.load()
        return docs

    def _text_pipeline_vanilla(self, file_path: str) -> None:
        """main function to process the TEXT files as they enter the system"""

        # Download the file from Dropbox into the pdf/text folder
        self.downloader.download(file_path, download=True)

        # load in the documents by page level
        docs_raw: list = self._documents_pdf_reader(file_path)

        # define the splitting mechanism
        splitter_tool: Any = self.create_splitter()['pdf']

        # create the splits in each of the pages of equal token size 
        all_splits = splitter_tool.split_documents(docs_raw)

        # process each of the splits by cleaning the text and putting it into the correct format for the vector database
        docs = self._process_documents[0](all_splits)

        # delete the file or move it to the manual upload directory
        if not self.settings.retain_files_condition:
            self.delete_file(f'{self.download_path_model.pdf_path}{file_path}')
        #else: 
        #    print('in move file manually')
        #    self.manual_move_file_back(file_path, local_dir=self.download_path_model.pdf_path)

        # load the documents into the vector store, document type 1 is for pdf/text
        self.load_documents(docs, document_type=1, batch_size=self.settings.embedding_batch_size)      

    def _text_pipeline_multimodal(self, file_path: str) -> None:
        """process the text pipeline with multimodal embeddings"""

        # Download the file from Dropbox into the pdf/text folder
        self.downloader.download(file_path, download=True)

        # generate the multimodal embeddings and the container for lancedb table entry 
        container = self.document_pdf_image_embeddings(file_path, scaling=False)

        # delete the file 
        if not self.settings.retain_files_condition:
            self.delete_file(f'{self.download_path_model.pdf_path}{file_path}')

        # load the documents into the vector store, document type 1 is for pdf/text
        self.load_documents(container, document_type=4, batch_size=self.settings.embedding_batch_size)
    
    def whole_table_splitter(self, file_path: str) -> List[Any]:
        """split the table into chunks"""

        # wait until the csv file is available in the directory
        print(file_path)
        while not os.path.exists(file_path):
            time.sleep(1)  # wait for 1 second before checking again

        # read in the csv file
        try: 
            df = pd.read_csv(file_path, chunksize=self.settings.csv_chunk_size)
        except FileNotFoundError:
            print(f'File not found: {file_path}, returning empty list')
            return []

        # Process the CSV file in chunks
        docs = []
        for chunk_num, chunk in enumerate(df):
            # Convert the chunk to a list of dictionaries
            chunk_data = chunk.to_dict(orient='records')

            # Create a document for this chunk
            doc = Document(
                page_content=str(chunk_data),  # Convert the list of dicts to a string
                metadata={
                    'source': self.sanitize_file_path(str(file_path)),
                    'page': int(chunk_num + 1),  # Use chunk number as page number and impose integer type
                    'location_on_page': 'full',
                    'additional_info': json.dumps({
                        'chunk_number': int(chunk_num + 1),
                        'num_columns': len(chunk.columns),
                        'num_rows': len(chunk),
                        'column_names': chunk.columns.tolist()
                    })
                }
            )
            docs.append(doc)

        return docs

    def _table_pipeline(self, file_path: str) -> None:
        """main function to process the TABLE files as they enter the system"""

        # Download the file from Dropbox
        print(f'Current File in Processing: {file_path}')
        self.downloader.download(file_path, download=True)

        # Load in the documents
        docs_raw = self.whole_table_splitter(f'{self.download_path_model.table_path}{file_path}')

        # Process the raw documents which goes straight into the raw rows database with chunking
        docs = self._process_documents[1](docs_raw, mode=1)  # mode=1 for table data

        # Process the CSV descriptions
        description_doc, renamed_headers = self.csv_analyzer.generate_csv_description(f'{self.download_path_model.table_path}{file_path}')

        # Process the written description of the tables, without the chunking taking in the description generated from before
        description_docs = self.process_table_descriptions(description_doc, f'{self.download_path_model.table_path}{file_path}', renamed_headers)

        # Delete the local file
        if not self.settings.retain_files_condition:
            self.delete_file(f'{self.download_path_model.table_path}{file_path}')

        # Load the documents into the vector store
        self.load_documents(docs, document_type=2)
        self.load_documents(description_docs, document_type=3)  # Assuming document_type=3 for table descriptions

    def _markdown_pipeline(self, file_path: str):
        """load in the markdown files and split them"""

        # download the file from dropbox into the markdown folder
        self.downloader.download(file_path, download=True)

        # load in the documents with langchain document loader
        docs_raw: list = self._markdown_reader(f'{self.download_path_model.markdown_path}{file_path}')

        # define the splitting ool
        splitter_tool: Any = self.create_splitter()['pdf'] # use this with the dropbox loader

        # create the splits
        all_splits = splitter_tool.split_documents(docs_raw)

        # split them into metadatas, contents, and ids
        docs = self._process_documents[0](all_splits)

        # delete the file
        if not self.settings.retain_files_condition:
            self.delete_file(file_path)

        # load the documents into the vector database
        self.load_documents(docs, document_type=1) # insert them into the pdf database

    def loader(self, files: list, func: Callable) -> None:
        """define the functionality for loading documents into the vector database"""

        # keep track of failed files 
        failed_files: list = []

        # loop through each file
        with ThreadPoolExecutor(max_workers=2) as executor:
            # create the future object for all the files
            futures = {executor.submit(func, file_path): file_path for file_path in files}
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'File {file_path} generated an exception: {exc}')
                    failed_files.append(file_path)
            
        # retry with a simple for loop 
        for failed_file in failed_files:
            try: 
                func(file_path)
            except: 
                print(f'File {failed_file} failed to load AGAIN')

    def postprocess_vectordb(self):
        """add any post processing to the vector database creation such as indexing or computing general stats"""

        # create fts indexes on search fields for lancedb setup
        if self.settings.vectordb_type == 'lancedb':
            self.create_lancedb_fts_index()

    def facade(self):
        """run for all PDF and CSV files in the directory"""

        # get all the recent files in the directory from the root
        recent_files: list = self.get_recent_files(dbx=self.dbx)

        # group the files by their file types
        file_type_idx: dict = self.group_files_by_type(recent_files)
        
        # handle file deletion by checking if file is already in database and if yes then delete it and update the whole thing
        deleted_files: list = self.list_deleted_files(dbx=self.dbx)
        print(deleted_files)
        for del_file in [file for sublist in [deleted_files, recent_files] for file in sublist]:
            print('file to delete: ', del_file)
            self.delete_documents(del_file)

        # define the directories that we will loop over
        for file_type, files in file_type_idx.items():
            try:
                self.loader(files, self.pipeline_idx[file_type])
            except KeyError:
                print(f"File type not found: {file_type}")

        # postprocess the database
        self.postprocess_vectordb()


class VespaColPaliSettings(BaseModel):
    """Settings for the ColPali model"""
    
    colpali_schema: Schema = Schema(
        name="pdf_page",
        document=VespaDocument(
            fields=[
                Field(name="id", type="string", indexing=["summary", "index"], match=["word"]),
                Field(name="url", type="string", indexing=["summary", "index"]),
                Field(name="title", type="string", indexing=["summary", "index"], match=["text"], index="enable-bm25"),
                Field(name="page_number", type="int", indexing=["summary", "attribute"]),
                Field(name="image", type="raw", indexing=["summary"]),
                Field(name="text", type="string", indexing=["index"], match=["text"], index="enable-bm25"),
                Field(
                    name="embedding",
                    type="tensor<int8>(patch{}, v[16])",
                    indexing=["attribute", "index"], # adds HNSW index for candidate retrieval.
                    ann=HNSW(distance_metric="hamming", max_links_per_node=32, neighbors_to_explore_at_insert=400),
                )
            ]
        ),
        fieldsets=[FieldSet(name="default", fields=["title", "text"])]
    )

    # define the name of the vespa application
    vespa_app_name: str = "jarviscolpali"

     # define the parameters for the model and processor 
    device_map: str = 'auto'
    torch_type: torch.dtype = torch.bfloat16
    model_name: str = "vidore/colpali-v1.2"

    # define the batch size for the data loader in torch, the larger the batch size the more memory intensive
    data_loader_batch_size: int = 2
    runtime_batch_size: int = 2

    # hyperparameter for inference operations
    target_hits_per_query_tensor: int = 20
    max_query_timeout: int = 120
    presentation_timing: bool = True
    max_inference_hits: int = 3

    # allow arbitrary types in definitions 
    class Config:
        arbitrary_types_allowed = True


class ColPaliLoader(Utilities): 
    """Implementation of loading in images for colpali functionality with Vespa"""

    def __init__(self):
        super().__init__()
        
        # Define the settings for the vespa application
        self.colpali_settings = VespaColPaliSettings()
        
        # Device and type selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.type = self.colpali_settings.torch_type
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        #    self.type = torch.float32
        else:
            self.device = torch.device("cpu")
            self.type = torch.float32
        
        # Define the model and processor and setup the model
        self.model = None
        self.processor_colpali = None
        self.setup_colpali_model()

        # define the colpali schema 
        self.colpali_schema = None

        # define the vespa application
        self.vespa_app = self._initialize_vespa_application()

    def deploy_vespa_application_docker(self, vespa_application_package):
        """
        Deploy the Vespa application using Docker.

        Args:
            vespa_application_package (ApplicationPackage): The Vespa application package to deploy.

        Returns:
            Vespa: A Vespa application instance that can be used to interact with the deployed application.
        """
        # Create a VespaDocker instance for deploying the application
        vespa_docker = VespaDocker()

        # Deploy the application package using Docker
        # This step creates a Docker container running the Vespa application
        app = vespa_docker.deploy(application_package=vespa_application_package)

        # Return the Vespa application instance
        return app

    def deploy_vespa_application_cloud(self, vespa_application_package):
        """Deploy the Vespa application using the cloud"""
        # Replace with your tenant name from the Vespa Cloud Console
        tenant_name = "jarvis1" 

        # Get the API key from environment variable
        key = os.getenv("VESPA_TEAM_API_KEY", None)
        if key is not None:
            key = key.replace(r"\n", "\n")  # To parse key correctly

        # Create VespaCloud instance
        vespa_cloud = VespaCloud(
            tenant=tenant_name,
            application=self.colpali_settings.vespa_app_name,
            key_content=key,  # Key is only used for CI/CD testing. Can be removed if logging in interactively
            application_package=vespa_application_package,
        )

        # Deploy the application
        app = vespa_cloud.deploy()

        return app
            
    def _initialize_vespa_application(self): 
        """Create the Vespa application"""
        
        # get the colpali schema from the settings model 
        self.colpali_schema = self.colpali_settings.colpali_schema

        # Create an ApplicationPackage instance for Vespa
        vespa_application_package = ApplicationPackage(
            # Set the name of the Vespa application from the settings
            name=self.colpali_settings.vespa_app_name,
            
            # This schema determines the structure of documents in the Vespa application
            schema=[self.colpali_settings.colpali_schema]
        )
        
        # add the reranker schema to the vespa application package
        self.colpali_schema = self._create_colpali_reranker_schema(self.colpali_schema)

        # deploy the vespa application
        vespa_app = self.deploy_vespa_application_docker(vespa_application_package)
        return vespa_app

    def _create_colpali_reranker_schema(self, colpali_schema):
        """
        Create the reranker schema for the ColPali model.
        
        Args:
            colpali_schema (Schema): The existing Vespa schema to add the rank profile to.
        
        Returns:
            Schema: The updated Vespa schema with the new rank profile.
        """
        # Define input query tensors
        input_query_tensors = []
        MAX_QUERY_TERMS = 64  # Maximum number of query terms to consider
        
        # Create tensors for each query term
        for i in range(MAX_QUERY_TERMS):
            # Each query term is represented as a 16-dimensional int8 tensor
            input_query_tensors.append((f"query(rq{i})", "tensor<int8>(v[16])"))

        # Add a tensor for the full query representation
        # This is a float tensor with dimensions (number of query tokens, 128)
        input_query_tensors.append(("query(qt)", "tensor<float>(querytoken{}, v[128])"))
        
        # Add a binary tensor representation of the query
        # This is an int8 tensor with dimensions (number of query tokens, 16)
        input_query_tensors.append(("query(qtb)", "tensor<int8>(querytoken{}, v[16])"))

        # Create the rank profile
        colpali_retrieval_profile = RankProfile(
            name="retrieval-and-rerank",
            inputs=input_query_tensors,
            functions=[
                Function(
                    name="max_sim",
                    expression="""
                        sum(
                            reduce(
                                sum(
                                    query(qt) * unpack_bits(attribute(embedding)) , v
                                ),
                                max, patch
                            ),
                            querytoken
                        )
                    """,
                ),
                Function(
                    name="max_sim_binary",
                    expression="""
                        sum(
                          reduce(
                            1/(1 + sum(
                                hamming(query(qtb), attribute(embedding)) ,v)
                            ),
                            max,
                            patch
                          ),
                          querytoken
                        )
                    """,
                )
            ],
            first_phase=FirstPhaseRanking(expression="max_sim_binary"),
            second_phase=SecondPhaseRanking(expression="max_sim", rerank_count=10)
        )

        # Add the rank profile to the schema
        print(colpali_schema)
        colpali_schema.add_rank_profile(colpali_retrieval_profile)

        return colpali_schema

    def setup_colpali_model(self) -> None:
        """Set up the ColPali model and processor"""
        try:
            # Initialize the processor
            self.processor_colpali = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.colpali_settings.model_name))
            
            # Initialize the model
            self.model = cast(
                ColPali,
                ColPali.from_pretrained(
                    self.colpali_settings.model_name,
                    torch_dtype=self.colpali_settings.torch_type if torch.cuda.is_available() else torch.float32,
                    device_map=self.colpali_settings.device_map,  # Adjust as needed
                )
            )
            
            # Move the model to the device
            self.model.to(self.device)
            
            print("ColPali model and processor initialized successfully.")
        except Exception as e:
            print(f"Error initializing ColPali model and processor: {str(e)}")

    @staticmethod
    def download_pdf(url):
        """
        Download a PDF file from a given URL.

        Args:
            url (str): The URL of the PDF file to download.

        Returns:
            BytesIO: A byte stream containing the PDF content.

        Raises:
            Exception: If the download fails or the status code is not 200.
        """
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # If successful, return the content as a BytesIO object
            return BytesIO(response.content)
        else:
            # If the request failed, raise an exception with the status code
            raise Exception(f"Failed to download PDF: Status code {response.status_code}")

    def get_pdf_images(self, file_path: str):
        """Get the images and text from a PDF file."""
        
        # Initialize a PDF reader
        reader = PdfReader(file_path)
        
        # Initialize a list to store text from each page
        page_texts = []
        
        # Extract text from each page of the PDF
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            text = page.extract_text()
            page_texts.append(text)
        
        # Convert each page of the PDF to an image
        images = convert_from_path(file_path)
        
        # Ensure that the number of images matches the number of text extractions
        assert len(images) == len(page_texts), "Mismatch between number of images and text extractions"
        
        # Clean up: delete the local file after processing
        self.delete_file(file_path)
        
        # Return both the list of images and the list of page texts
        return images, page_texts

    def ingest_document_feed(self, app, vespa_feed: list):
        """Ingest the document feed into the Vespa application"""
        async def _ingest():
            async with self.vespa_app.asyncio(connections=1, total_timeout=180) as session:
                for page in vespa_feed:
                    response: VespaResponse = await session.feed_data_point(
                        data_id=page['id'], fields=page, schema="pdf_page"
                    )
                    if not response.is_successful():
                        print(response.json())
        
        asyncio.run(_ingest())

    def prepare_vespa_feed(self, processed_pdfs):
        """Prepare the Vespa feed from processed PDFs."""
        vespa_feed = []
        for pdf in tqdm(processed_pdfs, desc="Preparing Vespa feed"):
            url = pdf['url']
            title = pdf.get('title', 'Untitled')  # Use 'Untitled' if title is not provided
            for page_number, (page_text, embedding, image) in enumerate(zip(pdf['texts'], pdf['embeddings'], pdf['images'])):
                # Scale the image and convert it to base64
                base_64_image = self.get_base64_image(self.scale_image(image, 640), add_url_prefix=False)
                
                # Prepare the embedding dictionary
                embedding_dict = {}
                for idx, patch_embedding in enumerate(embedding):
                    # Convert the embedding to a binary vector and then to a hex string
                    binary_vector = np.packbits(np.where(patch_embedding > 0, 1, 0)).astype(np.int8).tobytes().hex()
                    embedding_dict[str(idx)] = binary_vector  # Convert idx to string for JSON compatibility
                
                # Create a dictionary for this page
                page = {
                    "id": hash(url + str(page_number)),  # Create a unique ID for the page
                    "url": url,
                    "title": title,
                    "page_number": page_number,
                    "image": base_64_image,
                    "text": page_text,
                    "embedding": embedding_dict
                }
                
                # Add the page to the Vespa feed
                vespa_feed.append(page)
        
        # Return the complete Vespa feed
        return vespa_feed
    
    def _pdf_image_pipeline(self, file_path: str):
        """load in the pdf files and convert them to images"""

        # download the file from dropbox
        self.download_from_dropbox(self.dbx, file_path, local_dir=self.download_path_model.pdf_path)

        # Construct the full local file path
        local_file_path = os.path.join(self.download_path_model.pdf_path, os.path.basename(file_path))

        # get the images from the pdf
        images, page_texts = self.get_pdf_images(local_file_path)
        
        # delete the file
        self.delete_file(local_file_path)

        # return the images and page texts
        return images, page_texts
        
    def _create_pdf_embeddings(self, processed_pdfs: list) -> list:
        """Create the embeddings for the PDFs"""
        for pdf in tqdm(processed_pdfs, desc="Creating PDF embeddings"):
            page_embeddings = []
            images = pdf['images']  # Assuming this is a list of PIL.Image objects

            # Create a DataLoader over images
            dataloader = TorchDataLoader(
                images,
                batch_size=self.colpali_settings.data_loader_batch_size,
                shuffle=False,
                collate_fn=lambda batch: self.processor_colpali.process_images(batch),
            )

            # process the pages and iterate over the dataloader object
            for batch_doc in tqdm(dataloader, desc="Processing pages", leave=False):
                # Move batch to device
                batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}

                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(**batch_doc)

                # Collect embeddings
                page_embeddings.extend(list(torch.unbind(outputs.to("cpu"))))

            # Store the embeddings in the processed_pdfs list
            pdf['embeddings'] = page_embeddings

        return processed_pdfs

    def _process_pdf_batch(self, file_paths: list) -> None:
        """Process a batch of PDF files"""
        processed_pdfs = []

        for file_path in tqdm(file_paths, desc="Processing PDFs"):
            try:
                # Use the existing _pdf_image_pipeline function
                images, page_texts = self._pdf_image_pipeline(file_path)

                # Create a dictionary for the processed PDF
                processed_pdf = {
                    'url': file_path,  # Original Dropbox path
                    'images': images,
                    'texts': page_texts
                }

                processed_pdfs.append(processed_pdf)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

        # create the embeddings for each page
        processed_pdfs = self._create_pdf_embeddings(processed_pdfs)

        # prepare the vespa feed
        vespa_feed = self.prepare_vespa_feed(processed_pdfs)

        # ingest the vespa feed
        self.ingest_document_feed(vespa_feed)

    def facade(self) -> None: 
        """The facade method for running the document ingest and embedding creation"""

        # Get all the pdf files in the dropbox
        pdf_files = self.get_recent_files(self.dbx, file_type='pdf')
      
        # Use ThreadPoolExecutor to process batches of files concurrently
        with ThreadPoolExecutor(max_workers=self.colpali_settings.runtime_batch_size) as executor:
            for i in range(0, len(pdf_files), self.colpali_settings.runtime_batch_size):
                batch = pdf_files[i:i+self.colpali_settings.runtime_batch_size]
                executor.submit(self._process_pdf_batch, batch)


class ColPaliInference(ColPaliLoader):
    """Inference for the ColPali model"""

    def __init__(self):
        super().__init__()

    def _process_inference_query(self, queries: list):
        """
        Process the inference query using the ColPali model.

        Args:
            query (str): The input query string.

        Returns:
            list: A list of query embeddings.
        """
        # Create a DataLoader for the query
        dataloader = DataLoader(
            queries,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor_colpali.process_queries(x),
        )

        query_embeddings = []

        for batch_query in tqdm(dataloader, desc="Processing query"):
            with torch.no_grad():
                # Move batch to the appropriate device
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                
                # Generate embeddings
                embeddings_query = self.model(**batch_query)
                
                # Move embeddings to CPU and add to the list
                query_embeddings.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        return query_embeddings

    def _prepare_query_tensors(self, qs):
        """
        Prepare float and binary query embeddings and tensors.

        Args:
            qs (list): List of query embeddings.

        Returns:
            dict: Prepared query tensors.
        """
        # Convert the first query embedding to a list and enumerate it
        float_query_embedding = {k: v.tolist() for k, v in enumerate(qs[0])}
        
        # Initialize an empty dictionary for binary query embeddings
        binary_query_embeddings = {}
        
        # Convert float embeddings to binary (0 or 1) and then to packed int8
        for k, v in float_query_embedding.items():
            # Convert positive values to 1 and non-positive to 0
            binary = np.where(np.array(v) > 0, 1, 0)
            # Pack the binary values into 8-bit integers
            packed = np.packbits(binary)
            # Convert to int8 and then to a list
            binary_query_embeddings[k] = packed.astype(np.int8).tolist()

        # Prepare the query tensors dictionary
        query_tensors = {
            "input.query(qtb)": binary_query_embeddings,  # Binary query embedding
            "input.query(qt)": float_query_embedding      # Float query embedding
        }
        
        # Add individual binary embeddings as separate tensors
        for i in range(len(binary_query_embeddings)):
            query_tensors[f"input.query(rq{i})"] = binary_query_embeddings[i]

        return query_tensors

    def _implement_nn_algo(self, query_tensors, target_hits_per_query_tensor):
        """
        Implement the nearest neighbor algorithm using Vespa.

        Args:
            query_tensors (dict): The query tensors.
            target_hits_per_query_tensor (int): The target number of hits per query tensor."""

        # construct the nn query
        nn = [f"({{targetHits:{target_hits_per_query_tensor}}}nearestNeighbor(embedding,rq{i}))" 
                  for i in range(len(query_tensors) - 2)]  # -2 to exclude 'qtb' and 'qt'
        
        # join the nn queries with OR
        nn = " OR ".join(nn)
        return nn 

    def _process_hits(self, query: str, response: VespaQueryResponse):
        """
        Process the hits from the Vespa query response.

        Args:
            query (str): The original query string.
            response (VespaQueryResponse): The response from Vespa.

        Returns:
            dict: A dictionary containing the query and processed hits.
        """

        # create the container for the results
        results = {
            "query": query,
            "hits": []
        }
        
        for hit in response.hits:
            # Decode the image bytes
            image_bytes = base64.b64decode(hit['fields']['image'])

            # Append the processed hit to the results
            results["hits"].append({
                "title": hit['fields']['title'],
                "url": hit['fields']['url'],
                "page_number": hit['fields']['page_number'],
                "image_bytes": image_bytes,
                "relevance": hit['relevance']
            })
        
        return results

    async def inference(self, query: str):
        """
        Perform inference using the ColPali model and query Vespa.

        Args:
            query (str): The input query string.

        Returns:
            dict: A dictionary containing query results and image byte strings.
        """

        # Process the query to get embeddings
        qs = self._process_inference_query([query])

        async with self.vespa_app.asyncio(connections=1, total_timeout=180) as session:
            # Prepare query tensors for Vespa
            query_tensors = self._prepare_query_tensors(qs)

            # Implement nearest neighbor algorithm for Vespa query
            nn = self._implement_nn_algo(query_tensors, self.colpali_settings.target_hits_per_query_tensor)

            # Execute Vespa query
            response: VespaQueryResponse = await session.query(
                yql=f"select title, url, image, page_number from pdf_page where {nn}",
                ranking="retrieval-and-rerank",
                timeout=self.colpali_settings.max_query_timeout,
                hits=self.colpali_settings.max_inference_hits,
                body={
                    **query_tensors,
                    "presentation.timing": self.colpali_settings.presentation_timing
                }
            )

            # Check if the query was successful
            assert response.is_successful()
            
            # Process the hits from the Vespa response
            results = self._process_hits(query, response)
            
            return results  

    def run_inference(self, query: str):
        """
        Facade method to run inference on a given query.

        Args:
            query (str): The input query string.

        Returns:
            dict: A dictionary containing query results and image byte strings.
        """
        # Run the inference asynchronously
        loop = asyncio.get_event_loop()

        # Run the inference asynchronously
        return loop.run_until_complete(self.inference(query))

    def display_query_results(self, query, response, hits=5):
        """
        Display the query results in HTML format and save to a file.

        Args:
            query (str): The original query string.
            response (VespaQueryResponse): The response from Vespa.
            hits (int): Number of top results to display. Defaults to 5.
        """
        # Extract query time from the response and round to 2 decimal places
        query_time = response.json.get('timing', {}).get('searchtime', -1)
        query_time = round(query_time, 2)

        # Get the total count of results
        count = response.json.get('root', {}).get('fields', {}).get('totalCount', 0)

        # Start building the HTML content with query information
        html_content = f'<h3>Query text: \'{query}\', query time {query_time}s, count={count}, top results:</h3>'

        # Iterate through the top hits and add their details to the HTML content
        for i, hit in enumerate(response.hits[:hits]):
            title = hit['fields']['title']
            url = hit['fields']['url']
            page = hit['fields']['page_number']
            image = hit['fields']['image']
            score = hit['relevance']

            # Add a header for each result
            html_content += f'<h4>PDF Result {i + 1}</h4>'

            # Add title, URL, page number, and relevance score
            html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'

            # Add the image (assumed to be base64 encoded)
            html_content += f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'

        print('Code 200: Savings Results to HTML')

        # Append the HTML content to a file named 'results.html'
        with open('results.html', "a") as file:
            file.write(html_content)


data_loader = DataLoader()
#multimodal_obj = MultiModalDocumentProcessor()
#data_loader.facade()
#data_loader._table_pipeline('/Products_by_producer_distributor.csv')
#data_loader._text_pipeline_multimodal('/book_test.pdf')
#data_loader.postprocess_vectordb()
#filename = '/content/taboo-ratings-v3.csv'
#data_loader.csv_analyzer.generate_csv_description(filename)
#data_loader.refresh_access_token()
#data_loader.csv_analyzer._upload_to_dropbox('/content/tables/consentresultspublic.csv')

# test out the Colpali loader inference
#colpali_loader = ColPaliInference()
#colpali_loader._pdf_image_pipeline(['/book_test.pdf'])

#data_loader.move_file_manually('/book_test.pdf', source_dir=data_loader.download_path_model.table_path, local_dir=data_loader.download_path_model.coding_path)


print('done')