import spacy
import pandas as pd
import numpy as np
import json
import os
import shutil
import uuid
import warnings
import io
import fitz
import dropbox
import bs4
import base64
from time import perf_counter
import tabula
from abc import ABC, abstractmethod, ABCMeta
from functools import partial
from tqdm import tqdm
from typing import Any, Annotated, Callable, Dict, List, Sequence, TypedDict, ClassVar, Union
import re
from functools import lru_cache
from openparse import processing, DocumentParser
from time import sleep
from pprint import pprint
from locale import getpreferredencoding
import hashlib
from collections import OrderedDict
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field, validator
from langchain_core.pydantic_v1 import BaseModel as LangchainBaseModel, Field as LangchainField
from dropbox.files import ListFolderResult, DeletedMetadata, FileMetadata
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings as ChromaSettings
import whisper

from langchain import hub
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import (
    GPT4AllEmbeddings, HuggingFaceEmbeddings, LlamaCppEmbeddings, OpenAIEmbeddings
)
from langchain_community.vectorstores import Chroma, LanceDB
from langchain_community.chat_models import ChatOllama
from langchain.retrievers import EnsembleRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.csv_loader import CSVLoader

#from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI # not that azureopenai is only gpt-3.5-turbo integration
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import END, StateGraph

# import libaries for the hybrid search
import Stemmer

from ragatouille import RAGPretrainedModel
from langchain.load import dumps, loads
from langchain_text_splitters import MarkdownHeaderTextSplitter
from operator import itemgetter
from langchain.prompts.chat import ChatPromptTemplate
from flashrank import Ranker, RerankRequest
import cohere

# import lancedb libraries and dependencies
import lancedb
from lancedb.embeddings import get_registry, EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker
import sys
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    pass
try:
    from backend.setup import DocumentsSchema, DataLoader, api_key, co, RAG_colbert_downloaded, azure_api_key, azure_endpoint
    from backend.setup import ColPaliLoader, ColPaliInference # colpali imports
    from backend.chat_history_management import ChatHistoryManagement
except:
    try:
        from setup import DocumentsSchema, DataLoader, api_key, co, RAG_colbert_downloaded, azure_api_key, azure_endpoint
        from setup import ColPaliLoader, ColPaliInference # colpali imports
        from chat_history_management import ChatHistoryManagement
    except Exception as e:
        print(f"An error occurred while importing setup: {e}")
from langchain.callbacks.base import BaseCallbackHandler
import requests


@lru_cache(maxsize=1)
def get_data_loader():
    """Cached data loader initialization"""
    data_loader = DataLoader()
    return data_loader


class MyStreamHandler(BaseCallbackHandler):
    def __init__(self, url: str, start_token=""):
        self.text = start_token
        self.url = url

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text = token
        try:
            response = requests.post(self.url, json={"output": self.text})
        except requests.exceptions.RequestException as e:
            print(f"Failed to send data to {self.url}: {e}")


class ModelSpecifications(BaseModel):
    """base model for model parameters specifications"""

    fusion_template: str = """You are an expert AI research assistant specializing in Retrieval Augmented Generation (RAG) systems. Your task is to generate optimized search queries to improve information retrieval.

    Original Question (please read twice for thorough understanding):
    {question}

    Instructions:
    1. Carefully analyze the original question, considering its context, nuances, and potential implications.
    2. Generate 2 search queries related to the original question, adhering to these guidelines:
      - The first query should be an optimized version of the original question.
      - Make each query specific, detailed, and designed to retrieve highly relevant information.
      - Avoid using any placeholders or brackets.
      - Vary the focus of each query to cover different aspects or perspectives of the topic.
      - Use precise language and include key terms that are likely to appear in relevant documents.

    Output Format:
    Provide only the generated queries, one per line, without any numbering, explanations, or additional text.
    """

    model_id: str = 'gpt-4o' # gpt-4o-mini
    model_deployment_name: str = 'jarvis_llm'
    private_llm_mode: bool = True
    temperature: float = 0
    max_tokens: int = 1024
    # prompt: Any = prompt
    main_rag_template2: str = """Pretend you are a strategist/business analyst. Your goal is to provide actionable insights based on data. Provide deep analyis and infer beyond the data without making anything up. Be elaborate. Base your answer on this context:
                Textual Context:
                {text_context}
                Tabular Context:
                {table_context}

                Omit any filler text from the prompt in your final answer.

                Here is the question that was answered.
                Question: {question}
                Answer:
              """
    main_rag_template: str = """You are an analyst on a technical topic. Your goal is to provide insight that is detailed and accurate as possible based on data provided. Also, give analysis on the answer. Base your answer on this context:
                Textual Context:
                {text_context}
                Tabular Context:
                {table_context}

                Also make sure to use any content in the images provided by the user as additional context to the question they are asking.

                Omit any filler text from the prompt in your final answer.

                Here is the question:
                Question: {question}
                Answer:
              """
    chat_memory_threshold: int = 50

    # define the generation type for the citations
    # Options include {'multimodal', 'langchain'}
    citation_generation_type: str = Field(
        default='multimodal',
        description="Type of citation generation to use. Must be either 'multimodal' or 'langchain'."
    )

    @validator('citation_generation_type')
    def validate_citation_type(cls, v):
        allowed_values = {'multimodal', 'langchain'}
        if v not in allowed_values:
            raise ValueError(f'citation_generation_type must be one of {allowed_values}')
        return v

    # New concatenation prompt added as a string
    concatenate_template: str = """You are an AI assistant tasked with consolidating multiple responses that originate from different chunks of the same spreadsheet into a single, coherent answer.

    Original Question:
    {question}

    Individual Responses:
    {responses}

    Instructions:
    - Combine the individual responses into one comprehensive answer.
    - Ensure the final answer flows logically and covers all key points from the individual responses.
    - Maintain clarity and coherence in the combined response.
    - Remember that these responses come from different chunks of the same spreadsheet, so integrate the information accordingly.

    Combined Answer:
    """

    multimodal_rag_template_debugging: str = """You are an expert research assistant that knows technical topics. Here is the question: {question}. Answer is clearly and concisely."""
    # multimodal retrieval-augmented generation template
    multimodal_rag_template: str = """You are an expert research assistant specializing in various technical topics. Your goal is to provide detailed, accurate analysis by carefully reasoning about both textual and visual information.

    Approach this task using the following steps:

    1. UNDERSTAND THE CONTEXT:
    - Carefully analyze any images provided
    - Note key visual elements, colors, spatial relationships, and text
    - Consider both explicit and implicit information

    2. REASONING PROCESS:
    - Break down complex questions into simpler components
    - Consider multiple perspectives and potential interpretations
    - Look for relationships between different elements
    - Apply common sense reasoning to fill gaps

    3. EVIDENCE GATHERING:
    - Identify relevant information from both visual and textual sources
    - Cross-reference details across multiple sources when available
    - Note any uncertainties or assumptions made

    4. SYNTHESIS:
    - Combine evidence from all sources
    - Explain your reasoning step by step
    - Support conclusions with specific references
    - Address any potential contradictions

    Original Question:
    {question}

    Let's solve this step by step:
    1) First, let's understand what we're looking for...
    2) Next, let's examine the available evidence...
    3) Now, let's reason about the connections...
    4) Finally, let's form a complete answer...

    Based on this analysis, here is my response:
    """

    # define the parameters for multimodal inference
    low_resolution_img: bool = True
    resize_img_dim: int = 800


class LanceTableRetriever(BaseRetriever):
    """A retriever that contains the top 3 table documents from LanceDB."""

    table: Any
    reranker: Any
    k: int
    mode: str

    def _translate_lancedb_to_langchain(self, lancedb_documents, **kwargs) -> List[Document]:
        """Translate LanceDB documents to LangChain Documents before generation stage of RAG, filtering duplicates."""

        langchain_documents: list = []
        seen_identifiers = set()  # Initialize a set to track unique documents

        for lancedb_doc in lancedb_documents:
            # Use 'additional_info' as the page_content
            if not kwargs['disable_tabular']:
                page_content = getattr(lancedb_doc, 'page_content', None)
            else:
                page_content = getattr(lancedb_doc, 'additional_info', None)

            # Initialize metadata dictionary
            metadata = {}
            # Add all other fields except 'page_content' and 'additional_info' to metadata
            for key, value in lancedb_doc.__dict__.items():
                if key not in ['page_content']:
                    metadata[key] = value

            # Create a unique identifier for deduplication
            # You can adjust the fields used here based on your data
            unique_string = f"{metadata.get('source', '')}_{metadata.get('page', '')}_{page_content[:100]}"
            identifier = hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

            # Check if the identifier has been seen before
            if identifier in seen_identifiers:
                # Duplicate found, skip adding this document
                continue
            else:
                # Add the identifier to the set
                seen_identifiers.add(identifier)

            # Create the LangChain Document
            langchain_doc = Document(
                page_content=page_content,
                metadata=metadata
            )
            langchain_documents.append(langchain_doc)

        return langchain_documents

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs) -> List[Document]:
        """Retrieve documents from LanceDB by filtering without semantic search.

        Args:
            query (str): The query string.
            run_manager (CallbackManagerForRetrieverRun): Callback manager.
            **kwargs: Additional keyword arguments. Expected key:
                - chunks (str): Comma-separated string of page numbers, e.g., "1, 2, 3".

        Returns:
            List[Document]: List of retrieved and processed documents.
        """
        # Extract 'chunks' from kwargs with a default value if not provided
        chunks = kwargs.get('chunks', '1, 2, 3')  # Default chunks

        # get the disable tabular argument from the kwargs and return empty list if tabular is disabled
        disable_tabular = kwargs.get('disable_tabular', False)

        if disable_tabular:
            return []

        if not chunks:
            raise ValueError("The 'chunks' parameter is required and was not provided.")

        # Validate and sanitize the 'chunks' input, ensure it contains only numbers and commas
        if not re.match(r'^(\d+\s*,\s*)*\d+$', chunks):
            raise ValueError("The 'chunks' parameter must be a comma-separated string of integers, e.g., '1, 2, 3'.")

        # Perform the filter based on the 'chunks' condition
        try:
            documents = self.table.search().where(f"page in ({chunks})").to_pydantic(DocumentsSchema)
            #print(f"The number of documents found: {len(documents)}")
        except Exception as e:
            print(f"An error occurred while searching: {e}")
            documents = []

        # Postprocess the documents with deduplication
        results = self._translate_lancedb_to_langchain(documents, **kwargs)
        return results


class LanceRetrieverMultiModal(BaseRetriever):
    """A retriever that contains the top 3 table documents from LanceDB."""

    # define the specific retriever to use
    table: Any
    k: int
    mode: str
    embedding: Any
    vector_column_name: str

    def _process_documents(self, documents: list) -> List[Document]:
        """Process the documents to be returned"""

        # define the container for the documents
        container: list = []

        # loop through the documents and add them to the container
        for document in documents:
            container.append(
                {
                    'page_image_bytes': document['page_image_bytes'],
                    'source': document['source'],
                    'page': document['page'],
                    'distance': document['_distance']
                }
            )

        return container

    def search(self, query: Any, top_k: int = 2):
        """Search for similar documents using vector similarity search.

        Args:
            query (str): The query text to search for
            top_k (int, optional): Number of results to return. Defaults to 2.

        Returns:
            list: Top k most similar documents as a list
        """
        # get the query embedding with the multimodal embedding model
        vec_q = query #self.get_query_embedding([query])

        # perform the search
        r = self.table.search(vec_q, query_type="vector",
                              vector_column_name=self.vector_column_name).limit(top_k)

        # return the results in JSON format
        return r.to_list()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""

        # Perform the retrieval based on the query
        hit_lists = self.search(query, top_k=self.k)

        return self._process_documents(hit_lists)


class LanceRetriever(BaseRetriever):
    """A toy retriever that contains the top k documents that contain the user query.

    This retriever only implements the sync method _get_relevant_documents.

    If the retriever were to involve file access or network access, it could benefit
    from a native async implementation of `_aget_relevant_documents`.

    As usual, with Runnables, there's a default async implementation that's provided
    that delegates to the sync implementation running on another thread.
    """

    # define the specific retriever to use
    table: Any
    reranker: Any
    k: int
    mode: str

    def _translate_lancedb_to_langchain(self, lancedb_documents) -> List[Document]:
        langchain_documents: list = []

        for lancedb_doc in lancedb_documents:
            # Extract the page content
            page_content = getattr(lancedb_doc, 'page_content', None)

            # Extract metadata dynamically
            metadata = {}

            # Loop through each attribute of the LanceDB document to populate metadata
            for key, value in lancedb_doc.__dict__.items():
                if key != 'page_content':  # We exclude page_content as it goes directly to the LangChain document
                    metadata[key] = value

            # Create a LangChain Document object with dynamic metadata
            langchain_doc = Document(
                page_content=page_content,
                metadata=metadata
            )

            # Append to the list of LangChain documents
            langchain_documents.append(langchain_doc)

        return langchain_documents

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Sync implementations for retriever."""

        # do the retrieval and querying
        try:
            documents = self.table.search(
                str(query),
            query_type=self.mode,
            ).rerank(reranker=self.reranker).limit(self.k).to_pydantic(DocumentsSchema)
        except KeyError:
            return []
        # postprocess the documents
        return self._translate_lancedb_to_langchain(documents)


# define the output JSON parser
class RAGResponse(LangchainBaseModel):
    response: str = LangchainField(description="textual RAG response")


class CitationGeneration(metaclass=ABCMeta):
    """abstract base class for citation generation"""

    @abstractmethod
    def generate_citations(self, documents: list, source_field: str = 'source') -> str:
        """generate citations from the documents' metadata sources."""
        pass


class CitationGenerationLangChain(CitationGeneration):
    """implementation of the citation generation abstract base class"""

    def generate_citations(self, documents: list, source_field: str = 'source') -> str:
        """generate citations from the documents' metadata sources assuming the documents are LangChain Documents"""
        try:
          # Use OrderedDict to maintain insertion order and group by source
          sources = OrderedDict()

          for doc in documents:
              if hasattr(doc, 'metadata') and source_field in doc.metadata:
                  source = doc.metadata[source_field]
                  page = doc.metadata.get('page', 'N/A')

                  if source not in sources:
                      sources[source] = set()
                  sources[source].add(str(page))

          # Format the citations
          citations = []
          for i, (source, pages) in enumerate(sources.items(), 1):
              # Sort pages and join them
              sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
              page_str = ', '.join(sorted_pages)

              citation = f"{i}) Retrieved from {source}. Page(s): {page_str}"
              citations.append(citation)

          # Join the citations with newlines
          return "\n\n".join(citations)
        except TypeError:
          return 'Retrieved from LLM'


class CitationGenerationMultiModal(CitationGeneration):
    """implementation of the citation generation abstract base class"""

    def generate_citations(self, documents: list, source_field: str = 'source') -> str:
        """generate citations from the documents' metadata sources."""
        # Use OrderedDict to maintain insertion order and group by source
        try:
          sources = OrderedDict()

          for doc in documents:
              source = doc[source_field]
              page = doc.get('page', 'N/A')

              if source not in sources:
                  sources[source] = set()
              sources[source].add(str(page))

          # Format the citations
          citations = []
          for i, (source, pages) in enumerate(sources.items(), 1):
              # Sort pages and join them
              sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else float('inf'))
              page_str = ', '.join(sorted_pages)

              citation = f"{i}) Retrieved from {source}. Page(s): {page_str}"
              citations.append(citation)

          # Join the citations with newlines
          return "\n\n".join(citations)
        except TypeError:
          return 'Retrieved from LLM'


class RAGModelUtilities:
    """A utility class for managing Retrieval-Augmented Generation (RAG) models.

    This class is responsible for initializing and configuring the model specifications,
    handling chat history management, and managing the streaming of responses from the language model.
    It includes functionality for generating queries based on user input,
    maintaining a count of query generations, and managing citation generation through different strategies.
    The class also provides methods for cleaning user queries and updating the stream URL for the
    language model, ensuring that the model can adapt to different environments and configurations.
    """

    def __init__(self, stream_url: str = 'http://127.0.0.1:8005/stream'):
        self.model_specs: BaseModel = ModelSpecifications()
        self.chat_history_management = ChatHistoryManagement()
        self.stream_url = stream_url
        self.stream_handler = MyStreamHandler(url=self.stream_url, start_token="")  # Initialize stream_handler as None

        # define the data loader which contains the vector database, the settings, and the utilities
        self.data_loader = get_data_loader()  # Use the cached data loader

        # define the baseline model
        self.baseline_model = self.data_loader.initialize_llm(self.stream_handler, self.model_specs)

        # define the query generation counter
        self.query_generation_count = 0

        # define the dictionary of citation generation classes
        self.citation_generation_dict = {
            'langchain': CitationGenerationLangChain(),
            'multimodal': CitationGenerationMultiModal()
        }

        # define the prompts
        self.rag_fusion_prompt, self.main_rag_prompt, self.concatenation_prompt = self._create_rag_templates()

    def update_stream_url(self, new_url: str):
        """Update stream_url and reinitialize LLM with new handler"""
        self.stream_url = new_url
        #self.baseline_model = self.data_loader.initialize_llm()

    def clean_query(self, query: str) -> str:
        """Clean the query by removing invalid formatting characters."""
        # Remove square brackets and their content
        cleaned_query = re.sub(r'\[.*?\]', '', query)
        # Optionally, remove other unwanted characters
        cleaned_query = cleaned_query.replace('[', '').replace(']', '')
        return cleaned_query

    def reciprocal_rank_fusion(self, results: list[list], k=100):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        return reranked_results

    def get_chat_history_from_management(self, user_id: int, model_id: int, chat_id: int, chat_memory_threshold: int = 2):
        """Get and format the chat history from the chat history management."""
        chat_hist_result = self.chat_history_management.get_chat_history(user_id=user_id, model_id=model_id, chat_id=chat_id)

        # Process and format the chat history
        formatted_chat_history = []
        for question, answer, timestamp in chat_hist_result:
            formatted_entry = f"Q: {question}\nA: {answer}\n"
            formatted_chat_history.append(formatted_entry)

        return formatted_chat_history[-chat_memory_threshold:]

    def add_chat_history_to_management(self, user_id: int, model_id: int, chat_id: int, question_answer_pair: tuple) -> None:
        """Add the chat history to the chat history management."""
        print('question: ', question_answer_pair[0])
        print('answer: ', question_answer_pair[1])
        self.chat_history_management.save_message(user_id=user_id, model_id=model_id, chat_id=chat_id, question=question_answer_pair[0], answer=question_answer_pair[1])

    def _create_rag_templates(self):
        """Define the templates used in RAG fusion framework and concatenation"""

        # Define the prompt for RAG fusion by creating multiple questions
        prompt_rag_fusion = ChatPromptTemplate.from_template(self.model_specs.fusion_template)

        # Define the main RAG prompt
        prompt_main_rag = ChatPromptTemplate.from_template(self.model_specs.main_rag_template)

        # Define the concatenation prompt
        prompt_concatenation = ChatPromptTemplate.from_template(self.model_specs.concatenate_template)

        return prompt_rag_fusion, prompt_main_rag, prompt_concatenation

    @lru_cache(maxsize=100)
    def generate_queries(self, question: str) -> List[str]:
        # Increment the counter and log the generation
        self.query_generation_count += 1

        # Define the chain which generates the queries
        generate_queries: Any = (
            self.rag_fusion_prompt
            | self.baseline_model
            | StrOutputParser()
            | (lambda x: [line.strip() for line in x.split("\n") if line.strip()])
            | RunnableLambda(lambda queries: [self.clean_query(q) for q in queries])
        )
        result = generate_queries.invoke({"question": question})
        #print(f'results for generated queries: {result}')
        return result

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


class TextModel(RAGModelUtilities):
    """define the main model class with the chain for the textual side of the model"""

    def __init__(self, stream_url: str = 'http://127.0.0.1:8005/stream'):
        super().__init__(stream_url=stream_url)

        # define the model id
        self.model_id: int = 1

        # define the retrievers for the text and table databases (LanceDB in this case)
        self.text_retriever, self.table_retriever = self._create_lancedb_retrievers()

        # define the question asked by the user, defined globally in the class
        self.question: Union[str, None] = None

        # define whether to accept vision in the text model
        self.use_vision, self.image_bytes = False, None

        # define the reranking algorithm for post-retrieval processing
        self.ranker = Ranker()

        # define the output parser
        self.output_parser = JsonOutputParser(pydantic_object=RAGResponse)

        # define the chat history (only for debugging purposes)
        self.chat_history: list = []

        # define the citation generation class
        self.citation_generator = self.citation_generation_dict[self.model_specs.citation_generation_type]

        #self.query_generation_count = 0

    def _process_streamed_response(self, response: str) -> str:
        """Process the streamed response to display chunks cleanly"""
        global streamed_response  # Use the global streamed_response

        streamed_response = ""  # Reset before processing a new response

        for chunk in response:
            chunk_content = chunk.content
            streamed_response += chunk_content

            if chunk_content.strip():
                print(chunk_content, end='', flush=True)  # Print without extra newlines

        return streamed_response

    def _call_llm(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        """Call the LLM with the prompt and stream the response."""
        if kwargs.get('image_input', None) is None:
            message_content = [{"type": "text", "text": prompt.format(question=kwargs['question'],
                                                                      text_context=kwargs['text_context'],
                                                                      table_context=kwargs['table_context'],
                                                                      history=kwargs['history'])}]
            messages = [HumanMessage(content=message_content)]
            response = self.baseline_model.stream(messages)
            return {"response": self._process_streamed_response(response)}
        else:
            message_content = [{"type": "text", "text": prompt.format(question=kwargs['question'],
                                                                      text_context=kwargs['text_context'],
                                                                      table_context=kwargs['table_context'],
                                                                      history=kwargs['history'])}]
            for image_bytes in kwargs['image_input']:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}
                })

            messages = [HumanMessage(content=message_content)]
            response = self.baseline_model.stream(messages)
            return {"response": self._process_streamed_response(response)}

    def count_documents_in_collection(self, collection_name: str):
        """count the number of documents in the chromadb collection"""

        # get the collection
        collection = self.chroma_client.get_collection(collection_name)
        documents = collection.get()
        num_documents = len(documents['documents'])
        return num_documents

    def _create_lancedb_retrievers(self):
        """create the retrievers for the LanceDB database"""

        # define the reranker used in all tables
        reranker = reranker = LinearCombinationReranker(
            weight=self.data_loader.settings.hybrid_search_ratio,
        )

        # Initialize the custom retriever object for the TEXT retriever
        text_retriever = LanceRetriever(
            table=self.data_loader.lancedb_client[1],
            reranker=reranker,
            k=self.data_loader.settings.k_top,  # Ensure k is set
            mode='hybrid'  # Ensure mode is set
        )

        # Initialize the custom retriever object for the TABLE retriever
        table_retriever = LanceTableRetriever(
            table=self.data_loader.lancedb_client[2],
            reranker=reranker,
            k=self.data_loader.settings.k_top * self.data_loader.settings.k_tab_multiplier,  # Ensure k is set
            mode='fts'  # Ensure mode is set
        )

        return text_retriever, table_retriever

    def _create_retrievers(self):
        """create the retriever for the model to be used within the chain"""

        # create the chroma integration with langchain
        text_chroma = Chroma(client=self.chroma_client,
                             collection_name=self.data_loader.settings.vectordb_idx[1],
                             embedding_function=self.embedding)
        table_chroma = Chroma(client=self.chroma_client,
                              collection_name=self.data_loader.settings.vectordb_idx[2],
                              embedding_function=self.embedding)

        # get the number of documents and select the k for the retriever
        text_k = min(self.count_documents_in_collection(self.data_loader.settings.vectordb_idx[1]), self.data_loader.settings.k_top)
        table_k = min(self.count_documents_in_collection(self.data_loader.settings.vectordb_idx[2]), self.data_loader.settings.k_top)

        # create the retrievers
        retriever_text_chroma = text_chroma.as_retriever(**{"k": text_k})
        retriever_table_chroma = table_chroma.as_retriever(**{"k": table_k})

        # Define the retriever hash table
        retriever_idx = {
            'chroma': [retriever_text_chroma, retriever_table_chroma],
            'colbert': []
        }

        # if the database is colbert
        if self.data_loader.settings.vectordb_type == 'colbert':
            if self.colbert_text_col is not None:
                colbert_retriever_text = self.colbert_text_col.as_langchain_retriever(**self.data_loader.settings.colbert_params)
                retriever_idx['colbert'].append(colbert_retriever_text)
            if self.colbert_table_col is not None:
                colbert_retriever_table = self.colbert_table_col.as_langchain_retriever(**self.data_loader.settings.colbert_params)
                retriever_idx['colbert'].append(colbert_retriever_table)

        return retriever_idx[self.data_loader.settings.vectordb_type]

    def cohere_reranker(self, results: list) -> list:
        """cohere reranker used to speed up the reranking process"""

        # reformulate the documents to be used on the cohere reranker algo
        passages: list = []

        for idx, docs in enumerate(results):
            for doc in docs:
                try:
                    passages.append(
                        {
                            "id": idx + 1,
                            "text": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                except AttributeError:
                    continue

        # calculate k
        k = min(len(passages), self.data_loader.settings.reranked_top)
        #print(passages)

        # if the list is empty we want to avoid issues with the reranker
        if not k:
            return []

        # Call the Cohere rerank API
        response = co.rerank(
            model='rerank-english-v3.0',
            query=self.question,
            documents=passages,
            top_n=k,
            # return_documents=True,
        )
        #print(response.results)

        # Map reranked texts back to their metadata
        reranked_documents = []
        for reranked_result in response.results:
            document = passages[reranked_result.index]
            reranked_documents.append(
                Document(
                    page_content=document['text'],
                    metadata=document['metadata']
                )
            )

        return reranked_documents

    def colbert_reranker(self, results: list[list]):
        """the colbert reranker"""
        # print(results)
        # calculate k
        k = min(len(results), self.data_loader.settings.reranked_top)

        # do the reranking
        try:
            reranked_docs = RAG_colbert_downloaded.rerank(query=self.question, documents=results, k=k)
            # print(reranked_docs)
            return reranked_docs
        except:
            return []

    def flashrank_reranker(self, results: list[list], k=50):
        """the flashrank reranker"""

        # reformat the documents and passages
        passages: list = []
        for idx, docs in enumerate(results):
            for doc in docs:
                try:
                    passages.append(
                        {
                            "id": idx + 1,
                            "text": doc.page_content,
                            "metadata": doc.metadata
                        }
                    )
                except AttributeError:
                    continue
        #print(passages)
        # calculate k
        k = min(len(passages), self.data_loader.settings.reranked_top)

        # if the list is empty we want to avoid issues with the reranker
        if not k:
            return []

        # perform reranking with flashrank
        rerank_request = RerankRequest(query=self.question, passages=passages)
        reranked_docs = self.ranker.rerank(rerank_request)[:k]

        # reformat the documents to standardize the workflow
        reranked_docs = self.process_flashrank_responses(reranked_docs)
        # print(reranked_docs)
        return reranked_docs

    def _create_final_chain(self, chat_history: list, **kwargs) -> Any:
        """Create the final RAG chain, passing the 'chunks' argument to the table retriever."""

        # Define the fusion chain with reranking mechanism for the text retriever
        retrieval_text_rag_fusion = (
            RunnableLambda(lambda inputs: self.generate_queries(inputs["question"]))
            | self.text_retriever.map()
            | self.cohere_reranker
        )

        # Get the 'chunks' argument from **kwargs or set a default
        chunks = kwargs.get('chunks', '1, 2, 3')  # Default chunks if not provided

        # Validate the 'chunks' format
        if not re.match(r'^(\d+\s*,\s*)*\d+$', chunks):
            raise ValueError("The 'chunks' parameter must be a comma-separated string of integers, e.g., '1, 2, 3'.")

        # Define the partial function for the table retriever passing in the chunks argument and disable_tabular argument
        partial_table_retriever_func = partial(self.table_retriever._get_relevant_documents, chunks=chunks, disable_tabular=kwargs.get('disable_tabular', False))

        # Define the fusion chain for the table retriever, passing 'chunks'
        retrieval_table_rag_fusion = RunnableLambda(partial_table_retriever_func)

        # Define the entire chain
        final_rag_chain = (
            {
                "text_context": retrieval_text_rag_fusion,
                "table_context": retrieval_table_rag_fusion,
                "question": RunnablePassthrough(),
                "history": lambda x: chat_history
            }
            | RunnableLambda(lambda x: {
                **x,
                "combined_context": x["text_context"] + x["table_context"],
                "text_context": self.data_loader.format_docs_reranked_v2(x["text_context"]),
                "table_context": self.data_loader.format_docs_reranked_v2(x["table_context"])
            })
            | RunnableLambda(lambda x: {
                **x,
                "llm_response": self._call_llm(
                    prompt=self.main_rag_prompt,
                    question=x["question"],
                    text_context=x["text_context"],
                    table_context=x["table_context"],
                    history=x["history"],
                    image_input=kwargs.get('image_input', None)
                )
            })
            | RunnableLambda(lambda x: {
                "response": self.data_loader.format_output(x['llm_response']['response']),
                # "combined_context": x["combined_context"]
                "references": self.citation_generator.generate_citations(x["combined_context"], source_field=kwargs['source_field']),
            })
            # | RunnableLambda(lambda x: {
            #     "response": x["response"],
            #     "references": self.generate_citations(x["combined_context"], source_field=kwargs['source_field']),
            #     # 'combined_context': x['combined_context']
            # })
        )

        return final_rag_chain

    def _create_chain(self, chat_history: list, **kwargs):
        """Perform the RAG fusion operation by generating more contextual questions regarding a prompt"""

        # Define the rest of your chain here...
        final_rag_chain = self._create_final_chain(chat_history, **kwargs)

        return final_rag_chain

    def invoke(self, question: str, chat_memory_threshold_default: int = 2, source_field: str = 'source', **kwargs) -> Dict[str, Any]:
        """Call the chain and process the output.

        Args:
            question (str): The question to ask.
            image_input (Any): The image input to use as base64 encoded string.
            **kwargs: Additional keyword arguments. Expected key:
                - chunks (str): Comma-separated string of page numbers, e.g., "1, 2, 3".

        Returns:
            Dict[str, Any]: The response from the text model.
        """

        # Clean the question
        self.question = self.clean_query(question)

        # get the user id, model_id, and chat_id
        user_id, chat_id = self.get_query_auth(**kwargs)

        try:
            # self.stream_url = f"http://5.78.113.143:8005/update_stream/{self.user_id}/{self.chat_id}"
            new_stream_url = f"http://5.78.113.143:8005/update_stream/{user_id}/{chat_id}"
            self.update_stream_url(new_stream_url)
            print(new_stream_url)
        except Exception as e:
            print("error printing stream_url", e)

        #print("this is my input user_id, chat_id", user_id, chat_id)

        try:
            # Call the chain to operate with, unpacking kwargs
            chain = self._create_chain(
                self.get_chat_history_from_management(user_id=user_id,
                                                      model_id=self.model_id,
                                                      chat_id=chat_id,
                                                      chat_memory_threshold=self.data_loader.settings.chat_memory_threshold),
                source_field=source_field,
                **kwargs
            )

        except AttributeError:
            # Handle the case where settings or chat_history attributes are missing
            chain = self._create_chain(
                self.get_chat_history_from_management(user_id=user_id,
                                                      model_id=self.model_id,
                                                      chat_id=chat_id,
                                                      chat_memory_threshold=chat_memory_threshold_default),
                source_field=source_field,
                **kwargs
            )  # Fallback behavior

        if chain is None:
            # Handle the case where _create_chain returns None
            return {"error": "Chain creation failed"}

        # Compute the output of the chain
        output = chain.invoke({"question": self.question})

        # Add the response to the chain history
        #self.chat_history.append({self.question: output})
        try:
            self.add_chat_history_to_management(user_id=user_id, model_id=self.model_id, chat_id=chat_id, question_answer_pair=(self.question, output['response']))
        except Exception:
            raise Exception('No user_id found. User must login first')
        return output

    def _concatenate_wrapper_responses(self, question: str, responses: list) -> Dict[str, Any]:
        """Combine all the LLM responses from individual chunks into one answer given the original question"""

        # Prepare the input for the concatenation prompt
        concatenation_input = {
            "question": question,
            "responses": "\n\n".join(responses)
        }

        # Create a chain for the concatenation process
        concatenation_chain = (
            self.concatenation_prompt
            | self.llm_baseline
            | StrOutputParser()
        )

        # Generate the combined response
        combined_response = concatenation_chain.invoke(concatenation_input)

        # Return the combined response as a dictionary
        return {"text model response": combined_response}

    def spreadsheet_ts_wrapper(self, question: str) -> Dict[str, Any]:
        """wrapper function for the spreadsheet text search with concurrent processing"""

        # define chunk iteration ceiling
        chunk_ceiling = 2

        # gather all the responses for further analysis
        responses: list = []

        # iterative solution
        for i in range(1, chunk_ceiling+1):
            print(f'on iteration: {i}')
            response = self.invoke(question, chunks=str(i))['text model response']
            print(response)
            responses.append(response)

        # Use ThreadPoolExecutor for concurrent processing
        """with ThreadPoolExecutor(max_workers=chunk_ceiling) as executor:
            # Submit tasks for each chunk using a lambda function
            future_to_chunk = {
                executor.submit(
                    lambda c: self.invoke(question, chunks=str(c))['text model response'],
                    i
                ): i for i in range(1, 1+chunk_ceiling)
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                #try:
                response = future.result()
                responses.append(response)
                #except Exception as exc:
                #    print(f'Chunk {chunk} generated an exception: {exc}')"""

        # call LLM to combine responses
        print(f'responses: {responses}')
        combined_response = self._concatenate_wrapper_responses(question, responses)

        return combined_response

    def test_query_generation(self):
        # Test function to demonstrate caching behavior
        question1 = "What is the capital of France?"
        question2 = "What is the population of Tokyo?"

        print("First call for question 1:")
        self.generate_queries(question1)

        print("\nSecond call for question 1 (should use cache):")
        self.generate_queries(question1)

        print("\nFirst call for question 2:")
        self.generate_queries(question2)

        print("\nThird call for question 1 (should use cache):")
        self.generate_queries(question1)

        print(f"\nTotal actual query generations: {self.query_generation_count}")


class MultiModalModel(RAGModelUtilities):
    """define the main model class with the chain for the textual side of the model"""

    def __init__(self, stream_url: str = 'http://127.0.0.1:8005/stream'):
        """initialize the model"""
        super().__init__(stream_url=stream_url)
        # define the backend resources regarding the colpali model and searching capabilities
        #self.colpali_resources: ColPaliInference = ColPaliInference()
        self.model_id: int = 1

        # define the output parser
        self.output_parser = JsonOutputParser(pydantic_object=RAGResponse)

        # define the retriever
        self.retriever = self._create_retrievers()

        # define the citation generation class
        self.citation_generator = self.citation_generation_dict[self.model_specs.citation_generation_type]

    def _create_retrievers(self):
        """Create the retrievers for multimodal search"""

        # Initialize the multimodal retriever
        multimodal_retriever = LanceRetrieverMultiModal(
            table=self.data_loader.lancedb_client[4], # 4 is for multimodal table
            k=self.data_loader.settings.k_top_multimodal,
            mode='cosine',
            vector_column_name='vector',
            embedding=self.data_loader.embedding
        )

        return multimodal_retriever

    def _create_chain(self, chat_history: list = None, **kwargs):
        """Create the chain for multimodal search and response generation"""

        # Get the multimodal retriever
        multimodal_retriever = self._create_retrievers()

        # Define the main chain
        final_chain = (
            {
                "question": RunnablePassthrough(),
                "history": lambda x: chat_history if chat_history else []
            }
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "history": x["history"],
                "colpali_results": multimodal_retriever._get_relevant_documents(x["question"])
            })
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "history": x["history"],
                "response": self._call_llm(x["question"], x["colpali_results"]),
                "combined_context": x["colpali_results"]
            })
            | RunnableLambda(lambda x: {
                "response": x["response"]["response"],
                "references": self.citation_generator.generate_citations(x["combined_context"], source_field='source'),
                "combined_context": x["combined_context"]
            })
        )

        return final_chain

    @staticmethod
    def _handle_image_bytes_payload(image_bytes: bytes) -> str:
        """handle the image bytes payload"""

        # Make sure we have a proper data URL format
        if not image_bytes.startswith('data:image/'):
            # Encode to base64 if it's raw bytes
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            image_payload = f"data:image/jpeg;base64,{image_base64}"
        else:
            image_payload = image_bytes

        return image_payload

    def _create_prompt(self, input_dict: dict) -> list:
        """Create the prompt for the model, including text and image content."""

        # Extract question and vision_results from input dictionary and define the seen images set to keep track of duplicates
        question, vision_results, seen_images = input_dict['question'], input_dict['vision_context'], set()
        print(f"Number of total retrieved images: {len(vision_results)}")
        # Create the message content starting with the text part and format the prompt
        message_content = [{"type": "text", "text": self.model_specs.multimodal_rag_template.format(question=question)}]

        # Iterate through each image result
        for result in vision_results:
            # Create a unique identifier for this image using source and page
            image_id = (result['source'], result['page'])

            # Skip if we've seen this image or add it to the seen images
            if image_id in seen_images:
                continue
            else:
                seen_images.add(image_id)

            # Get the image bytes from the result
            image_payload = self._handle_image_bytes_payload(result['page_image_bytes'])

            # Add the image to the message content
            message_content.append({
                "type": "image_url",
                "image_url": {"url": image_payload, "detail":
                            "low" if self.model_specs.low_resolution_img else "high",
                            "resize": self.model_specs.resize_img_dim}
            })
            print(f"Added image from source: {result['source']}, page: {result['page']}")

        # Create the HumanMessage and return it
        return [HumanMessage(content=message_content)]

    def _process_vision_response(self, response: list) -> Any:
        """process the colpali response"""

        # create the containers for images
        container_images = []

        # iterate through the response and append the images to the container
        for doc in response:
            container_images.append(doc['page_image_bytes'])

        return container_images

    def _query_colpali(self, question: str) -> Any:
        """query the colpali model"""

        # clean the user question
        cleaned_question = self._clean_user_question(question)

        # perform colpali inference with the user question
        response = self.colpali_resources._run_inference(cleaned_question)

        # process the response
        processed_response = self._process_colpali_response(response)

        return processed_response

    def _call_llm(self, prompt: Any) -> str:
        """Call the LLM with the prompt and image data. The output is a string."""

        # get the results from the LLM
        results = self.baseline_model.invoke(prompt)

        # get the total input and output tokens from the results
        print(f'Total input tokens: {results.usage_metadata["input_tokens"]}')
        print(f'Total output tokens: {results.usage_metadata["output_tokens"]}')

        return results.content

    def get_query_embedding_voyageai(self, query):
        """Get the embedding vector for a query using the multimodal embedding model.

        Args:
            query: The query text to embed

        Returns:
            numpy.ndarray: The embedding vector as a numpy array with float32 dtype
        """
        # get the embedding with the multimodal embedding model and input must be an embedding list
        qs = self.data_loader.embedding.multimodal_embed([[query]], model='voyage-multimodal-3').embeddings
        return np.array(qs[0], dtype=np.float32)

    def _create_chain(self, chat_history: list = None):
        """create the chain for the model"""

        # define the retrieval chain for the multimodal model to generate the query embeddings and output the images as byte strings in a JSON structure
        retrieval_rag_fusion = (
            RunnableLambda(lambda x: {
                'questions_array': self.generate_queries(x['question'])
            })
            | RunnableLambda(lambda x: {
                'query_embeddings': [*map(self.get_query_embedding_voyageai, x['questions_array'])]
            })
            | RunnableLambda(lambda x: [self.retriever.invoke(q) for q in x['query_embeddings']])
            | RunnableLambda(lambda x: [item for sublist in x for item in sublist])  # Flatten the list
        )

        # Define the chain for the LLM
        llm_call_chain = (
            RunnableLambda(lambda x: self._create_prompt(x))
            | RunnableLambda(lambda x: self._call_llm(x))
            | StrOutputParser()
        )

        # use the chain to create the final chain
        final_chain = (
            {
                "question": RunnablePassthrough(),
                "history": lambda _: chat_history if chat_history else [],
                "vision_context": retrieval_rag_fusion
            }
            | RunnableLambda(lambda x: {
                **x,
                "llm_response": llm_call_chain.invoke(x)
            })
            | RunnableLambda(lambda x: {
                "result": x["llm_response"],
                "references": self.citation_generator.generate_citations(x["vision_context"], source_field='source'),
            })
        )

        return final_chain

    def invoke(self, question: str, chat_memory_threshold_default: int = 2, **kwargs) -> Dict[str, Any]:
        """invoke the model"""

        # Clean the question
        self.question = self.clean_query(question)

        # get the user id, model_id, and chat_id
        user_id, chat_id = self.get_query_auth(**kwargs)

        # get the chat history
        chat_history = self.get_chat_history_from_management(user_id=user_id, model_id=self.model_id, chat_id=chat_id, chat_memory_threshold=chat_memory_threshold_default)

        try:
            # self.stream_url = f"http://5.78.113.143:8005/update_stream/{self.user_id}/{self.chat_id}"
            new_stream_url = f"http://5.78.113.143:8005/update_stream/{user_id}/{chat_id}"
            self.update_stream_url(new_stream_url)
            print(new_stream_url)
        except Exception as e:
            print("error printing stream_url", e)

        # create the chain
        chain = self._create_chain(chat_history=chat_history)

        # invoke the chain
        output = chain.invoke({"question": self.question})
        return output


# TO DO: deploy this on a personal cloud server for privacy reasons
class AudioTranscriptionModel(RAGModelUtilities):
    """Class for the audio transcription model using Whisper and GPT-3.5-turbo."""

    # define which whisper model to use
    whisper_model_name: str = 'tiny' # or use base, small, turbo

    def __init__(self):
        super().__init__()  # Initialize the parent class RAGModelUtilities

        # Define the language model using Langchain's ChatOpenAI
        self.llm_baseline = ChatOpenAI(
            temperature=self.model_specs.temperature,  # Set the temperature for the model's responses
            openai_api_key=api_key,  # Use the OpenAI API key for authentication
            model=self.model_specs.model_id  # Specify the model ID to use
        )

        # define the model name and version
        self.model_name: str = 'nvlabs/parakeet-rnnt-1.1b'
        self.version: str = '73ddbebaef172a47c8dfdd79381f110bfdc7691bcc7a4edde82f0a39e380ce50'

        # load the whisper model
        self.whisper_model = whisper.load_model(self.whisper_model_name)

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper."""

        # make sure the file is a valid .wav or .mp3 file
        if not audio_file_path.endswith(('.wav', '.mp3')):
            raise ValueError("Invalid file format. Please provide a .wav or .mp3 file.")

        # transcribe the audio file
        transcription = self.whisper_model.transcribe(audio_file_path)

        # return the transcription text
        return transcription['text']

    def clean_transcription(self, transcription):
        """Clean and process the transcription using GPT-3.5-turbo via Langchain."""
        prompt = f"Clean and process this question: {transcription}"  # Create a prompt for the language model
        response = self.llm_baseline(prompt)  # Get the response from the language model
        cleaned_text = response['choices'][0]['message']['content']  # Extract the cleaned text from the response
        return cleaned_text  # Return the cleaned text

    def process_audio_file(self, audio_file_path, clean_transcription: bool = True):
        """Process the audio file and return the cleaned question."""
        transcription = self.transcribe_audio(audio_file_path)  # Transcribe the audio file
        if clean_transcription:
            cleaned_question = self.clean_transcription(transcription)  # Clean the transcription
            return cleaned_question  # Return the cleaned question
        else:
            return transcription  # Return the transcription


def run():
  # Instantiate Settings
  model_obj = TextModel()

  # Test the new spreadsheet_ts_wrapper function
  question = "What is the highest value the stock reached during the entire period?"
  response = model_obj.spreadsheet_ts_wrapper(question)

  # Print the combined response
  print("Combined Response:")
  pprint(response)

def test():
  model = TextModel()
  model.test_query_generation()
#run()

username = 'testuser'
email = 'testuser@example.com'
password = 'securepassword'
model_type = 1 # text model
chat_id = 2
user_id = 1

model = TextModel(stream_url='http://5.78.113.143:8005/stream')
print(model.invoke("What is an endogenous variable in econometrics?", disable_tabular=False,
             user_id=user_id, chat_id=chat_id))

#model = MultiModalModel(stream_url='http://5.78.113.143:8005/stream')
#print(model.invoke("Describe this project. How does it work?", disable_tabular=False,
#            user_id=user_id, chat_id=chat_id))
#query = 'What is an endogenous variable in econometrics?'
#query_vector = model.get_query_embedding_voyageai(query)
#print(model.retriever.invoke(query_vector))

#audio_model = AudioTranscriptionModel()
#audio_model.process_audio_file('/content/test_jarvis.wav', False)

