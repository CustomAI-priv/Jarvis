try:    
    from setup import DataLoader # import data loader
    from text_model import TextModel, LanceRetriever # import text model
except Exception as e:
    print(f"Error importing modules: {e}")

from pydantic import BaseModel
from typing import Any
import requests 
import json
from lancedb.rerankers import LinearCombinationReranker
import os
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import schedule
import time

# email imports
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# pdf and markdown handling tools
from langchain_text_splitters import MarkdownHeaderTextSplitter
import pymupdf4llm
from pprint import pprint 
import fitz  # PyMuPDF

# get tools for github search
from crewai_tools import GithubSearchTool, tool 


# define the general settings for the developer education model in pydantic
class DeveloperEduConfigs(BaseModel):
    """Developer Edu Configs""" 
    
    # file path for research interests 
    research_interests_file_path: str = "data/research_interests.json"

    # number of articles to return from the abstract retriever
    num_articles: int = 20

    # decide if we want to use perplexity or not
    use_perplexity: bool = True
    use_genai_keywords: bool = True

    # email configs
    email_sender: str = "bigbridgeai@gmail.com"
    app_password: str = "cosl seed mcml rxqx"  # os.getenv("APP_PASSWORD")

    # github tool configs
    github_content_types: list = ['code', 'repo']


# define the prompts for the developer education model in pydantic
class DeveloperEduPrompts(BaseModel): 
    """define the prompts to be used by the developer education model"""

    # define the summary model that reads over each section of the research paper 
    summary_prompt: str = """You are summarizing a technical research paper section by section. Your goal is to educate the reader about the science behind the paper while providing concise, informative summaries. First, analyze the section by generating and answering 5 essential questions that capture the main points and core meaning:

    When formulating your questions:
    a. Address the central theme or argument
    b. Identify key supporting ideas
    c. Highlight important facts or evidence
    d. Reveal the author's purpose or perspective
    e. Explore any significant implications or conclusions

    Then, tailor your summary based on the section type:

    For all sections:
    - Provide a concise summary (2-3 sentences) that captures the essential content.
    - Use clear, engaging language accessible to software developers and technical professionals.
    - Explain complex concepts succinctly, using technical jargon where appropriate.
    - Highlight innovative aspects or breakthrough findings.

    Section-specific guidelines:
    Introduction:
    - Clearly state the problem and current situation the paper addresses.
    - Highlight the paper's main objectives and potential impact on the field.

    Methodology:
    - Describe the research approach using precise technical terms.
    - Outline key steps, tools, or frameworks employed, focusing on their scientific relevance.

    Results:
    - Quote 1-2 significant results, using exact figures or data points when available.
    - Explain the meaning and implications of these results in the context of the research.

    Discussion:
    - Articulate how the findings can be applied in practical, real-world scenarios.
    - Identify potential applications or impacts on software development or related technical fields.

    Ensure your summary effectively teaches the reader about the paper's content, methodology, and significance in the field. Maintain a cohesive narrative that can be easily understood when all sections are combined.
    Here is the section of the paper you are summarizing: {paper_section}
    """

    # define the prompt to select interesting papers
    select_interesting_paper_prompt: str = """You are a technical research curator helping software developers find practical, implementable research papers. Your task is to analyze numbered research paper abstracts and select the 4 most relevant and practical papers for developers interested in {topic}.

    When selecting papers, prioritize:
    1. Direct practical applications in software development
    2. Clear implementation possibilities
    3. Papers that describe tools, frameworks, or algorithms developers can use
    4. Research that solves real-world engineering problems

    Avoid papers that:
    - Are purely theoretical
    - Lack concrete implementation details
    - Focus on mathematical proofs without practical applications
    - Are too domain-specific or niche

    Given these numbered abstracts:
    {abstracts}

    Return only a Python list containing exactly 4 numbers corresponding to the most relevant paper indices. 
    Example format: [1, 4, 7, 12]

    These numbers should represent the papers that would be most useful for a software developer working with {topic}."""

    # define the markdown template for the summaries with HTML formatting
    markdown_template: str = """
    <h1>{title}</h1>
    <h2>{authors}</h2>
    <h3>{updated}</h3>
    <h4>{published}</h4>
    <p>{summary}</p>
    """


# define the utilities class for general functionality and definitions 
class DeveloperEduUtilities: 
    """Class to handle utilities for the developer education"""

    def __init__(self):

        # define the configs 
        self.configs = DeveloperEduConfigs()
        self.model_prompts = DeveloperEduPrompts()

        # connect to gpt-4o with langchain 
        self.github_keyword_llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=100)
        
        # define the summary language model 
        self.summary_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        # define the perplexity api key
        self.perplexity_api_key = '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'

    def delete_directory_files(self, directory: str = "~/pdf") -> None:
        """
        Delete all files in the specified directory.
        
        Args:
            directory (str): Path to directory to clean. Defaults to "~/pdf"
        """
        # Expand the ~ to full home directory path
        directory = os.path.expanduser(directory)
        
        try:
            # Check if directory exists
            if not os.path.exists(directory):
                print(f"Directory {directory} does not exist")
                return
                
            # Iterate through files and remove them
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                    
            print(f"Successfully cleaned directory: {directory}")
            
        except Exception as e:
            print(f"Error cleaning directory {directory}: {str(e)}")


# define the markdown parser class 
class MarkdownParser:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]

        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def parse_pdf_to_markdown(self, pdf_path: str) -> str:
        return pymupdf4llm.to_markdown(pdf_path)

    def get_page_numbers(self, pdf_path: str, text_to_find: str) -> int:
        """Find page number for given text in PDF"""
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            if text_to_find.lower() in page.get_text().lower():
                return page_num + 1  # Adding 1 because page numbers typically start at 1
        return None

    @staticmethod 
    def clean_json_content(content: str) -> str:
      # Remove extra backticks and line numbers
      lines = content.split('\n')
      cleaned_lines = []
      for line in lines:
          # Remove backticks
          line = line.replace('`', '')
          # Remove line numbers at the start
          if line.strip() and line.strip()[0].isdigit():
              line = ' '.join(line.split()[1:])
          # Remove extra spaces around colons
          line = line.replace(' : ', ': ')
          # Clean up URLs
          line = line.replace(' :', ':')
          line = line.replace(' /', '/')
          # Join broken URLs
          line = line.replace('https: //', 'https://')
          # Remove extra spaces
          line = ' '.join(line.split())
          if line.strip():
              cleaned_lines.append(line)

      # Join lines and remove code block markers
      cleaned_content = '\n'.join(cleaned_lines)
      cleaned_content = cleaned_content.replace('```\n', '').replace('\n```', '')

      return cleaned_content

    def print_major_sections(self, markdown_text: str, pdf_path: str) -> list:
        """
        Returns a list of [header, content] pairs from markdown text with cleaned JSON formatting.
        If a numbered section has less than 100 words, combines it with all non-numbered sections below it
        until the next numbered section is encountered.
        Excludes References section and everything after it.
        """
        splits = self.splitter.split_text(markdown_text)
        sections = []
        buffer_content = ""
        current_numbered_section = None
        MIN_CONTENT_LENGTH = 100

        def is_numbered_header(header):
            return any(char.isdigit() for char in header.split()[0])

        def is_references_section(header):
            return header.lower().strip() in ['references', 'reference', 'bibliography']

        for split in splits:
            # Get the deepest header level present
            header = None
            for level in ["header4", "header3", "header2", "header1"]:
                if level in split.metadata:
                    header = split.metadata[level]
                    break

            if not header:
                continue

            # Stop processing if we hit the references section
            if is_references_section(header):
                break

            content = self.clean_json_content(split.page_content)
            
            if is_numbered_header(header):
                # If we have a previous numbered section being buffered, add it to sections
                if current_numbered_section:
                    sections.append([current_numbered_section[0], 
                                current_numbered_section[1] + "\n\n" + buffer_content])
                    buffer_content = ""
                
                # Start a new numbered section
                current_numbered_section = [header, content]
                
                # If this numbered section is long enough, add it immediately
                if len(content.split()) >= MIN_CONTENT_LENGTH:
                    sections.append(current_numbered_section)
                    current_numbered_section = None
                    
            else:  # Non-numbered header
                if current_numbered_section:
                    # If we have a numbered section being buffered, add this content to buffer
                    buffer_content += f"\n\n{header}\n{content}"
                else:
                    # If no numbered section is being buffered, add this as its own section
                    sections.append([header, content])

        # Handle the last section if it exists
        if current_numbered_section:
            sections.append([current_numbered_section[0], 
                          current_numbered_section[1] + "\n\n" + buffer_content])

        return sections
    
    def generate_research_paper_sections(self, pdf_path: str): 
        """process the pdf using the path of the file"""
        
        # generate the markdown text 
        markdown_text = self.parse_pdf_to_markdown(pdf_path)

        # get the major sections 
        sections = self.print_major_sections(markdown_text, pdf_path)
        return sections


# make a class to handle research paper downloads 
class ResearchPaperDownloader(DeveloperEduUtilities):
    
    def __init__(self):
        super().__init__()
        # define the base url for the arxiv API and perplexity model
        self.arxiv_base_url = "http://export.arxiv.org/api/query?"
        self.perplexity_base_url = "https://api.perplexity.ai/chat/completions"

        # query strategy
        self.query_strategy = "all"

        # define the sort by strategy 
        self.sort_by = "relevance"

        # call the data loader 
        self.data_loader = DataLoader()

        # configure the vector databases in the settings model
        self.data_loader.settings.vectordb_idx = {1: 'db_research_papers', 2: 'db_summary_updates'}
        self.data_loader.settings.text_databases = [1, 2]
        self.data_loader.settings.documents_schema_index = 1
        self.data_loader.settings.indexable_field = 'summary'

        # create the vector databases
        self.data_loader.lancedb_client = self.data_loader.fetch_lancedb_client()

        # define the genai keywords method 
        self.genai_keywords_method = self._get_specific_topics_gpt4

    def process_results(self, results: Any) -> list:
        """
        Process the results from the arxiv API and return a list of paper details.
        List fields are converted to strings with "-" delimiter for LanceDB compatibility.
        
        Args:
            results: XML response from arXiv API
            
        Returns:
            list: List of dictionaries containing paper information with keys:
                - title: str
                - authors: str (delimited)
                - updated: str
                - published: str
                - pdf_link: str
                - summary: str
                - affiliations: str (delimited)
        """
        try:
            # Import xml parser
            import xml.etree.ElementTree as ET
            
            # Parse XML string
            root = ET.fromstring(results)
            
            # Initialize list to store papers
            papers = []
            
            # Define namespace mapping
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Process each entry
            for entry in root.findall('atom:entry', ns):
                paper = {}
                
                # Extract basic information
                paper['title'] = entry.find('atom:title', ns).text.strip()
                paper['updated'] = entry.find('atom:updated', ns).text
                paper['published'] = entry.find('atom:published', ns).text
                paper['summary'] = entry.find('atom:summary', ns).text.strip()
                
                # Extract authors and affiliations
                authors = []
                affiliations = []
                for author in entry.findall('atom:author', ns):
                    authors.append(author.find('atom:name', ns).text)
                    affiliation = author.find('arxiv:affiliation', ns)
                    if affiliation is not None:
                        affiliations.append(affiliation.text)
                
                # Convert lists to delimited strings
                paper['authors'] = "-".join(authors) if authors else ""
                paper['affiliations'] = "-".join(affiliations) if affiliations else ""
                
                # Extract PDF link
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        paper['pdf_link'] = link.get('href')
                        break
                else:
                    paper['pdf_link'] = None
                
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Error processing arXiv results: {str(e)}")
            return []

    def _get_specific_topics_perplexity(self, topic: list[str]) -> list:
        """
        Use the perplexity model to get specific topics from the search query.
        
        Args:
            interest_topics (list[str]): List of general research topics
            
        Returns:
            list: List containing original topics plus their related specific topics
        """
        total_tokens = 0
        cost_per_1k_tokens = 0.0002  # Current Perplexity API cost per 1k tokens
        
        try:
            # Construct the prompt for the specific topic
            prompt = f"""
            Given the research topic '{topic}', provide exactly 2 closely related 
            but more specific research topics. Format your response as a Python list with 
            exactly 3 items, where the first item is the original topic. Example format:
            ["Original Topic", "Specific Topic 1", "Specific Topic 2"]
            """
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research topic expert. Provide only the requested list format with no additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "max_tokens": 100
            }
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.perplexity_base_url,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                # Extract tokens used from response
                tokens_used = response.json().get('usage', {}).get('total_tokens', 0)
                total_tokens += tokens_used
                
                # Extract the topics list
                topics_list = eval(response.json()['choices'][0]['message']['content'])
                if isinstance(topics_list, list) and len(topics_list) == 3:
                    result = topics_list
                else:
                    # Fallback for this topic
                    result = [
                        topic,
                        f"{topic} Applications",
                        f"{topic} Methods"
                    ]
            else:
                # Fallback for failed request
                result = [
                    topic,
                    f"{topic} Applications",
                    f"{topic} Methods"
                ]
            
            # Calculate and print total cost
            total_cost = (total_tokens / 1000) * cost_per_1k_tokens
            print(f"API Usage Stats:")
            print(f"Total tokens used: {total_tokens}")
            print(f"Estimated cost: ${total_cost:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Error getting specific topics: {str(e)}")
            # Fallback for all topics if there's an error
            result = [
                topic,
                f"{topic} Applications",
                f"{topic} Methods"
            ]
            return result

    def _get_specific_topics_gpt4(self, topic: list[str]) -> list:
        """
        Use GPT-4 model to get specific topics from the search query.
        
        Args:
            interest_topics (list[str]): List of general research topics
            
        Returns:
            list: List containing original topics plus their related specific topics
        """
        try:  
            # Create messages list
            messages = [
                (
                    "system",
                    "You are a research topic expert. Provide only the requested list format with no additional text."
                ),
                (
                    "human",
                    """Given the research topic {topic}, provide exactly 2 closely related 
                    but more specific research topics. Format your response as a Python list with 
                    exactly 3 items, where the first item is the original topic. Example format:
                    ["Original Topic", "Specific Topic 1", "Specific Topic 2"]""".format(topic=topic)
                ),
            ]
            
            # Invoke the LLM directly
            response = self.github_keyword_llm.invoke(messages)
            
            # Extract and validate the topics list
            topics_list = eval(response.content)
            if isinstance(topics_list, list) and len(topics_list) == 3:
                result = topics_list
            else:
                # Fallback for invalid format
                result = [
                    topic,
                    f"{topic} Applications",
                    f"{topic} Methods"
                ]
                
            return result
            
        except Exception as e:
            print(f"Error getting specific topics with GPT-4: {str(e)}")
            # Fallback for all topics if there's an error
            result = [
                topic,
                f"{topic} Applications",
                f"{topic} Methods"
            ]
            return result

    def query_research_papers(self, search_query: str) -> dict:
        """
        Query research papers from the arxiv API with date filtering for the last month.

        Args:
            search_query (str): The search query to use for finding relevant papers.
                          Example: "multi agent systems" or "reinforcement learning"

        Returns:
            dict: JSON response from the arxiv API containing paper information.
                 Includes titles, abstracts, authors, and other metadata.

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
        
        # Get yesterday's date
        yesterday = datetime.now() - timedelta(days=1)
        # Get date one month before yesterday
        month_ago = yesterday - timedelta(days=30)
        
        # Format dates in arXiv format (YYYYMMDDHHSS)
        yesterday_formatted = yesterday.strftime("%Y%m%d2000")
        month_ago_formatted = month_ago.strftime("%Y%m%d2000")
        
        # Construct date filter
        date_filter = f"submittedDate:[{month_ago_formatted}+TO+{yesterday_formatted}]"
        
        # add the search query to the base url with date filter
        url = (f"{self.arxiv_base_url}search_query={self.query_strategy}:{search_query}"
               f"+AND+{date_filter}")
        
        # add the max results and start
        url += "&max_results=100&start=0"

        # also sort by relevance in an ascending order
        url += f"&sortBy={self.sort_by}&sortOrder=ascending"

        # send a request to the arxiv API
        response = requests.get(url)
        return self.process_results(response.content)

    def download_facade(self, topics: list[str]): 
        """
        Facade method to get research papers for a list of research interests and call the perplexity model to get specific topics. 
        Afterwards we query the arxiv API for each of the specific topics.
        We then insert the results into the backend vector database.
        """

        # get the research interests
        research_interests = topics #self.get_research_interest()

        # query the arxiv API for each of the specific topics
        for original_topic in research_interests:
            topics = self.genai_keywords_method(original_topic) if self.configs.use_genai_keywords else [original_topic]

            # query the arxiv API for each of the specific topics
            print(topics)
            for topic in topics:
                research_papers = self.query_research_papers(topic)

                # insert the results into the vector database
                self.data_loader.load_documents(research_papers, 1, batch_size=20)

        # create the index for the research papers
        self.data_loader.postprocess_vectordb()


class SummarizeResearchPapers(ResearchPaperDownloader):
    """Class to summarize research papers"""

    def __init__(self):
        super().__init__()

        # create the retrievers for the LanceDB database
        self.abstract_retriever, self.summary_retriever = self._create_lancedb_retrievers()

        # initiate the markdown and sections generation tool 
        self.markdown_tool = MarkdownParser()

    def _create_lancedb_retrievers(self):
        """
        Create the retrievers for the LanceDB database.
        
        Note:
            The LanceDB table indices are defined in the parent class ResearchPaperDownloader:
            - Index 1: Research paper abstracts
            - Index 2: Research paper summaries
            
        Creates two retrievers:
            - abstract_retriever: For searching through paper abstracts
            - summary_retriever: For searching through paper summaries
        """

        # define the reranker used in all tables
        reranker = reranker = LinearCombinationReranker(
            weight=self.data_loader.settings.hybrid_search_ratio,
        )

        # Initialize the custom retriever object for the TEXT retriever
        abstract_retriever = LanceRetriever(
            table=self.data_loader.lancedb_client[1],
            reranker=reranker,
            k=self.configs.num_articles,  # Ensure k is set
            mode='hybrid'  # Ensure mode is set
        )

        # create a retriever for the summary updates to get chunks from the written content 
        summary_retriever = LanceRetriever(
            table=self.data_loader.lancedb_client[2],
            reranker=reranker,
            k=self.configs.num_articles,
            mode='hybrid'
        )

        return abstract_retriever, summary_retriever
    
    def download_pdf_to_local(self, pdf_url: str) -> str:
        """
        Download a PDF from a URL to the PDF directory.
        
        Args:
            pdf_url (str): URL of the PDF to download
            
        Returns:
            str: Path to the downloaded file
            
        Raises:
            Exception: If download fails or PDF cannot be saved
        """
        try:
            # Send GET request to download PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paper_{timestamp}.pdf"
            pdf_path = os.path.join(os.path.expanduser("~/pdf"), filename)
            
            # Create pdf directory if it doesn't exist
            os.makedirs(os.path.expanduser("~/pdf"), exist_ok=True)
            
            # Write PDF content to file
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            print(f"PDF downloaded successfully to: {pdf_path}")
            return pdf_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {str(e)}")
            raise
        except Exception as e:
            print(f"Error saving PDF: {str(e)}")
            # Clean up file if it was created
            if 'pdf_path' in locals() and os.path.exists(pdf_path):
                os.unlink(pdf_path)
            raise

    def github_search_generate_keywords(self, paper_abstract: str) -> list[str]:
        """Generate keywords from the paper abstract for GitHub search
        
        Args:
            paper_abstract (str): The abstract text from a research paper
            
        Returns:
            list[str]: 1-2 technical keywords suitable for GitHub search
        """
        prompt = f"""Given this research paper abstract, extract 1-2 technical highly-specific keywords that would be most useful for finding specific related GitHub repositories. Focus on implementation-specific terms, frameworks, tools, or algorithms. Things that would be useful for a developer to integrate into their own coding.

            Abstract: {paper_abstract}

            Return only a Python list of keywords. Example format: ["transformer", "BERT"]"""

        response = self.github_keyword_llm.invoke(prompt)
        try:
            # Extract keywords from the response and ensure it's a list
            keywords = eval(response.content)
            if isinstance(keywords, list) and len(keywords) <= 2:
                return keywords
            else:
                return keywords[:2]  # Limit to first 2 if more were generated
        except:
            # Fallback if response parsing fails
            return ["algorithm"]

    def get_github_pages_perplexity(self, vectordb_paper_objs: list[Any], keywords: list[list[str]]) -> str:
        """Generate GitHub search queries for multiple papers in a single API call
        
        Args:
            vectordb_paper_objs (list[Any]): List of paper objects from vector database
            keywords (list[list[str]]): List of keyword lists corresponding to each paper
            
        Returns:
            str: String containing GitHub repository information in easily parseable format
        """
        try:
            # Build paper information list
            papers_info = []
            for paper, kw in zip(vectordb_paper_objs, keywords):
                title = paper.metadata.get('title', '')
                authors = ', '.join(paper.metadata.get('authors', '').split(',')[:3])  # First 3 authors
                papers_info.append(f"PAPER:\nTitle: {title}\nAuthors: {authors}\nKeywords: {', '.join(kw)}\n")
            
            papers_info_concatenated = '\n'.join(papers_info)

            prompt = f"""Find relevant GitHub repositories for these research papers. For each paper, list 2-3 most relevant repositories.

            Papers to analyze:
            {papers_info_concatenated}

            Format each repository on a new line as:
            REPO|||repository_name|||github_url|||brief_description

            Example format:
            REPO|||transformers|||https://github.com/huggingface/transformers|||Implementation of state-of-the-art transformers
            
            Return ONLY the repository listings, no other text."""

            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a technical expert who finds relevant GitHub repositories."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": 1000  # Increased for multiple papers
            }
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.perplexity_base_url,
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"Perplexity API error: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error in get_github_pages_perplexity: {str(e)}")
            return ""

    def parse_github_pages_perplexity(self, perplexity_response: str) -> list[dict]:
        """Parse the perplexity API response using string splitting to extract repository information
        
        Args:
            perplexity_response (str): Response from perplexity API containing REPO||| formatted lines
            
        Returns:
            list[dict]: List of dictionaries containing repository information
        """
        try:
            repos = []
            # Split response into lines and process each REPO line
            for line in perplexity_response.split('\n'):
                if line.startswith('REPO|||'):
                    # Split by delimiter and unpack values
                    _, name, url, description = line.split('|||')
                    
                    # Only include valid GitHub URLs
                    if url.startswith('https://github.com/'):
                        repos.append({
                            'name': name.strip(),
                            'url': url.strip().rstrip('/'),  # Clean URL
                            'description': description.strip()
                        })
            
            return repos
            
        except Exception as e:
            print(f"Error parsing repository information: {str(e)}")
            return []

    def retrieve_most_interesting_paper(self, topic: str) -> list:
        """Get the most interesting paper from the vector database
        
        Args:
            topic (str): Research topic to search for
            
        Returns:
            list[tuple]: List of (index, abstract) tuples
        """
        # Get abstracts using the retrievers
        abstract_results = self.abstract_retriever.get_relevant_documents(topic)
        
        # Combine and format results as numbered tuples
        abstracts = []
        for idx, doc in enumerate(abstract_results, start=1):
            abstracts.append((idx, doc.page_content))
        
        # get the most interesting papers 
        response = self.summary_llm.invoke(
            messages=[
                {"role": "system", "content": self.model_prompts.select_interesting_paper_prompt.format(abstracts=abstracts, topic=topic)}
            ]
        )

        # get the indices of the most interesting papers
        interesting_paper_indices = eval(response.content)

        # get the most interesting papers
        most_interesting_papers = [abstract_results[i-1] for i in interesting_paper_indices]

        return most_interesting_papers
    
    def summarize_research_paper(self, research_paper_path: str) -> list[dict]:
        """Summarize the research papers"""
        
        # get the section of the research paper 
        sections = self.markdown_tool.generate_research_paper_sections(research_paper_path)

        # Process each section with the summary LLM
        summaries = []
        for section_full_context in sections:
            response = self.summary_llm.invoke(
                messages=[
                    {"role": "system", "content": self.model_prompts.summary_prompt.format(paper_section=section_full_context)},
                ]
            )
            summaries.append(response.content)

        return summaries, '\n'.join(summaries)

    def updated_summarize_research_paper(self, research_paper_path: str) -> list[dict]:
        """Define the markdown template and use structured output..."""
        pass

    def summarize_research_paper_notebooklm(self, research_paper_path: str) -> list[dict]:
        """Summarize the research papers using the NotebookLM model"""
        pass

    def save_and_index_sumary(self, dates_updated: list[str], github_links: list[str], pdf_links: list[str], topic: str, summaries: list[str]) -> str:
        """Save the summary to a file and index it in the vector database"""
        
        # format the summaries according to the document schema and override the title with the topic
        summaries = [
            {
                "summary": summary,
                "title": topic,  # repurpose the topic as the title
                "authors": "",  # You may need to provide this from elsewhere
                "updated": json.dumps(dates_updated) if isinstance(dates_updated, list) else json.dumps([]),  # Convert list to JSON string
                "published": "",  # You may need to provide this from elsewhere
                "pdf_link": json.dumps(pdf_links) if isinstance(pdf_links, list) else json.dumps([]),  # Convert list to JSON string
                "affiliations": json.dumps(github_links) if isinstance(github_links, list) else json.dumps([])  # Convert list to JSON string
            }
            for summary in summaries
        ]

        # insert the summaries into the vector database 
        self.data_loader.load_documents(summaries, 1, batch_size=100)

        # create the index for the summaries
        self.data_loader.postprocess_vectordb()

    def summarize_facade(self, topics: list[str]):
        """Facade method to download the research papers, get the github pages, and summarize the research papers"""
        
        # download the research papers and index them 
        self.download_facade(topics)

        # define container for the github keywords
        github_keywords_container: list = []
        github_pages_container: list = []

        # iterate through each of the topics
        for topic in topics: 
            # get the most interesting research papers - these are LangChain Document(page_content, metadata) type
            top_papers: list = self.retrieve_most_interesting_paper(topic)

            # download each of the papers 
            for research_paper in top_papers: 
                research_paper_file_path: str = self.download_pdf_to_local(research_paper.metadata['pdf_link'])

                # generate the summaries of the research papers
                summaries_seperated, full_summary_concatenated = self.summarize_research_paper(research_paper_file_path)

                # save the summaries to the vector database
                #self.save_and_index_sumary(topic, full_summary_concatenated)

                # generate github keywords from the abstract 
                github_keywords: list = self.github_search_generate_keywords(research_paper.page_content)

                # add the keywords to the container
                github_keywords_container.append(github_keywords)
        
                # get the github pages for each of the keywords
                for keywords in github_keywords_container:
                    github_pages_container.append(self.get_github_pages_perplexity(top_papers, keywords))


class Orchestrator(SummarizeResearchPapers):
    """Class to orchestrate the research paper download, summarization, and email sending"""

    def __init__(self):
        super().__init__()

        # Track last processed topic index in a file
        self.TRACKER_FILE = "last_processed_topic.json"

        # record the latest summary
        self.latest_summary: str = ""
    
    def send_email(self, receiver_email: str, subject_line: str, body: str) -> bool:
        """Send an email to the user"""
        
        # define the message object
        message = MIMEMultipart("alternative")
        message["Subject"] = subject_line
        message["From"] = self.configs.email_sender
        message["To"] = receiver_email

        # define the email part object and attach it 
        part = MIMEText(body, "html")
        message.attach(part)

        # Create secure connection with server and send email
        try: 
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(self.configs.email_sender, self.configs.app_password)
                server.sendmail(
                    self.configs.email_sender, receiver_email, message.as_string()
                )
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
        return True

    def generate_email_subject_line(self, summary: str) -> str:
        """Generate an email subject line from the research summary"""
        prompt = f"""Create a professional email subject line (max 60 chars) that highlights the key technical finding or method from this summary. Be engaging but not clickbait.

        Summary: {summary}

        Return only the subject line."""

        response = self.github_keyword_llm.invoke(prompt)
        return response.content.strip()

    def create_email_body(self, summary: str) -> str:
        """Create an email HTML body from the research summary"""
        pass

    def email_pipeline(self, summary: str) -> None:
        """Create an email template"""
        
        # generate the subject line
        subject_line: str = self.generate_email_subject_line(summary)

        # create the email body
        email_body: str = self.create_email_body(summary)

        # send the email
        self.send_email(self.configs.email_receiver, subject_line, email_body)

    def get_research_interest(self) -> list:
        """
        Read and return research topics from the research_interests.json file.
        
        Returns:
            list: List of research topics
        """
        try:
            with open(self.configs.research_interests_file_path, 'r') as file:
                data = json.load(file)
                # Return the list of research interests
                return data.get('research_interests', [])
        except FileNotFoundError:
            print(f"Error: Research interests file not found at {self.configs.research_interests_file_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {self.configs.research_interests_file_path}")
            return []
        except Exception as e:
            print(f"Error reading research interests file: {str(e)}")
            return []

    def load_last_processed(self, TRACKER_FILE: str) -> dict:
        """Load the last processed topic index and timestamp from tracking file
        
        Args:
            TRACKER_FILE (str): Path to JSON file tracking processing state
            
        Returns:
            dict: Contains:
                - last_index (int): Index of last processed topic (-1 if none)
                - last_run (str): ISO timestamp of last run (None if never run)
                
        Note:
            Creates fresh tracking state if file doesn't exist
        """
        try:
            with open(TRACKER_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"last_index": -1, "last_run": None}
                
    def save_last_processed(self, index: int, TRACKER_FILE: str):
        """Save the current processing state to tracking file
        
        Args:
            index (int): Index of topic that was just processed
            TRACKER_FILE (str): Path to JSON file tracking processing state
            
        Note:
            - Updates both the index and timestamp
            - Creates file if it doesn't exist
            - Overwrites previous state
        """
        with open(TRACKER_FILE, 'w') as f:
            json.dump({
                "last_index": index,
                "last_run": datetime.now().isoformat()
            }, f)
            
    def daily_task(self, TRACKER_FILE: str):
        # Get current research interests
        topics = self.get_research_interest()
        if not topics:
            print("No topics found")
            return
            
        # Load last processed index
        tracker = self.load_last_processed(TRACKER_FILE)
        start_index = (tracker["last_index"] + 1) % len(topics)
        
        # Process next topic
        current_topic = topics[start_index]
        print(f"Processing topic: {current_topic}")
        self.latest_summary = self.summarize_facade([current_topic])
        
        # Save progress
        self.save_last_processed(start_index, TRACKER_FILE)     

    def orchestrator_facade(self):
        """Facade method to orchestrate daily research paper processing"""
        
        # Schedule paper processing at midnight
        schedule.every().day.at("00:00").do(lambda: self.daily_task(self.TRACKER_FILE))
        
        # Schedule email sending at 7 AM every morning
        schedule.every().day.at("07:00").do(lambda: self.email_pipeline(self.latest_summary))
        
        # Start the scheduling loop
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour


class LiveInferenceAgent(SummarizeResearchPapers):
    """Class to perform live inference on the research papers"""

    def __init__(self):
        super().__init__()

        # define the text model 
        self.text_model = TextModel()

        # change the settings k top parameter dynamically to return only the most relevant summary for the topic
        self.text_model.settings.k_top = 1
    
    @tool
    def search_github(self, query: str = 'github link?') -> str:
        """Search GitHub for a given query"""
        return self.github_search_tool.run(query)

    def create_github_search_tool(self, github_repo: str = None) -> GithubSearchTool:
        """Create the github search tool
        
        Args:
            github_repo (str, optional): Specific GitHub repository URL to search.
                                       If None, will search across all repositories.
        
        Returns:
            GithubSearchTool: Tool for semantic GitHub searches
        """
        if github_repo:
            # Initialize tool for specific repository
            return GithubSearchTool(
                github_repo=github_repo,
                content_types=self.configs.github_content_types
            )
        else:
            # Initialize tool for general repository search
            return GithubSearchTool(
                content_types=['code', 'repo', 'issue', 'pr']
            )

    def decide_if_github_search_needed(self, question: str) -> bool:
        """Decide if the question wants to search github for additional code context"""
        return 'github' or 'git' or 'source code' in question.lower()

    def _format_response(self, response: dict, github_search_results: Any) -> str:
        """Format the response from the text model into a readable chat message
        
        Args:
            response (dict): Raw response from text model
            
        Returns:
            str: Formatted string with answer, references, and GitHub results
        """
        try:
            # Load JSON strings
            pdf_links = json.loads(response['combined_context']['pdf_links'])
            dates = json.loads(response['combined_context']['date_updated'])
            
            # Start with the main answer
            output_parts = [
                response.get('answer', ''),
                "\n\n### References:",
            ]
            
            # Add formatted references
            for link, date in zip(pdf_links, dates):
                formatted_date = datetime.fromisoformat(date).strftime('%Y-%m-%d')
                output_parts.append(f"- arXiv API. ({formatted_date}). Retrieved from {link}")
            
            # Add GitHub results if they exist
            if github_search_results:
                output_parts.extend([
                    "\n\n### Related GitHub Repositories:",
                    *[f"- [{repo['name']}]({repo['url']}): {repo['description']}"
                      for repo in self.github_search_results]
                ])
            
            return "\n".join(output_parts)
            
        except Exception as e:
            print(f"Error formatting response: {str(e)}")
            return f"{response.get('answer', '')}\n\nError loading references and GitHub results."

    def invoke(self, question: str) -> str: 
        """Invoke the agent to answer a question"""
        
        # invoke the text model
        response = self.text_model.invoke(question, source_field='pdf_link')

        # decide if the question wants to search github for additional code context 
        github_search_results = None
        if self.decide_if_github_search_needed(question):
            # create the github search tool
            github_links = json.loads(response['combined_context']['affiliations'])
            # take the first github link
            github_search_tool = self.create_github_search_tool(github_links[0])

            # search github
            github_search_results = github_search_tool.run(question)
        
        return self._format_response(response, github_search_results)


# test the downloads
paper_downloads = ResearchPaperDownloader()
paper_downloads.query_research_papers('deep learning')






