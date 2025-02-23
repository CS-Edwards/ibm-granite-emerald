import requests
import json
from requests.exceptions import HTTPError, RequestException
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
import logging
from typing import Optional
from neo4j import GraphDatabase, basic_auth


# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_instructions(file_path):
    """
    Loads LLM system instructions from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file containing the instructions.

    Returns:
    - dict: The loaded instructions.
    """
    try:
        with open(file_path, 'r') as file:
            instructions = json.load(file)
        return instructions
    except FileNotFoundError:
        raise Exception(f"Instructions file not found: {file_path}")
    except json.JSONDecodeError:
        raise Exception(f"Error decoding JSON from file: {file_path}")


#
# Source: Docling Docs
#

def document_processing(doc_source: str, max_tokens: Optional[int] = 32000) -> str:
    """
    Processes a document by converting it, chunking it into manageable pieces,
    and returning the concatenated text of all chunks.

    Parameters:
    - doc_source (str): The source of the document to be processed, which can be a file path or a URL.
    - max_tokens (int, optional): The maximum token limit for each chunk. Default is 3200.

    Returns:
    - str: The concatenated text of all processed chunks.

    Raises:
    - ValueError: If the document conversion or chunking fails.
    - Exception: For any other unexpected errors during processing.
    """
    try:
        # Convert the document
        logger.info("Starting document conversion...")
        doc = DocumentConverter().convert(source=doc_source).document
        logger.info("Document conversion successful.")

        # Load the tokenizer model
        logger.info("Loading tokenizer model...")
        model_path = "ibm-granite/granite-3.1-8b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer model loaded successfully.")

        # Set up the chunker
        logger.info("Setting up the chunker...")
        chunker = HybridChunker(
            tokenizer=tokenizer, 
            max_tokens=max_tokens,  # Set max tokens per chunk
            merge_peers=True         # Option to merge peers if needed
        )

        # Chunk the document
        logger.info("Chunking the document...")
        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)

        print(f"LENGTH of CHUNKS LIST:{len(chunks)}")

        # Concatenate all chunks
        input_text = ""
        for chunk in chunks:  # Assuming chunks is a list of DocChunk objects
            input_text += chunk.text + "\n"  # Concatenate chunk text into one string.
            logger.info("Document processing completed.")
        return input_text

    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise Exception(f"An error occurred during document processing: {e}")



#
# Extract keywords and themes based on the document chunks
#


def generate_text_granite_instruct(system_instruct, input_text, my_token):
    """
    Generates text using IBM Granite AI language model based on the provided system instructions and input text.

    Parameters:
    - system_instruct(str): The instruction that defines the role or behavior of the AI system.
    - input_text (str): The input text that the AI will process.
    - my_token (str): The authorization token for accessing the IBM API.

    Returns:
    - dict: The JSON response from the IBM Granite API.

    Raises:
    - ValueError: If any required parameter is missing or invalid.
    - HTTPError: If the API request returns a non-200 status code.
    - RequestException: For network-related issues or timeouts.
    """
    
    # Validate input parameters
    if not all([system_instruct, input_text, my_token]):
        raise ValueError("All parameters (system_instruct, input_text, my_token) must be provided.")

    # Define the URL and the body of the request
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": f"""<|start_of_role|>system<|end_of_role|>\"You are Granite, an AI language model developed by IBM in 2024. You are an insightful assistant, carefully analyzing the provided text to identify the core themes, key topics, and important relationships. {system_instruct}\<|end_of_text|> 
        <|start_of_role|>assistant<|end_of_role|> 

        %%start
        {input_text}
        %%end
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        },
        "model_id": "ibm/granite-3-8b-instruct",
        "project_id": "c9a0a6ff-e9af-49ae-8362-d11da4bbf632"
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {my_token}"
    }

    try:
        # Send the POST request to the IBM API
        response = requests.post(url, headers=headers, json=body)

        # Raise an HTTPError if the response status code is not 200
        response.raise_for_status()

        # Return string
        data = response.json()
        return data['results'][0]['generated_text']

    except HTTPError as http_err:
        # Handle HTTP error (non-200 status codes)
        raise HTTPError(f"HTTP error occurred: {http_err}")
    
    except RequestException as req_err:
        # Handle other request errors (network issues, timeouts)
        raise RequestException(f"Request error occurred: {req_err}")
    
    except Exception as err:
        # Catch any other unexpected exceptions
        raise Exception(f"An error occurred: {err}")
    


#
# Create/ fix Cypher query
#



def generate_code_granite_instruct(input_text: str, system_instruct:str, my_token)->str:
    """
    Generates code using the IBM Granite code language model by providing input text and system instructions.
    
    Parameters:
    - input_text (str): The input text that Granite will process.
    - system_instruct (str): The system instruction that defines the context for the AI model.
    - my_token (str): The API token for authorization.
    - url (str): The URL of the IBM Granite API endpoint (default is set to the standard endpoint).

    Returns:
    - dict: The JSON response from the IBM Granite API.

    Raises:
    - ValueError: If any required parameter is missing or invalid.
    - HTTPError: If the API request returns a non-200 status code.
    - RequestException: For network-related issues or timeouts.
    """
    
    # Validate input parameters
    if not all([input_text, system_instruct, my_token]):
        raise ValueError("All parameters (input_text, system_instruct, my_token) must be provided.")

    # Define the URL and the body of the request
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": f"""System:
        You are an intelligent AI programming assistant, utilizing a Granite code language model developed by IBM. Your primary function is to assist users in programming tasks,
        including code generation, code explanation, code fixing, generating unit tests, generating documentation, application modernization, vulnerability detection, function calling,
        code translation, and all sorts of other software engineering tasks.
        {system_instruct}

        %%start
        {input_text}
        %%end
        Answer:
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        },
        "model_id": "ibm/granite-34b-code-instruct",
        "project_id": "c9a0a6ff-e9af-49ae-8362-d11da4bbf632"
    }

    # Define the headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {my_token}"
    }

    try:
        # Send the POST request to the IBM API
        response = requests.post(url, headers=headers, json=body)

        # Raise an HTTPError if the response status code is not 200
        response.raise_for_status()

        # Return string
        data = response.json()
        return data['results'][0]['generated_text']

    except requests.exceptions.HTTPError as http_err:
        # Handle HTTP error (non-200 status codes)
        raise Exception(f"HTTP error occurred: {http_err}")

    except requests.exceptions.RequestException as req_err:
        # Handle other request errors (network issues, timeouts)
        raise Exception(f"Request error occurred: {req_err}")

    except Exception as err:
        # Catch any other unexpected exceptions
        raise Exception(f"An error occurred: {err}")


#
# Clean Cypher Query
#
def clean_cypher_query(cypher_query:str)->str:
    '''
    Workaround/ Mitigation to truncate the draft query and use send data to Neo4j 
    '''
    query_lines = cypher_query.strip().split('\n')
    
    query_lines = query_lines[:-1]
    
    cleaned_query = '\n'.join(query_lines)
    
    return cleaned_query




#
# Connect to Neo4j and Submit Query
#


def query_neo(cypher_query:str):
    driver = GraphDatabase.driver(
    "bolt://44.192.6.23:7687",
    auth=basic_auth("neo4j", "anthem-bowl-oscillators"))

    cypher_query = f'''
                    {cypher_query}
                    '''
    with driver.session(database="neo4j") as session:
        results = session.execute_write(
            lambda tx: tx.run(cypher_query).data())
        for record in results:
            print(record['count'])

        driver.close()
    print("Neo4j Sandbox Updated")


def special_delim_token(input_text: str) -> str:
    """
    Adds a special delimiter token to each node/edge/node entry in the input text to facilitate 
    code generation accuracy in code-LLMs.

    Args:
        input_text (str): The input text containing node/edge/node entries, separated by newlines.

    Returns:
        str: The processed text with a special delimiter token added after each line.

    Raises:
        ValueError: If input_text is not a string or is empty.
    """
    if not isinstance(input_text, str):
        raise ValueError("Input must be a string.")
    
    if not input_text.strip():  # Check if the input text is empty or just whitespace
        return ""

    # Split the input text into lines, process each line, and add the delimiter
    lines = input_text.strip().split('\n')
    lines_with_delimiter = [line + " |<special-end-tok>|" for line in lines]
    
    # Join the processed lines back into a single string
    special_token_output_str = '\n'.join(lines_with_delimiter)

    return special_token_output_str