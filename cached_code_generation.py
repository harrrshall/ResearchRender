import os
import time
import logging
import hashlib
import json
from typing import Dict, Tuple, Optional
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions
import traceback
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.critical("MONGO_URI environment variable is not set")
    raise ValueError("MONGO_URI environment variable is not set")

db_name = os.getenv("MONGO_DB_NAME") or "researchrender_db"

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[db_name]
    cache_collection = db.cache
    logger.info("MongoDB initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize MongoDB: {str(e)}")
    raise

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY environment variable is not set")
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY environment variable is not set")
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize API clients and MongoDB
try:
    genai.configure(api_key=GEMINI_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("API clients initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize API clients: {str(e)}")
    raise

# Rate limiting configuration
GEMINI_RATE_LIMIT = 15  # requests per minute
GROQ_RATE_LIMIT = 30  # requests per minute
last_gemini_request = 0
last_groq_request = 0


def rate_limit(api_type: str) -> None:
    """Apply rate limiting for API requests."""
    global last_gemini_request, last_groq_request
    current_time = time.time()

    if api_type == "gemini":
        if current_time - last_gemini_request < 60 / GEMINI_RATE_LIMIT:
            sleep_time = 60 / GEMINI_RATE_LIMIT - \
                (current_time - last_gemini_request)
            logger.debug(
                f"Rate limiting Gemini API, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        last_gemini_request = time.time()
    elif api_type == "groq":
        if current_time - last_groq_request < 60 / GROQ_RATE_LIMIT:
            sleep_time = 60 / GROQ_RATE_LIMIT - \
                (current_time - last_groq_request)
            logger.debug(
                f"Rate limiting Groq API, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        last_groq_request = time.time()
    else:
        logger.warning(f"Unknown API type for rate limiting: {api_type}")


def get_content_hash(content: str) -> str:
    """Generate a hash for the content."""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    logger.debug(f"Generated hash for content: {content_hash}")
    return content_hash


def load_cache(content_hash: str, cache_type: str) -> Optional[str]:
    """Load cache from MongoDB."""
    try:
        cache_item = cache_collection.find_one(
            {"content_hash": content_hash, "type": cache_type})
        if cache_item:
            logger.debug(
                f"Cache loaded for hash {content_hash} and type {cache_type}")
            return cache_item['data']
        logger.debug(
            f"No existing cache found for hash {content_hash} and type {cache_type}")
        return None
    except Exception as e:
        logger.error(f"Failed to load cache: {str(e)}")
        return None


def save_cache(content_hash: str, cache_type: str, data: str) -> None:
    """Save cache to MongoDB."""
    try:
        cache_collection.update_one(
            {"content_hash": content_hash, "type": cache_type},
            {"$set": {"data": data}},
            upsert=True
        )
        logger.debug(
            f"Cache saved for hash {content_hash} and type {cache_type}")
    except Exception as e:
        logger.error(f"Failed to save cache: {str(e)}")


def convert_paper_to_steps(paper_text: str) -> Optional[str]:
    """Convert paper text to steps, with caching and error handling."""
    content_hash = get_content_hash(paper_text)
    cached_steps = load_cache(content_hash, "steps")

    if cached_steps:
        logger.info("Steps loaded from cache")
        return cached_steps

    start_time = time.time()

    prompt = f"""
    You are the world's best researcher. You will be given a research paper and your task is to give a step-by-step list of instructions to implement the research paper.
    {paper_text}

    If it machine learning research paper then, Please generate a step-by-step list of instructions to implement the main ideas and algorithms described in this paper.
    Provide the output in the following format:
    - Steps: List of steps to implement the main ideas and algorithms described in this paper
    If it is not machine learning research paper then,
    - It is not research paper 
    """

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            rate_limit("gemini")
            model = genai.GenerativeModel('models/gemini-1.5-flash-001')
            result = model.generate_content(prompt)
            steps = result.text

            operation_time = time.time() - start_time
            logger.info(
                f"Paper processed successfully in {operation_time:.2f} seconds")

            save_cache(content_hash, "steps", steps)

            return steps
        except exceptions.InternalServerError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Internal server error, retrying in {retry_delay} seconds: {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to process paper after {max_retries} attempts: {str(e)}")
                raise
        except exceptions.ResourceExhausted as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing paper: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    return None


def steps_to_code(steps: str) -> Optional[str]:
    """Convert steps to code, with caching and error handling."""
    content_hash = get_content_hash(steps)
    cached_code = load_cache(content_hash, "code")

    if cached_code:
        logger.info("Code loaded from cache")
        return cached_code

    code_start_time = time.time()

    try:
        rate_limit("groq")
        code_creation = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f'''You are the world's best programmer. If You will be given a list of steps to implement a research paper. 
                                Please generate a Python code that implements the main ideas and algorithms described in this paper with example code.
                                You need to return only the code, no explanations or texts.
                                
                                List of steps: - {steps} 

                                Oherwise return "It is not research paper"
                                '''
                }
            ],
            model="llama3-8b-8192",
        )

        generated_code = code_creation.choices[0].message.content
        code_generation_time = time.time() - code_start_time
        logger.info(
            f"Code generated successfully in {code_generation_time:.2f} seconds")

        save_cache(content_hash, "code", generated_code)

        return generated_code
    except Exception as e:
        logger.error(f"Failed to generate code: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None


def process_paper(content: str, generate_steps: bool = True, generate_code: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Process paper content and return generated steps and/or code.

    Args:
        content (str): The paper content or existing steps.
        generate_steps (bool): Whether to generate steps from the content. Default is True.
        generate_code (bool): Whether to generate code from the steps. Default is True.

    Returns:
        Tuple[Optional[str], Optional[str]]: Generated steps and code, or None for each if generation fails.
    """
    try:
        steps = None
        code = None

        if generate_steps:
            logger.info("Generating steps from paper content")
            steps = convert_paper_to_steps(content)
            if steps is None:
                logger.warning("Failed to generate steps")
                return None, None
        else:
            steps = content  # Use the provided content as steps

        if generate_code:
            logger.info("Generating code from steps")
            code = steps_to_code(steps)
            if code is None:
                logger.warning("Failed to generate code")
                return steps, None

        logger.info("Paper processing completed successfully")
        return steps, code
    except Exception as e:
        logger.error(f"Error processing paper content: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return None, None


# if __name__ == "__main__":
#     # Example usage and testing
#     test_paper_content = "This is a test paper content."
#     logger.info("Testing with sample paper content")
#     steps, code = process_paper(test_paper_content)
#     if steps and code:
#         logger.info("Test successful: Steps and code generated")
#     else:
#         logger.error("Test failed: Unable to generate steps or code")
