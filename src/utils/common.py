import requests
import logging

logger = logging.getLogger(__name__)

FASTAPI_ENDPOINT = "http://127.0.0.1:8000/query"


def query_fastapi_backend(query: str, timeout: int = 300) -> dict:
    """
    Sends a POST request to the FastAPI backend and parses the response.
    Includes graceful error handling for network issues.
    """
    payload = {"query": query}

    try:
        response = requests.post(FASTAPI_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()

    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to FastAPI backend.")
        return {"error": "Connection to backend failed. Please ensure the API is running."}
    except requests.exceptions.Timeout:
        logger.error("FastAPI backend timed out.")
        return {"error": "Request timed out. The model might be loading or busy."}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}