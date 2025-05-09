import argparse
import json
import logging
import sys
import os # Added for ensure_dir_exists
from typing import List, Tuple, Dict, Optional, Any, Union

from datasets import Dataset, Features, Value, Sequence # Added for Dataset type hints
from pydantic import BaseModel, ValidationError, field_validator # Added for Pydantic

# --- Logger Setup ---
# Global logger instance, configured by setup_logging
logger = logging.getLogger("probir_pipeline") # Renamed for clarity

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level=logging.INFO, log_file_name: Optional[str] = None, log_format: str = DEFAULT_LOG_FORMAT):
    """
    Configures logging for the application.
    Logs to console and optionally to a file.
    """
    # Ensure logger is clean (important if this function is called multiple times)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.setLevel(level)
    formatter = logging.Formatter(log_format)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Changed from stderr to stdout
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (if log_file_name is provided)
    if log_file_name:
        # Ensure the directory for the log file exists
        log_dir = os.path.dirname(log_file_name)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Created log directory: {log_dir}")
            except OSError as e:
                # Use a basic print here as logger might not be fully set up or could recurse
                print(f"Error creating log directory {log_dir}: {e}", file=sys.stderr)
                # Fallback: don't use file logging if dir creation fails
                log_file_name = None # Prevent further errors with this handler

        if log_file_name: # Re-check in case it was set to None
            try:
                fh = logging.FileHandler(log_file_name, mode='a') # Append mode
                fh.setLevel(level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                logger.info(f"Logging to file: {log_file_name}")
            except Exception as e:
                print(f"Error setting up file handler for {log_file_name}: {e}", file=sys.stderr)
    
    logger.info(f"Logging initialized with level {logging.getLevelName(level)}.")


# --- Argument Parsers ---
def create_default_arg_parser(description: str) -> argparse.ArgumentParser:
    """Creates a default argument parser with common arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file (or DB file for create_hf_dataset.py).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: Number of examples to process.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default=None, # Default to None, meaning console-only logging unless specified
        help="Optional: Name of the log file. If provided, logs will be written to this file in the 'logs/' directory (e.g., 'my_script.log' becomes 'logs/my_script.log'). If None, only console logging.",
    )
    return parser

def create_llm_arg_parser(description: str) -> argparse.ArgumentParser:
    """Creates an argument parser with common LLM-related arguments, inheriting default ones."""
    parser = create_default_arg_parser(description)
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="phi3:mini", # A more reasonable default
        help="Name of the Ollama model to use (e.g., 'phi3:mini', 'llama3:8b').",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="http://localhost:11434", # Default to localhost
        help="URL of the Ollama API host (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4000,
        help="Size of text chunks for LLM processing (default: 4000 characters).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between text chunks for LLM processing (default: 200 characters).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4, # A more conservative default
        help="Maximum number of parallel workers for LLM processing (default: 4).",
    )
    return parser


# --- Dataset I/O ---
def load_jsonl_dataset(file_path: str, limit: Optional[int] = None) -> Dataset:
    """Loads a Hugging Face Dataset from a JSONL file."""
    try:
        if limit is not None:
            dataset = Dataset.from_json(file_path, split=f"train[:{limit}]")
            logger.info(f"Loaded {len(dataset)} examples (limited to {limit}) from {file_path}.")
        else:
            dataset = Dataset.from_json(file_path) # Corrected: was load_dataset, should be from_json
            logger.info(f"Loaded {len(dataset)} examples from {file_path}.")
        return dataset
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}", exc_info=True)
        raise

def save_jsonl_dataset(dataset: Dataset, file_path: str, force_ascii: bool = False):
    """Saves a Hugging Face Dataset to a JSONL file."""
    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True)
            raise # Re-raise to prevent writing to a potentially bad path

    try:
        dataset.to_json(file_path, force_ascii=force_ascii, lines=True)
        logger.info(f"Successfully saved {len(dataset)} examples to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving dataset to {file_path}: {e}", exc_info=True)
        raise

# --- Text Processing ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    if chunk_size <= chunk_overlap:
        logger.warning("Chunk size should be greater than chunk overlap. Returning single chunk.")
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap
        # Ensure we don't create an empty chunk if overlap is large and text is short
        if start >= len(text):
            break
            
    # logger.debug(f"Chunked text into {len(chunks)} chunks. Original length: {len(text)}, chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    return chunks

# --- Pydantic Models for Data Validation ---

class Message(BaseModel):
    role: str
    content: str

    @field_validator('role')
    @classmethod
    def role_must_be_known(cls, v: str) -> str:
        known_roles = {"system", "user", "assistant", "tool_code", "tool_outputs"} # Added tool roles
        if v not in known_roles:
            raise ValueError(f"Role must be one of {known_roles}, got '{v}'")
        return v

class BaseTrace(BaseModel):
    messages: List[Message]
    completion: str # This is the target completion for SFT

    # Optional fields that might be added by pipeline steps
    # These allow validation to pass even if they are not present initially
    # but will be validated if they are present.
    trace_id: Optional[str] = None # Example: if we add a unique ID later
    session_id: Optional[str] = None
    turn_in_session_id: Optional[int] = None
    original_messages: Optional[List[Message]] = None # For anonymization steps
    original_completion: Optional[str] = None      # For anonymization steps
    anonymization_details: Optional[Dict[str, Any]] = None
    llm_anonymization_details: Optional[Dict[str, Any]] = None
    complexity_score: Optional[float] = None
    complexity_reasoning: Optional[str] = None
    is_user_feedback_on_error: Optional[bool] = None
    is_direct_correction: Optional[bool] = None
    correction_similarity_score: Optional[float] = None
    correction_analysis_details: Optional[Dict[str, Any]] = None
    # Add other fields as they are defined by pipeline steps

    # Example of a model-level validator if needed
    # @model_validator(mode='after')
    # def check_consistency(self) -> 'BaseTrace':
    #     if self.anonymization_details and not self.original_messages:
    #         raise ValueError("If anonymization_details are present, original_messages must also be present.")
    #     return self

# --- Validation Function ---
def validate_dataset(dataset: Union[Dataset, List[Dict]], model: BaseModel, step_name: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Validates each example in a Hugging Face Dataset or list of dicts against a Pydantic model.
    Returns a tuple of (valid_examples, invalid_examples_with_errors).
    """
    valid_examples = []
    invalid_examples_with_errors = []
    
    # If it's a Hugging Face Dataset, convert to list of dicts for easier iteration
    if isinstance(dataset, Dataset):
        examples_to_validate = [dict(example) for example in dataset]
    elif isinstance(dataset, list) and all(isinstance(item, dict) for item in dataset):
        examples_to_validate = dataset
    else:
        logger.error(f"[{step_name}] Validation error: Input must be a Hugging Face Dataset or a list of dictionaries.")
        # Return all as invalid if the type is wrong, or handle as per desired strictness
        return [], [{"example": "Dataset type error", "errors": "Input must be a Dataset or list of dicts"}]


    for i, example_dict in enumerate(examples_to_validate):
        try:
            # Attempt to parse the dictionary using the Pydantic model
            model.model_validate(example_dict) # Pydantic v2
            valid_examples.append(example_dict)
        except ValidationError as e:
            logger.warning(f"[{step_name}] Validation failed for example {i}: {e.errors(include_url=False)}") # Pydantic v2 errors
            invalid_examples_with_errors.append({"example_index": i, "example_data": example_dict, "errors": e.errors(include_url=False)})
        except Exception as e_generic: # Catch any other unexpected errors during validation
            logger.error(f"[{step_name}] Unexpected error validating example {i}: {e_generic}", exc_info=True)
            invalid_examples_with_errors.append({"example_index": i, "example_data": example_dict, "errors": [{"type": "unexpected_error", "msg": str(e_generic)}]})


    if invalid_examples_with_errors:
        logger.warning(f"[{step_name}] Validation summary: {len(valid_examples)} valid, {len(invalid_examples_with_errors)} invalid examples.")
    else:
        logger.info(f"[{step_name}] Validation summary: All {len(valid_examples)} examples are valid.")
        
    return valid_examples, invalid_examples_with_errors


def ensure_dir_exists(dir_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        except OSError as e:
            logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True)
            raise # Re-raise to signal failure

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Setup logging for testing this module directly
    setup_logging(level=logging.DEBUG, log_file_name="logs/common_utils_test.log")

    # Test Pydantic model
    test_data_valid = {
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        "completion": "Hi there!"
    }
    test_data_invalid_role = {
        "messages": [{"role": "customer", "content": "Query"}], # Invalid role
        "completion": "Response"
    }
    test_data_missing_field = { # Missing 'completion'
        "messages": [{"role": "user", "content": "Test"}]
    }

    try:
        BaseTrace.model_validate(test_data_valid)
        logger.info("Valid test data parsed successfully by BaseTrace.")
    except ValidationError as e:
        logger.error(f"Error validating test_data_valid: {e.errors(include_url=False)}")

    try:
        BaseTrace.model_validate(test_data_invalid_role)
        logger.info("Invalid role test data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for invalid role: {e.errors(include_url=False)}")
    
    try:
        BaseTrace.model_validate(test_data_missing_field)
        logger.info("Missing field test data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for missing field: {e.errors(include_url=False)}")

    # Test dataset validation
    sample_dataset_list = [
        test_data_valid,
        test_data_invalid_role,
        {"messages": [{"role": "system", "content": "System prompt"}], "completion": "OK", "session_id": "sess123"}, # Valid with optional field
        test_data_missing_field
    ]
    
    logger.info("\nTesting validate_dataset function:")
    valid_items, invalid_items = validate_dataset(sample_dataset_list, BaseTrace, "TestStep")
    logger.info(f"From list: Valid items: {len(valid_items)}, Invalid items: {len(invalid_items)}")
    # for item in invalid_items:
    #     logger.debug(f"Invalid item details: {item}")

    # Test with Hugging Face Dataset
    try:
        hf_sample_dataset = Dataset.from_list(sample_dataset_list)
        valid_hf_items, invalid_hf_items = validate_dataset(hf_sample_dataset, BaseTrace, "TestStepHF")
        logger.info(f"From HF Dataset: Valid items: {len(valid_hf_items)}, Invalid items: {len(invalid_hf_items)}")
        # for item in invalid_hf_items:
        #     logger.debug(f"Invalid HF item details: {item}")
    except ImportError:
        logger.warning("Skipping Hugging Face Dataset test for validate_dataset as 'datasets' library might not be installed.")
    except Exception as e:
        logger.error(f"Error during HF Dataset test for validate_dataset: {e}", exc_info=True)
        
    # Test chunk_text
    logger.info("\nTesting chunk_text function:")
    example_text = "This is a test sentence to demonstrate the chunking functionality. It should be split into several overlapping parts."
    chunks = chunk_text(example_text, chunk_size=30, chunk_overlap=10)
    logger.info(f"Original text (len {len(example_text)}): '{example_text}'")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1} (len {len(chunk)}): '{chunk}'")

    chunks_edge = chunk_text("short", chunk_size=10, chunk_overlap=2)
    logger.info(f"Edge case 'short': {chunks_edge}")
    chunks_exact = chunk_text("1234567890", chunk_size=5, chunk_overlap=1) # Exact multiple
    logger.info(f"Edge case 'exact': {chunks_exact}")
    chunks_toolong_overlap = chunk_text("text", chunk_size=3, chunk_overlap=4)
    logger.info(f"Edge case 'too long overlap': {chunks_toolong_overlap}")

    # Test ensure_dir_exists
    ensure_dir_exists("logs/test_dir")
    ensure_dir_exists("data/test_dir/another_level")
    logger.info("ensure_dir_exists tests completed (check logs/ and data/ folders).")
