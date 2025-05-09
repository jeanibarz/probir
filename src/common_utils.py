import argparse
import json
import logging
import sys
import os
from typing import List, Tuple, Dict, Optional, Any, Union
from dotenv import load_dotenv # Added for .env file loading

from datasets import Dataset, Features, Value, Sequence # Added for Dataset type hints
from pydantic import BaseModel, ValidationError, field_validator # Added for Pydantic

# --- Logger Setup ---
# Global logger instance, configured by setup_logging
# Using "probir_pipeline" as the logger name for messages from common_utils itself,
# but setup_logging will configure the root logger.
logger = logging.getLogger("probir_pipeline") 

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level=logging.INFO, log_file_name: Optional[str] = None, log_format: str = DEFAULT_LOG_FORMAT):
    """
    Configures the root logger for the application.
    Logs to console and optionally to a file.
    """
    root_logger = logging.getLogger() # Get the root logger

    # If we want to avoid adding duplicate handlers, we could check existing ones.
    # For now, to fix caplog, we will not remove existing handlers.
    # This might lead to duplicate handlers if setup_logging is called multiple times
    # in a context other than isolated pytest tests.
    # However, pytest's caplog relies on its handler NOT being removed.
    
    # Set level on root logger. If called multiple times, this is fine.
    root_logger.setLevel(level)
    formatter = logging.Formatter(log_format)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Changed from stderr to stdout
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # File Handler (if log_file_name is provided)
    if log_file_name:
        # Determine the full path for the log file
        if os.path.isabs(log_file_name) or os.path.dirname(log_file_name): # If log_file_name already includes a path or is absolute
            full_log_path = log_file_name
        else: # log_file_name is just a filename, prepend default log_dir "logs"
            full_log_path = os.path.join("logs", log_file_name)

        log_dir_for_file = os.path.dirname(full_log_path)
        
        if log_dir_for_file and not os.path.exists(log_dir_for_file):
            try:
                os.makedirs(log_dir_for_file, exist_ok=True)
                # Temporarily use print for this specific info as logger might not have file handler yet
                print(f"INFO (pre-log): Created log directory: {log_dir_for_file}")
            except OSError as e:
                print(f"Error creating log directory {log_dir_for_file}: {e}", file=sys.stderr)
                full_log_path = None # Fallback: don't use file logging if dir creation fails

        if full_log_path:
            try:
                fh = logging.FileHandler(full_log_path, mode='a') # Append mode
                fh.setLevel(level)
                fh.setFormatter(formatter)
                root_logger.addHandler(fh)
                # This message will now go to the file as well, if setup is successful
                # Use the local 'logger' instance for messages from common_utils itself,
                # or directly use root_logger if preferred for this specific message.
                logging.getLogger().info(f"Logging to file: {full_log_path}") # Changed to root_logger for this message
            except Exception as e:
                print(f"Error setting up file handler for {full_log_path}: {e}", file=sys.stderr)
    
    logging.getLogger().info(f"Logging initialized with level {logging.getLevelName(level)}.") # Changed to root_logger


# --- Config & Secrets Loading Helper ---
def load_config_value(var_name: str, cli_value: Optional[Any], default_value: Optional[Any] = None, is_bool: bool = False) -> Optional[Any]:
    """
    Loads a configuration value based on a hierarchy:
    1. CLI argument (if provided and not None)
    2. Environment variable (uppercase var_name)
    3. Value from .env file (uppercase var_name)
    4. Default value
    Returns None if no value is found and no default is provided.
    For boolean flags, ENV/'.env' values 'true', '1', 'yes' are True; 'false', '0', 'no' are False.
    """
    load_dotenv() # Loads .env file into environment variables if .env exists

    # 1. CLI argument
    if cli_value is not None:
        # For boolean flags from argparse (action="store_true"/"store_false"),
        # cli_value will be True/False directly.
        # If it's a typed arg that could be None, this check is fine.
        return cli_value

    # 2. Environment variable (potentially loaded from .env)
    env_value_str = os.getenv(var_name.upper())
    if env_value_str is not None:
        if is_bool:
            if env_value_str.lower() in ['true', '1', 'yes', 'y']:
                return True
            elif env_value_str.lower() in ['false', '0', 'no', 'n']:
                return False
            # else, fall through to default if env var is not a recognized boolean string
        else:
            return env_value_str # Return as string, type conversion happens later if needed

    # 3. Default value (if no CLI or ENV var)
    return default_value

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
        default=None, # Default resolution will be handled by load_config_value
        help="Name of the Ollama model. Overrides OLLAMA_MODEL env var and .env file. (Default: 'phi3:mini' if not set elsewhere).",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=None, # Default resolution will be handled by load_config_value
        help="URL of the Ollama API host. Overrides OLLAMA_HOST env var and .env file. (Default: 'http://localhost:11434' if not set elsewhere).",
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
from datasets.exceptions import DatasetGenerationError # Ensure this is imported for specific handling

def load_jsonl_dataset(file_path: str, limit: Optional[int] = None) -> Dataset:
    """Loads a Hugging Face Dataset from a JSONL file with fallback for robust loading."""
    logger.info(f"Attempting to load dataset from {file_path}...")
    
    # Initial check for existence and emptiness using os functions
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path} (checked with os.path.exists before any loading attempt)")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        logger.warning(f"Input file {file_path} is empty (checked with os.path.getsize). Returning an empty dataset.")
        return Dataset.from_list([])

    full_dataset: Optional[Dataset] = None
    dataset_loaded_source: str = "unknown"

    try:
        full_dataset = Dataset.from_json(file_path)
        dataset_loaded_source = "Dataset.from_json"
        logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} using {dataset_loaded_source}.")
    
    except FileNotFoundError as e_ds_fnf:
        logger.warning(f"Dataset.from_json raised FileNotFoundError for {file_path}: {e_ds_fnf}")
        # Re-check existence, as Dataset.from_json might have its own view of the filesystem
        if os.path.exists(file_path): # Check again
            logger.warning(
                f"File {file_path} confirmed to exist by os.path.exists, "
                f"despite Dataset.from_json error. Attempting manual load."
            )
            try:
                records = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as jde:
                            logger.error(f"Manual load: JSONDecodeError in {file_path} at line {line_number}: {jde}. Skipping line.")
                            continue # Skip malformed lines
                
                if not records:
                    logger.warning(f"Manual load of existing file {file_path} resulted in no valid records. File might be empty or entirely malformed. Returning empty dataset.")
                    return Dataset.from_list([]) # Return empty, consistent with getsize() == 0
                
                full_dataset = Dataset.from_list(records)
                dataset_loaded_source = "manual JSONL parse"
                logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} using {dataset_loaded_source}.")
            except Exception as e_manual:
                logger.error(f"Manual load and parse of existing file {file_path} also failed. Error: {e_manual}", exc_info=True)
                raise e_ds_fnf from e_manual # Re-raise the original FileNotFoundError from datasets
        else:
            logger.error(f"File {file_path} confirmed NOT to exist by os.path.exists after Dataset.from_json error. Original error: {e_ds_fnf}")
            raise e_ds_fnf # Re-raise the original error

    except DatasetGenerationError as dge:
        cause_str = str(dge.__cause__).lower() if dge.__cause__ else ""
        main_err_str = str(dge).lower()
        if "schemainferenceerror" in cause_str or \
           "please pass `features` or at least one example" in cause_str or \
           ("empty" in main_err_str and "schema" in main_err_str):
            logger.warning(
                f"File {file_path} seems empty or schema cannot be inferred by Dataset.from_json. "
                f"Returning empty dataset. Original error: {dge}"
            )
            return Dataset.from_list([])
        else:
            logger.error(f"Error loading dataset from {file_path} via Dataset.from_json: {dge}", exc_info=True)
            raise
    
    except Exception as e: # Catch other unexpected errors from Dataset.from_json or initial checks
        logger.error(f"Unexpected error loading dataset from {file_path}: {e}", exc_info=True)
        raise

    # Apply limit if dataset was successfully loaded (either way)
    if full_dataset is None:
        # This should ideally not be reached if errors are re-raised properly, but as a safeguard:
        logger.error(f"Dataset loading failed for {file_path} and full_dataset is None. Raising FileNotFoundError as a fallback.")
        raise FileNotFoundError(f"Failed to load dataset from {file_path}, result was None.")

    if limit is not None and limit > 0 and limit < len(full_dataset):
        final_dataset = full_dataset.select(range(limit))
        logger.info(f"Returning {len(final_dataset)} examples (limited to {limit} from {len(full_dataset)} total, loaded via {dataset_loaded_source}) from {file_path}.")
    else:
        final_dataset = full_dataset
        log_msg_limit_part = f"(limit was {limit})" if limit is not None else "(no limit)"
        logger.info(f"Returning all {len(final_dataset)} examples {log_msg_limit_part}, loaded via {dataset_loaded_source} from {file_path}.")
    
    return final_dataset


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
        known_roles = {"system", "user", "assistant", "tool_code", "tool_outputs"}
        if v not in known_roles:
            raise ValueError(f"Role must be one of {known_roles}, got '{v}'")
        return v

# Base model for input to the pipeline (after trace_id generation)
class BasePipelineInput(BaseModel):
    trace_id: str  # Mandatory unique identifier for the trace
    schema_version: str # Version of the data schema
    messages: List[Message]
    completion: str  # This is the target completion for SFT

# Output schema after Step 1: Session Identification (src/step2b_identify_sessions.py)
class SessionIdentificationOutput(BasePipelineInput):
    session_id: str
    turn_in_session_id: int

# Output schema after Step 2: Regex-based Anonymization (src/step1_anonymize_data.py)
class RegexAnonymizationOutput(SessionIdentificationOutput):
    messages: List[Message]  # Content within messages is now potentially anonymized
    completion: str          # Completion is now potentially anonymized
    original_messages: Optional[List[Message]] = None
    original_completion: Optional[str] = None
    anonymization_details: Dict[str, Any] # Details about regex patterns found

# Output schema after Step 3: LLM-based Anonymization (src/step1b_anonymize_llm.py)
class LlmAnonymizationOutput(RegexAnonymizationOutput):
    messages: List[Message]  # Content potentially further anonymized by LLM
    completion: str          # Completion potentially further anonymized by LLM
    # original_messages and original_completion are inherited and updated if necessary
    llm_anonymization_details: Dict[str, Any] # Details about LLM PII detection

# Output schema after Step 4: Heuristic Complexity Scoring (src/step2_score_complexity.py)
class ComplexityScoringOutput(LlmAnonymizationOutput):
    complexity_score: float
    complexity_reasoning: str
    # Optional fields for future LLM-based complexity scoring
    llm_complexity_score: Optional[float] = None
    llm_complexity_rationale: Optional[str] = None

# Output schema after Step 5: Feedback/Correction Pattern Analysis (src/step3_analyze_correction_patterns.py)
# This can be considered the schema for the final output of the current pipeline.
class CorrectionAnalysisOutput(ComplexityScoringOutput):
    is_user_feedback_on_error: bool
    is_assistant_self_correction: bool
    is_assistant_error_before_correction: bool
    correction_rationale: Optional[str] = None
    correction_analysis_details: Dict[str, Any] # Details from LLM analysis of corrections

    # The fields `is_direct_correction` and `correction_similarity_score` from the old BaseTrace
    # were not found to be populated by current scripts, so they are omitted here.
    # If they are needed, they can be added back.

# --- Pydantic Models for Pipeline Configuration ---

class StepInputConfig(BaseModel):
    main: str

class StepOutputConfig(BaseModel):
    main: str

class StepConfig(BaseModel):
    name: str
    script: str
    enabled: bool = True
    inputs: StepInputConfig
    outputs: StepOutputConfig
    args: List[Any] = []
    description: Optional[str] = None

class PipelineConfig(BaseModel):
    pipeline_name: str
    default_base_input: str
    steps: List[StepConfig]

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
    if not dir_path:
        return # Do nothing if path is empty or None

    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            err_msg = f"Path exists but is not a directory: {dir_path}"
            logger.error(err_msg)
            raise FileExistsError(err_msg) # More specific error
        # Path exists and is a directory, do nothing further
        return
    
    # Path does not exist, try to create it
    try:
        os.makedirs(dir_path, exist_ok=True) # exist_ok=True is good for multi-level creation
        logger.info(f"Created directory: {dir_path}")
    except OSError as e: # Catches FileExistsError if a component of path is a file during makedirs
        logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True)
        raise # Re-raise to signal failure

# Example usage (can be removed or kept for testing)
if __name__ == '__main__':
    # Setup logging for testing this module directly
    setup_logging(level=logging.DEBUG, log_file_name="logs/common_utils_test.log")

    # Test Pydantic model
    test_data_valid_base_input = {
        "trace_id": "test-trace-001",
        "schema_version": "1.0",
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        "completion": "Hi there!"
    }
    test_data_invalid_role_base_input = {
        "trace_id": "test-trace-002",
        "schema_version": "1.0",
        "messages": [{"role": "customer", "content": "Query"}], # Invalid role
        "completion": "Response"
    }
    test_data_missing_completion_base_input = { # Missing 'completion'
        "trace_id": "test-trace-003",
        "schema_version": "1.0",
        "messages": [{"role": "user", "content": "Test"}]
    }
    test_data_missing_trace_id = {
        "schema_version": "1.0",
        "messages": [{"role": "user", "content": "Test"}],
        "completion": "OK"
    }
    test_data_missing_schema_version = {
        "trace_id": "test-trace-004",
        "messages": [{"role": "user", "content": "Test"}],
        "completion": "OK"
    }


    try:
        BasePipelineInput.model_validate(test_data_valid_base_input)
        logger.info("Valid base input data parsed successfully by BasePipelineInput.")
    except ValidationError as e:
        logger.error(f"Error validating test_data_valid_base_input: {e.errors(include_url=False)}")

    try:
        BasePipelineInput.model_validate(test_data_invalid_role_base_input)
        logger.info("Invalid role base input data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for invalid role (BasePipelineInput): {e.errors(include_url=False)}")
    
    try:
        BasePipelineInput.model_validate(test_data_missing_completion_base_input)
        logger.info("Missing completion base input data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for missing completion (BasePipelineInput): {e.errors(include_url=False)}")

    try:
        BasePipelineInput.model_validate(test_data_missing_trace_id)
        logger.info("Missing trace_id data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for missing trace_id (BasePipelineInput): {e.errors(include_url=False)}")
    
    try:
        BasePipelineInput.model_validate(test_data_missing_schema_version)
        logger.info("Missing schema_version data parsed (this should not happen).")
    except ValidationError as e:
        logger.info(f"Correctly caught validation error for missing schema_version (BasePipelineInput): {e.errors(include_url=False)}")


    # Test dataset validation using BasePipelineInput for basic structure
    sample_dataset_list_for_base = [
        test_data_valid_base_input,
        test_data_invalid_role_base_input,
        test_data_missing_trace_id,
        test_data_missing_schema_version
    ]
    
    logger.info("\nTesting validate_dataset function with BasePipelineInput:")
    valid_items, invalid_items = validate_dataset(sample_dataset_list_for_base, BasePipelineInput, "TestStepBase")
    logger.info(f"From list (BasePipelineInput): Valid items: {len(valid_items)}, Invalid items: {len(invalid_items)}")
    # for item in invalid_items:
    #     logger.debug(f"Invalid item details: {item}")

    # Test with Hugging Face Dataset
    try:
        hf_sample_dataset_base = Dataset.from_list(sample_dataset_list_for_base)
        valid_hf_items, invalid_hf_items = validate_dataset(hf_sample_dataset_base, BasePipelineInput, "TestStepHFBase")
        logger.info(f"From HF Dataset (BasePipelineInput): Valid items: {len(valid_hf_items)}, Invalid items: {len(invalid_hf_items)}")
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
