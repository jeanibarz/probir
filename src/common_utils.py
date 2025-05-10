import argparse
import json
import logging
import sys
import os
import copy # Added for deepcopy
from typing import List, Tuple, Dict, Optional, Any, Union
from dotenv import load_dotenv # Added for .env file loading

from datasets import Dataset, Features as DatasetFeatures, Value, Sequence, load_dataset # Renamed Features to avoid conflict, Added for Dataset type hints
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
    # Check if a console handler with stdout is already present to avoid duplicates if called multiple times
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in root_logger.handlers
    )
    if not has_stdout_handler:
        ch = logging.StreamHandler(sys.stdout) 
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)

    # File Handler (if log_file_name is provided)
    if log_file_name:
        # Determine the full path for the log file
        if os.path.isabs(log_file_name) or os.path.dirname(log_file_name): 
            full_log_path = log_file_name
        else: 
            full_log_path = os.path.join("logs", log_file_name)

        log_dir_for_file = os.path.dirname(full_log_path)
        
        if log_dir_for_file and not os.path.exists(log_dir_for_file):
            try:
                os.makedirs(log_dir_for_file, exist_ok=True)
                print(f"INFO (pre-log): Created log directory: {log_dir_for_file}")
            except OSError as e:
                print(f"Error creating log directory {log_dir_for_file}: {e}", file=sys.stderr)
                full_log_path = None 

        if full_log_path:
            # Check if a file handler for this path already exists
            has_this_file_handler = any(
                h.__class__.__name__ == 'FileHandler' and hasattr(h, 'baseFilename') and os.path.abspath(h.baseFilename) == os.path.abspath(full_log_path)
                for h in root_logger.handlers
            )
            if not has_this_file_handler:
                try:
                    fh = logging.FileHandler(full_log_path, mode='a') 
                    fh.setLevel(level)
                    fh.setFormatter(formatter)
                    root_logger.addHandler(fh)
                    logging.getLogger().info(f"Logging to file: {full_log_path}") 
                except Exception as e:
                    print(f"Error setting up file handler for {full_log_path}: {e}", file=sys.stderr)
            # else: # Optional: log if handler already exists
                # logging.getLogger().debug(f"File handler for {full_log_path} already exists.")
    
    logging.getLogger().info(f"Logging initialized with level {logging.getLevelName(level)}.")


# --- Config & Secrets Loading Helper ---
def load_config_value(var_name: str, cli_value: Optional[Any], default_value: Optional[Any] = None, is_bool: bool = False) -> Optional[Any]:
    load_dotenv() 
    if cli_value is not None:
        return cli_value
    env_value_str = os.getenv(var_name.upper())
    if env_value_str is not None:
        if is_bool:
            if env_value_str.lower() in ['true', '1', 'yes', 'y']: return True
            elif env_value_str.lower() in ['false', '0', 'no', 'n']: return False
        else: return env_value_str 
    return default_value

# --- Argument Parsers ---
def create_default_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Number of examples to process.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    parser.add_argument("--log_file_name", type=str, default=None, help="Optional: Name of the log file.")
    return parser

def create_llm_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = create_default_arg_parser(description)
    parser.add_argument("--ollama_model", type=str, default=None, help="Name of the Ollama model.")
    parser.add_argument("--ollama_host", type=str, default=None, help="URL of the Ollama API host.")
    parser.add_argument("--chunk_size", type=int, default=4000, help="Size of text chunks for LLM processing.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between text chunks.")
    parser.add_argument("--max_workers", type=int, default=4, help="Max parallel workers for LLM processing.")
    return parser


# --- Dataset I/O ---
from datasets.exceptions import DatasetGenerationError 

def load_jsonl_dataset(file_path: str, limit: Optional[int] = None, features: Optional[DatasetFeatures] = None) -> Dataset:
    logger.info(f"Attempting to load dataset from {file_path}...")
    if features:
        logger.info(f"Explicit features provided for loading.") # Removed features object from log for brevity
    
    if not os.path.exists(file_path):
        logger.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        logger.warning(f"Input file {file_path} is empty. Returning an empty dataset.")
        return Dataset.from_list([])

    full_dataset: Optional[Dataset] = None
    dataset_loaded_source: str = "unknown"

    try:
        if features:
            logger.info(f"Attempting load_dataset for {file_path} with explicit features.")
            try:
                full_dataset = load_dataset("json", data_files=file_path, features=features, split="train")
                dataset_loaded_source = "load_dataset (json script with explicit features)"
                logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} using {dataset_loaded_source}.")
            except Exception as e_load_dataset: 
                logger.warning(f"load_dataset('json', ...) failed for {file_path} with features: {e_load_dataset}. Attempting manual parse then Dataset.from_list.")
                try:
                    records = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_number, line in enumerate(f, 1):
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError as jde:
                                logger.error(f"Manual parse for from_list: JSONDecodeError in {file_path} at line {line_number}: {jde}. Skipping line.")
                                continue
                    if not records:
                        logger.warning(f"Manual parse for from_list of {file_path} resulted in no records. Returning empty dataset with provided features.")
                        full_dataset = Dataset.from_list([], features=features) 
                    else:
                        full_dataset = Dataset.from_list(records, features=features)
                    
                    dataset_loaded_source = "manual JSONL parse + Dataset.from_list (fallback for features)"
                    logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} using {dataset_loaded_source}.")

                except Exception as e_from_list:
                    logger.error(f"Manual parse + Dataset.from_list also failed for {file_path} with features: {e_from_list}", exc_info=True)
                    raise e_load_dataset 
        else:
            full_dataset = Dataset.from_json(file_path) 
            dataset_loaded_source = "Dataset.from_json (schema inference)"
            logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} using {dataset_loaded_source} with schema inference.")
    
    except FileNotFoundError as e_ds_fnf:
        logger.warning(f"Dataset loading raised FileNotFoundError for {file_path}: {e_ds_fnf}")
        if os.path.exists(file_path): 
            logger.warning(f"File {file_path} confirmed to exist. Attempting manual load.")
            try:
                records = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, 1):
                        try: records.append(json.loads(line))
                        except json.JSONDecodeError as jde: 
                            logger.error(f"Manual load: JSONDecodeError in {file_path} at line {line_number}: {jde}. Skipping.")
                            continue
                if not records:
                    logger.warning(f"Manual load of {file_path} yielded no records. Returning empty dataset.")
                    return Dataset.from_list([], features=features if features else None)
                full_dataset = Dataset.from_list(records, features=features if features else None)
                dataset_loaded_source = "manual JSONL parse (after FNF)"
                logger.info(f"Successfully loaded {len(full_dataset)} records from {file_path} via manual parse.")
            except Exception as e_manual:
                logger.error(f"Manual load of {file_path} also failed: {e_manual}", exc_info=True)
                raise e_ds_fnf from e_manual
        else:
            logger.error(f"File {file_path} confirmed NOT to exist. Original error: {e_ds_fnf}")
            raise e_ds_fnf

    except DatasetGenerationError as dge:
        cause_str = str(dge.__cause__).lower() if dge.__cause__ else ""
        main_err_str = str(dge).lower()
        if "schemainferenceerror" in cause_str or \
           "please pass `features` or at least one example" in cause_str or \
           ("empty" in main_err_str and "schema" in main_err_str):
            logger.warning(f"File {file_path} seems empty or schema cannot be inferred. Returning empty dataset. Error: {dge}")
            return Dataset.from_list([], features=features if features else None)
        else:
            logger.error(f"Error loading dataset from {file_path}: {dge}", exc_info=True)
            raise
    
    except Exception as e: 
        logger.error(f"Unexpected error loading dataset from {file_path}: {e}", exc_info=True)
        raise

    if full_dataset is None:
        logger.error(f"Dataset loading failed for {file_path}, full_dataset is None. Raising error.")
        raise RuntimeError(f"Failed to load dataset from {file_path}, result was None after all attempts.")

    if limit is not None and limit > 0 and limit < len(full_dataset):
        final_dataset = full_dataset.select(range(limit))
        logger.info(f"Returning {len(final_dataset)} examples (limited from {len(full_dataset)}, loaded via {dataset_loaded_source}) from {file_path}.")
    else:
        final_dataset = full_dataset
        logger.info(f"Returning all {len(final_dataset)} examples (loaded via {dataset_loaded_source}) from {file_path}.")
    
    return final_dataset


def save_jsonl_dataset(dataset: Dataset, file_path: str, force_ascii: bool = False):
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        try: os.makedirs(output_dir, exist_ok=True); logger.info(f"Created output directory: {output_dir}")
        except OSError as e: logger.error(f"Error creating output directory {output_dir}: {e}", exc_info=True); raise 
    try:
        dataset.to_json(file_path, force_ascii=force_ascii, lines=True)
        logger.info(f"Successfully saved {len(dataset)} examples to {file_path}.")
    except Exception as e: logger.error(f"Error saving dataset to {file_path}: {e}", exc_info=True); raise

# --- Text Processing ---
def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text: return []
    if chunk_size <= chunk_overlap: logger.warning("Chunk size <= overlap. Returning single chunk."); return [text]
    chunks = []; start = 0
    while start < len(text):
        end = start + chunk_size; chunks.append(text[start:end])
        if end >= len(text): break
        start += chunk_size - chunk_overlap
        if start >= len(text): break
    return chunks

# --- Pydantic Models for Data Validation ---
class Message(BaseModel):
    role: str
    content: str
    @field_validator('role')
    @classmethod
    def role_must_be_known(cls, v: str) -> str:
        if v not in {"system", "user", "assistant", "tool_code", "tool_outputs"}:
            raise ValueError(f"Role must be known, got '{v}'")
        return v

class BasePipelineInput(BaseModel):
    trace_id: str; schema_version: str; messages: List[Message]; completion: str
class SessionIdentificationOutput(BasePipelineInput):
    session_id: str; turn_in_session_id: int
class RegexAnonymizationOutput(SessionIdentificationOutput):
    messages: List[Message]; completion: str
    original_messages: Optional[List[Message]] = None
    original_completion: Optional[str] = None
    anonymization_details: Dict[str, Any]
class LlmAnonymizationOutput(RegexAnonymizationOutput):
    messages: List[Message]; completion: str
    llm_anonymization_details: Dict[str, Any]
class ComplexityScoringOutput(LlmAnonymizationOutput):
    complexity_score: float; complexity_reasoning: str
    llm_complexity_score: Optional[float] = None
    llm_complexity_rationale: Optional[str] = None
class CorrectionAnalysisOutput(ComplexityScoringOutput):
    is_user_feedback_on_error: bool; is_assistant_self_correction: bool
    is_assistant_error_before_correction: bool
    correction_rationale: Optional[str] = None
    correction_analysis_details: Dict[str, Any]

# --- Dataset Feature Schemas ---
llm_anonymization_output_features = DatasetFeatures({
    "trace_id": Value("string"), "schema_version": Value("string"),
    "messages": Sequence(feature=DatasetFeatures({'role': Value('string'), 'content': Value('string')})),
    "completion": Value("string"), "session_id": Value("string"), "turn_in_session_id": Value("int32"),
    "original_messages": Sequence(feature=DatasetFeatures({'role': Value('string'), 'content': Value('string')})),
    "original_completion": Value("string"),
    "anonymization_details": DatasetFeatures({'regex_patterns_found': Sequence(Value('string')), 'regex_categories_found': Sequence(Value('string'))}),
    "llm_anonymization_details": DatasetFeatures({
        'llm_sensitive_categories_found': Sequence(Value('string')),
        'llm_detected_pii_items_messages': Sequence(feature=DatasetFeatures({'category': Value('string'), 'value': Value('string')})),
        'llm_detected_pii_items_completion': Sequence(feature=DatasetFeatures({'category': Value('string'), 'value': Value('string')}))
    })
})

# --- Pydantic Models for Pipeline Configuration ---
class StepInputConfig(BaseModel): main: str
class StepOutputConfig(BaseModel): main: str
class StepConfig(BaseModel):
    name: str; script: str; enabled: bool = True
    inputs: StepInputConfig; outputs: StepOutputConfig
    args: List[Any] = []; description: Optional[str] = None
class PipelineConfig(BaseModel):
    pipeline_name: str; default_base_input: str; steps: List[StepConfig]

# --- Validation Function Helper ---
def _convert_transposed_messages_if_needed(data_dict: Dict[str, Any], field_name: str) -> None:
    """
    Checks if data_dict[field_name] is in a transposed format (dict of lists from Hugging Face Datasets)
    and converts it to a list of dicts (Pydantic Message model format) in place.
    Handles cases where the field might be None, already a list of dicts, or not present.
    """
    if field_name not in data_dict:
        return

    field_value = data_dict[field_name]

    if isinstance(field_value, dict) and \
       'role' in field_value and 'content' in field_value and \
       isinstance(field_value['role'], list) and \
       isinstance(field_value['content'], list):
        
        if len(field_value['role']) != len(field_value['content']):
            logger.warning(
                f"Field '{field_name}' in trace_id '{data_dict.get('trace_id', 'N/A')}' "
                f"appears transposed but 'role' and 'content' lists have different lengths "
                f"({len(field_value['role'])} vs {len(field_value['content'])}). "
                f"Skipping conversion for this field. Pydantic validation will likely fail."
            )
            return 

        actual_messages_list = []
        for r, c in zip(field_value['role'], field_value['content']):
            actual_messages_list.append({'role': r, 'content': c})
        
        data_dict[field_name] = actual_messages_list
        logger.debug(f"Converted transposed field '{field_name}' for trace_id '{data_dict.get('trace_id', 'N/A')}' to list of dicts.")

# --- Validation Function ---
def validate_dataset(dataset: Union[Dataset, List[Dict]], model: BaseModel, step_name: str) -> Tuple[List[Dict], List[Dict]]:
    valid_examples = []; invalid_examples_with_errors = []
    
    if isinstance(dataset, Dataset):
        examples_to_validate = [dict(example) for example in dataset]
    elif isinstance(dataset, list) and all(isinstance(item, dict) for item in dataset):
        examples_to_validate = dataset
    else:
        logger.error(f"[{step_name}] Validation error: Input must be a Dataset or list of dicts.")
        return [], [{"example": "Dataset type error", "errors": "Input must be a Dataset or list of dicts"}]

    for i, example_dict_original in enumerate(examples_to_validate):
        example_dict_for_validation = copy.deepcopy(example_dict_original)
        try:
            _convert_transposed_messages_if_needed(example_dict_for_validation, 'messages')
            _convert_transposed_messages_if_needed(example_dict_for_validation, 'original_messages')
            
            model.model_validate(example_dict_for_validation) 
            valid_examples.append(example_dict_original) 
        except ValidationError as e:
            trace_id_info = example_dict_original.get('trace_id', 'N/A')
            logger.warning(f"[{step_name}] Validation failed for example {i} (trace_id: {trace_id_info}): {e.errors(include_url=False)}")
            
            messages_data = example_dict_for_validation.get('messages', 'Field not present')
            original_messages_data = example_dict_for_validation.get('original_messages', 'Field not present')
            
            try: messages_debug_str = json.dumps(messages_data, indent=2) if not isinstance(messages_data, str) else messages_data
            except TypeError: messages_debug_str = str(messages_data)
            try: original_messages_debug_str = json.dumps(original_messages_data, indent=2) if not isinstance(original_messages_data, str) else original_messages_data
            except TypeError: original_messages_debug_str = str(original_messages_data)

            logger.debug(f"Data for 'messages' field passed to Pydantic (trace_id: {trace_id_info}): {messages_debug_str}")
            logger.debug(f"Data for 'original_messages' field passed to Pydantic (trace_id: {trace_id_info}): {original_messages_debug_str}")
            
            invalid_examples_with_errors.append({"example_index": i, "example_data": example_dict_original, "errors": e.errors(include_url=False)})
        except Exception as e_generic: 
            logger.error(f"[{step_name}] Unexpected error validating example {i} (trace_id: {example_dict_original.get('trace_id', 'N/A')}): {e_generic}", exc_info=True)
            invalid_examples_with_errors.append({"example_index": i, "example_data": example_dict_original, "errors": [{"type": "unexpected_error", "msg": str(e_generic)}]})

    if invalid_examples_with_errors:
        logger.warning(f"[{step_name}] Validation summary: {len(valid_examples)} valid, {len(invalid_examples_with_errors)} invalid examples.")
    else:
        logger.info(f"[{step_name}] Validation summary: All {len(valid_examples)} examples are valid.")
        
    return valid_examples, invalid_examples_with_errors


def ensure_dir_exists(dir_path: str):
    if not dir_path: return
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            err_msg = f"Path exists but is not a directory: {dir_path}"; logger.error(err_msg); raise FileExistsError(err_msg)
        return
    try: os.makedirs(dir_path, exist_ok=True); logger.info(f"Created directory: {dir_path}")
    except OSError as e: logger.error(f"Error creating directory {dir_path}: {e}", exc_info=True); raise 

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
