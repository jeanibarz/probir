import pytest
import os
import shutil
import argparse
import json # For creating test JSONL files
from unittest.mock import patch, MagicMock
from datasets import Dataset # For creating test datasets

from common_utils import (
    chunk_text, 
    ensure_dir_exists, 
    create_default_arg_parser, 
    create_llm_arg_parser,
    load_jsonl_dataset,
    save_jsonl_dataset,
    setup_logging,
    validate_dataset, # Added for testing
    Message,          # Needed for validate_dataset tests
    BasePipelineInput # Needed for validate_dataset tests
)
import logging
import sys 
import common_utils 
from pydantic import ValidationError # For Pydantic model tests (though validate_dataset handles it)
# DatasetGenerationError might not be needed anymore if test_load_jsonl_dataset_empty_file changes
# from datasets.exceptions import DatasetGenerationError 

def test_chunk_text_empty():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == ["   "] # Test with whitespace only

def test_chunk_text_single_chunk():
    text = "This is a short text."
    chunks = chunk_text(text, 100, 10)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_multiple_chunks_no_overlap():
    text = "This is a longer text that will be split into multiple chunks."
    #         012345678901234567890123456789012345678901234567890123456789
    #         0         1         2         3         4         5
    chunk_size = 20
    overlap = 0
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 4
    assert chunks[0] == "This is a longer tex"
    assert chunks[1] == "t that will be split"
    assert chunks[2] == " into multiple chunk" # Corrected expected text
    assert chunks[3] == "s." # Added 4th chunk assertion

def test_chunk_text_multiple_chunks_with_overlap():
    text = "This is a longer text that will be split into multiple chunks with some overlap."
    #         01234567890123456789012345678901234567890123456789012345678901234567890123456789
    #         0         1         2         3         4         5         6         7
    chunk_size = 30
    overlap = 10
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Expected:
    # Chunk 1 (0-30): "This is a longer text that wil" (start: 0)
    # Next start: 30 - 10 = 20
    # Chunk 2 (20-50): "l be split into multiple chun" (start: 20)
    # Next start: 50 - 10 = 40
    # Chunk 3 (40-70): "ks with some overlap." (start: 40)
    # Next start: 70 - 10 = 60
    # Chunk 4 (60-eof): " overlap." (start: 60) -> text[60:78]

    assert len(chunks) == 4
    assert chunks[0] == text[0:30] # "This is a longer text that wil"
    assert chunks[1] == text[20:50] # "l be split into multiple chun"
    assert chunks[2] == text[40:70] # "ks with some overlap."
    assert chunks[3] == text[60:]   # " overlap."

def test_chunk_text_exact_multiple_no_overlap():
    text = "onetwothreefour" # 15 chars
    chunk_size = 5
    overlap = 0
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == "onetw"
    assert chunks[1] == "othre"
    assert chunks[2] == "efour"

def test_chunk_text_exact_multiple_with_overlap():
    text = "onetwothreefourfive" # 19 chars
    chunk_size = 7
    overlap = 3 # step = 4
    # 0-7: onetwot (0)
    # 4-11: wothree (4)
    # 8-15: reefour (8)
    # 12-19: urfive (12)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 4
    assert chunks[0] == text[0:7]
    assert chunks[1] == text[4:11]
    assert chunks[2] == text[8:15]
    assert chunks[3] == text[12:19]

def test_chunk_text_chunk_size_larger_than_text():
    text = "short"
    chunks = chunk_text(text, 10, 2)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_invalid_chunk_size():
    # Expect warning and returning the original text as a single chunk
    assert chunk_text("some text", 0, 0) == ["some text"]
    assert chunk_text("some text", -5, 0) == ["some text"]

def test_chunk_text_invalid_overlap():
    # Expect warning or specific behavior based on current implementation
    # chunk_size <= chunk_overlap is the main guard.
    # For chunk_text("some text", 10, -1), 10 <= -1 is False. It will chunk.
    # Expected: ["some text"] because step will be 11, so only one chunk.
    assert chunk_text("some text", 10, -1) == ["some text"]
    # For chunk_text("some text", 10, 10), 10 <= 10 is True. Returns [text].
    assert chunk_text("some text", 10, 10) == ["some text"]
    # For chunk_text("some text", 10, 11), 10 <= 11 is True. Returns [text].
    assert chunk_text("some text", 10, 11) == ["some text"]

def test_chunk_text_unicode():
    text = "こんにちは世界" # Hello World in Japanese
    chunk_size = 3
    overlap = 1
    # Chars: こ ん に ち は 世 界
    # 0-3: こんに (0)
    # 2-5: にちは (2)
    # 4-7: は世界 (4)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == "こんに"
    assert chunks[1] == "にちは" # Start index is by character, not byte
    assert chunks[2] == "は世界"

def test_chunk_text_overlap_equals_chunk_size_minus_one():
    text = "abcdefghij" # len 10
    chunk_size = 5
    overlap = 4 # step = 1
    # 0-5: abcde (0)
    # 1-6: bcdef (1)
    # 2-7: cdefg (2)
    # 3-8: defgh (3)
    # 4-9: efghi (4)
    # 5-10: fghij (5)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 6
    assert chunks[0] == "abcde"
    assert chunks[1] == "bcdef"
    assert chunks[2] == "cdefg"
    assert chunks[3] == "defgh"
    assert chunks[4] == "efghi"
    assert chunks[5] == "fghij"

def test_chunk_text_final_chunk_smaller_than_chunk_size():
    text = "This is a test text." # 20 chars
    chunk_size = 8
    overlap = 2 # step = 6
    # 0-8: This is  (0)
    # 6-14: is a tes (6)
    # 12-20: test text. (12)
    chunks = chunk_text(text, chunk_size, overlap)
    assert len(chunks) == 3
    assert chunks[0] == text[0:8]
    assert chunks[1] == text[6:14]
    assert chunks[2] == text[12:20]
    assert len(chunks[2]) == 8

    text2 = "This is a test text A" # 21 chars
    # 0-8: This is 
    # 6-14: is a tes
    # 12-20: test tex
    # 18-21: xt A
    chunks2 = chunk_text(text2, chunk_size, overlap) # step = 6
    # text2[0:8] = "This is "
    # text2[6:14] = "is a tes"
    # text2[12:20] = "test tex"
    # text2[18:21] = "xt A"
    assert len(chunks2) == 4
    assert chunks2[0] == text2[0:8]
    assert chunks2[1] == text2[6:14]
    assert chunks2[2] == text2[12:20]
    assert chunks2[3] == text2[18:21]
    assert len(chunks2[3]) == 3

# --- Tests for ensure_dir_exists ---

TEST_DIR_ROOT = "temp_test_ensure_dir"

@pytest.fixture(autouse=True)
def cleanup_test_dirs():
    """Clean up any created test directories after each test."""
    if os.path.exists(TEST_DIR_ROOT):
        shutil.rmtree(TEST_DIR_ROOT)
    yield # Test runs here
    if os.path.exists(TEST_DIR_ROOT):
        shutil.rmtree(TEST_DIR_ROOT)

@patch('common_utils.logger')
def test_ensure_dir_exists_creates_new_directory(mock_logger):
    test_path = os.path.join(TEST_DIR_ROOT, "new_dir")
    assert not os.path.exists(test_path)
    ensure_dir_exists(test_path)
    assert os.path.exists(test_path)
    assert os.path.isdir(test_path)
    mock_logger.info.assert_called_with(f"Created directory: {test_path}")

@patch('common_utils.logger')
def test_ensure_dir_exists_existing_directory(mock_logger):
    test_path = os.path.join(TEST_DIR_ROOT, "existing_dir")
    os.makedirs(test_path, exist_ok=True) # Pre-create the directory
    assert os.path.exists(test_path)
    ensure_dir_exists(test_path) # Should not raise error or try to re-create
    assert os.path.exists(test_path) # Still exists
    assert os.path.isdir(test_path)
    # Assert that logger.info was not called with "Created directory..."
    # This requires checking that a specific call was *not* made,
    # or that the call count for info is 0 if no other info logs are expected.
    # For simplicity, we'll check that the specific "Created directory" message wasn't logged.
    for call_args in mock_logger.info.call_args_list:
        assert f"Created directory: {test_path}" not in call_args[0][0]


@patch('common_utils.logger')
def test_ensure_dir_exists_creates_nested_directories(mock_logger):
    test_path = os.path.join(TEST_DIR_ROOT, "parent", "child", "grandchild")
    assert not os.path.exists(test_path)
    ensure_dir_exists(test_path)
    assert os.path.exists(test_path)
    assert os.path.isdir(test_path)
    mock_logger.info.assert_called_with(f"Created directory: {test_path}")

@patch('common_utils.logger')
def test_ensure_dir_exists_empty_path(mock_logger):
    # ensure_dir_exists with "" or None should not attempt to create.
    # The function has a guard: `if dir_path and not os.path.exists(dir_path):`
    ensure_dir_exists("")
    ensure_dir_exists(None)
    mock_logger.info.assert_not_called()
    mock_logger.error.assert_not_called()
    # Just ensure no errors are raised.

@patch('common_utils.logger')
def test_ensure_dir_exists_path_is_file_raises_error(mock_logger):
    # os.makedirs will raise FileExistsError if the path is a file.
    # ensure_dir_exists should propagate this.
    file_path = os.path.join(TEST_DIR_ROOT, "a_file.txt")
    os.makedirs(TEST_DIR_ROOT, exist_ok=True) # Ensure root exists
    with open(file_path, "w") as f:
        f.write("I am a file.")
    
    with pytest.raises(FileExistsError): # More specific error
        ensure_dir_exists(file_path)
    mock_logger.error.assert_called_once() # Check that an error was logged

# --- Tests for Argument Parsers ---

def test_create_default_arg_parser_basic():
    parser = create_default_arg_parser("Test Description")
    assert parser is not None
    assert parser.description == "Test Description"

def test_create_default_arg_parser_required_args():
    parser = create_default_arg_parser("Test")
    # Test missing required arguments
    with pytest.raises(SystemExit): # argparse exits on error
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--input_file", "in.jsonl"]) # Missing output_file
    with pytest.raises(SystemExit):
        parser.parse_args(["--output_file", "out.jsonl"]) # Missing input_file

    # Test with required arguments
    args = parser.parse_args(["--input_file", "in.jsonl", "--output_file", "out.jsonl"])
    assert args.input_file == "in.jsonl"
    assert args.output_file == "out.jsonl"

def test_create_default_arg_parser_defaults_and_overrides():
    parser = create_default_arg_parser("Test")
    
    # Test defaults
    args_default = parser.parse_args(["--input_file", "in.jsonl", "--output_file", "out.jsonl"])
    assert args_default.limit is None
    assert args_default.log_level == "INFO"
    assert args_default.log_file_name is None

    # Test overrides
    args_override = parser.parse_args([
        "--input_file", "in.jsonl",
        "--output_file", "out.jsonl",
        "--limit", "100",
        "--log_level", "DEBUG",
        "--log_file_name", "test.log"
    ])
    assert args_override.limit == 100
    assert args_override.log_level == "DEBUG"
    assert args_override.log_file_name == "test.log"

def test_create_llm_arg_parser_inherits_default_and_adds_llm_args():
    parser = create_llm_arg_parser("LLM Test Description")
    assert parser.description == "LLM Test Description"

    # Check for a default arg
    with pytest.raises(SystemExit):
        parser.parse_args(["--output_file", "out.jsonl"]) # Missing input_file

    # Test LLM defaults
    args_llm_default = parser.parse_args(["--input_file", "in.jsonl", "--output_file", "out.jsonl"])
    # create_llm_arg_parser sets defaults to None; resolution happens via load_config_value in scripts
    assert args_llm_default.ollama_model is None 
    assert args_llm_default.ollama_host is None
    assert args_llm_default.chunk_size == 4000
    assert args_llm_default.chunk_overlap == 200
    assert args_llm_default.max_workers == 4
    # Check inherited default
    assert args_llm_default.log_level == "INFO"


    # Test LLM overrides
    args_llm_override = parser.parse_args([
        "--input_file", "in.jsonl",
        "--output_file", "out.jsonl",
        "--ollama_model", "test_model",
        "--ollama_host", "http://testhost:1234",
        "--chunk_size", "100",
        "--chunk_overlap", "10",
        "--max_workers", "8",
        "--log_level", "WARNING" # Overriding an inherited default
    ])
    assert args_llm_override.ollama_model == "test_model"
    assert args_llm_override.ollama_host == "http://testhost:1234"
    assert args_llm_override.chunk_size == 100
    assert args_llm_override.chunk_overlap == 10
    assert args_llm_override.max_workers == 8
    assert args_llm_override.log_level == "WARNING"

# --- Tests for Dataset I/O ---

SAMPLE_DATA = [
    {"id": 1, "text": "This is the first example."},
    {"id": 2, "text": "This is the second example."},
    {"id": 3, "text": "This is the third example."},
]

@patch('common_utils.logger')
def test_load_jsonl_dataset_valid_file(mock_logger):
    os.makedirs(TEST_DIR_ROOT, exist_ok=True)
    file_path = os.path.join(TEST_DIR_ROOT, "sample.jsonl")
    with open(file_path, "w") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item) + "\n")
    
    dataset = load_jsonl_dataset(file_path)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(SAMPLE_DATA)
    for i, row in enumerate(dataset):
        assert row["id"] == SAMPLE_DATA[i]["id"]
        assert row["text"] == SAMPLE_DATA[i]["text"]
    # Updated log message check
    expected_log_msg_part = f"Returning all {len(SAMPLE_DATA)} examples"
    found_log = False
    for call in mock_logger.info.call_args_list:
        if expected_log_msg_part in call[0][0] and f"from {file_path}" in call[0][0] and "loaded via Dataset.from_json" in call[0][0]:
            found_log = True
            break
    assert found_log, f"Expected log containing '{expected_log_msg_part}' and 'loaded via Dataset.from_json from {file_path}' not found."

@patch('common_utils.logger')
def test_load_jsonl_dataset_with_limit(mock_logger):
    os.makedirs(TEST_DIR_ROOT, exist_ok=True)
    file_path = os.path.join(TEST_DIR_ROOT, "sample_limit.jsonl")
    with open(file_path, "w") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item) + "\n")
    
    limit = 2
    dataset = load_jsonl_dataset(file_path, limit=limit)
    assert len(dataset) == limit
    assert dataset[0]["id"] == SAMPLE_DATA[0]["id"]
    # Updated log message check
    expected_log_msg_part = f"Returning {limit} examples (limited to {limit} from {len(SAMPLE_DATA)} total, loaded via Dataset.from_json)"
    found_log = False
    for call in mock_logger.info.call_args_list:
        if expected_log_msg_part in call[0][0] and f"from {file_path}" in call[0][0]:
            found_log = True
            break
    assert found_log, f"Expected log containing '{expected_log_msg_part} from {file_path}' not found."

@patch('common_utils.logger')
def test_load_jsonl_dataset_file_not_found(mock_logger):
    file_path = os.path.join(TEST_DIR_ROOT, "non_existent.jsonl")
    with pytest.raises(FileNotFoundError):
        load_jsonl_dataset(file_path)
    mock_logger.error.assert_called_with(f"Input file not found: {file_path} (checked with os.path.exists before any loading attempt)")

@patch('common_utils.logger')
def test_load_jsonl_dataset_empty_file(mock_logger):
    os.makedirs(TEST_DIR_ROOT, exist_ok=True)
    file_path = os.path.join(TEST_DIR_ROOT, "empty.jsonl")
    with open(file_path, "w") as f:
        pass # Create an empty file
    
    # The function should now return an empty dataset and log a warning
    dataset = load_jsonl_dataset(file_path)
    
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 0
    mock_logger.warning.assert_called_with(f"Input file {file_path} is empty (checked with os.path.getsize). Returning an empty dataset.")
    mock_logger.error.assert_not_called() # Ensure no error is logged for this case


@patch('common_utils.logger')
def test_load_jsonl_dataset_malformed_json(mock_logger):
    os.makedirs(TEST_DIR_ROOT, exist_ok=True)
    file_path = os.path.join(TEST_DIR_ROOT, "malformed.jsonl")
    with open(file_path, "w") as f:
        f.write("this is not json\n")
        f.write(json.dumps(SAMPLE_DATA[0]) + "\n") 
    
    # Behavior of datasets.Dataset.from_json with malformed lines can vary.
    # It might skip malformed lines or raise an error.
    # Let's assume it raises an error or results in partial load.
    # For this test, we'll check if an error is logged.
    # The function itself re-raises, so we expect an exception.
    with pytest.raises(Exception): # Could be json.JSONDecodeError or other Arrow/datasets error
        load_jsonl_dataset(file_path)
    mock_logger.error.assert_called()


@patch('common_utils.logger')
def test_save_jsonl_dataset_basic(mock_logger):
    file_path = os.path.join(TEST_DIR_ROOT, "output.jsonl")
    dataset_to_save = Dataset.from_list(SAMPLE_DATA)
    
    save_jsonl_dataset(dataset_to_save, file_path)
    
    assert os.path.exists(file_path)
    mock_logger.info.assert_any_call(f"Successfully saved {len(dataset_to_save)} examples to {file_path}.")

    # Verify content
    loaded_data = []
    with open(file_path, "r") as f:
        for line in f:
            loaded_data.append(json.loads(line))
    
    assert len(loaded_data) == len(SAMPLE_DATA)
    for i, item in enumerate(loaded_data):
        assert item["id"] == SAMPLE_DATA[i]["id"]
        assert item["text"] == SAMPLE_DATA[i]["text"]

@patch('common_utils.logger')
def test_save_jsonl_dataset_creates_output_directory(mock_logger):
    dir_path = os.path.join(TEST_DIR_ROOT, "new_output_dir")
    file_path = os.path.join(dir_path, "output_in_new_dir.jsonl")
    assert not os.path.exists(dir_path)

    dataset_to_save = Dataset.from_list(SAMPLE_DATA)
    save_jsonl_dataset(dataset_to_save, file_path)

    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)
    assert os.path.exists(file_path)
    mock_logger.info.assert_any_call(f"Created output directory: {dir_path}")
    mock_logger.info.assert_any_call(f"Successfully saved {len(dataset_to_save)} examples to {file_path}.")

@patch('common_utils.os.makedirs') # Mock os.makedirs to simulate failure
@patch('common_utils.logger')
def test_save_jsonl_dataset_output_dir_creation_failure(mock_logger, mock_makedirs):
    mock_makedirs.side_effect = OSError("Simulated permission denied")
    
    dir_path = os.path.join(TEST_DIR_ROOT, "uncreatable_output_dir")
    file_path = os.path.join(dir_path, "output.jsonl")
    dataset_to_save = Dataset.from_list(SAMPLE_DATA)

    with pytest.raises(OSError):
        save_jsonl_dataset(dataset_to_save, file_path)
    
    mock_logger.error.assert_called_with(
        f"Error creating output directory {dir_path}: Simulated permission denied",
        exc_info=True
    )

# --- Tests for setup_logging ---

# Fixture to reset logger handlers before and after each logging test
@pytest.fixture
def clean_logger():
    # Get the root logger instance, as setup_logging configures the root logger
    logger_instance = logging.getLogger() # Get root logger
    original_handlers = logger_instance.handlers[:]
    original_level = logger_instance.level
    # Ensure logger is clean before test
    for handler in original_handlers:
        logger_instance.removeHandler(handler)
    logger_instance.setLevel(logging.NOTSET) # Reset level

    yield logger_instance # Test runs

    # Restore original handlers and level after test
    current_handlers = logger_instance.handlers[:]
    for handler in current_handlers: # Remove handlers added during test
        logger_instance.removeHandler(handler)
    for handler in original_handlers: # Add back original handlers
        logger_instance.addHandler(handler)
    logger_instance.setLevel(original_level)


def test_setup_logging_basic_console(clean_logger):
    setup_logging(level=logging.DEBUG)
    assert clean_logger.level == logging.DEBUG
    
    console_handler = None
    for h in clean_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
            console_handler = h
            break
    assert console_handler is not None, "Console handler to stdout not found"
    assert console_handler.level == logging.DEBUG
    
    assert isinstance(console_handler.formatter, logging.Formatter)
    assert console_handler.formatter._fmt == common_utils.DEFAULT_LOG_FORMAT

@patch('common_utils.os.makedirs', return_value=None)
@patch('common_utils.os.path.exists', return_value=False)
@patch('common_utils.logging.FileHandler')
def test_setup_logging_with_file(mock_file_handler_cls, mock_path_exists, mock_os_makedirs, clean_logger):
    log_file = os.path.join(TEST_DIR_ROOT, "test_log_file.log")
    log_dir = os.path.dirname(log_file)

    # Get the mock instance that FileHandler() will return
    file_handler_instance = mock_file_handler_cls.return_value
    
    # Configure the mock's setLevel to also set the .level attribute on this instance
    def mock_set_level_side_effect(level_val):
        file_handler_instance.level = level_val
    file_handler_instance.setLevel.side_effect = mock_set_level_side_effect

    # Now call the function under test. This will use the configured mock.
    setup_logging(level=logging.INFO, log_file_name=log_file)

    # Assertions
    if log_dir: 
        mock_path_exists.assert_called_with(log_dir)
        mock_os_makedirs.assert_called_with(log_dir, exist_ok=True)
    
    mock_file_handler_cls.assert_called_once_with(log_file, mode='a')
    
    file_handler_instance.setLevel.assert_called_with(logging.INFO)
    assert file_handler_instance.level == logging.INFO, "Mocked FileHandler's level attribute was not set correctly by setLevel side_effect"
    assert isinstance(file_handler_instance.setFormatter.call_args[0][0], logging.Formatter)
    
    assert file_handler_instance in clean_logger.handlers

@patch('common_utils.os.makedirs', side_effect=OSError("Cannot create dir"))
@patch('common_utils.os.path.exists', return_value=False)
@patch('common_utils.logging.FileHandler') 
@patch('builtins.print') 
def test_setup_logging_file_dir_creation_error(mock_print, mock_file_handler_cls, mock_path_exists, mock_os_makedirs, clean_logger):
    log_file = os.path.join(TEST_DIR_ROOT, "dir_error_log.log")
    log_dir = os.path.dirname(log_file)

    setup_logging(level=logging.INFO, log_file_name=log_file)
    
    if log_dir:
        mock_os_makedirs.assert_called_with(log_dir, exist_ok=True)
    mock_print.assert_any_call(f"Error creating log directory {log_dir}: Cannot create dir", file=sys.stderr)
    
    mock_file_handler_cls.assert_not_called()

    # Check that only one console handler remains and no file handler was added
    num_console_handlers = 0
    num_file_handlers = 0
    has_stdout_handler = False
    for h in clean_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
            num_console_handlers += 1
            has_stdout_handler = True
        # When common_utils.logging.FileHandler is mocked by mock_file_handler_cls,
        # h will be an instance returned by mock_file_handler_cls() if it's a file handler.
        # This instance is mock_file_handler_cls.return_value.
        elif h == mock_file_handler_cls.return_value: # Check if h is the mocked FileHandler instance
            num_file_handlers += 1
    assert num_console_handlers >= 1, "Expected at least one console handler to stdout"
    assert has_stdout_handler, "Console handler to stdout not found" # More specific
    assert num_file_handlers == 0, "Expected no file handlers"
    # Note: We assert num_console_handlers >= 1 because caplog might also add StreamHandlers
    # The key is that OUR StreamHandler to sys.stdout is there, and NO FileHandler.
    # A more precise check for *our* handler would involve checking its formatter or level if unique.
    # For now, ensuring one is to stdout and no file handlers is the main goal.
    # The original assertion was len == 1. If caplog adds 2, and we add 1, total 3.
    # If we only care about OUR console handler and NO file handler:
    # Find our specific console handler.
    our_console_handler_found = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout and h.formatter._fmt == common_utils.DEFAULT_LOG_FORMAT
        for h in clean_logger.handlers
    )
    assert our_console_handler_found, "The specific console handler added by setup_logging was not found."


@patch('common_utils.os.makedirs', return_value=None) # Assume dir creation is fine or dir exists
@patch('common_utils.os.path.exists', return_value=True) 
@patch('common_utils.logging.FileHandler', side_effect=IOError("Cannot open file"))
@patch('builtins.print')
def test_setup_logging_file_open_error(mock_print, mock_file_handler_cls, mock_path_exists, mock_os_makedirs, clean_logger):
    log_file = os.path.join(TEST_DIR_ROOT, "file_open_error.log")
    
    setup_logging(level=logging.INFO, log_file_name=log_file)
    
    mock_file_handler_cls.assert_called_once_with(log_file, mode='a')
    mock_print.assert_any_call(f"Error setting up file handler for {log_file}: Cannot open file", file=sys.stderr)

    num_console_handlers = 0
    num_file_handlers = 0
    has_stdout_handler = False
    for h in clean_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
            num_console_handlers += 1
            has_stdout_handler = True
        # When common_utils.logging.FileHandler is mocked by mock_file_handler_cls,
        # h will be an instance returned by mock_file_handler_cls() if it's a file handler.
        # This instance is mock_file_handler_cls.return_value.
        elif h == mock_file_handler_cls.return_value: # Check if h is the mocked FileHandler instance
            num_file_handlers += 1
    assert num_console_handlers >= 1, "Expected at least one console handler to stdout"
    assert has_stdout_handler, "Console handler to stdout not found"
    assert num_file_handlers == 0, "Expected no file handlers"
    our_console_handler_found = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout and h.formatter._fmt == common_utils.DEFAULT_LOG_FORMAT
        for h in clean_logger.handlers
    )
    assert our_console_handler_found, "The specific console handler added by setup_logging was not found."

def test_setup_logging_custom_format(clean_logger):
    custom_format = "%(levelname)s: %(message)s"
    setup_logging(level=logging.DEBUG, log_format=custom_format)
    
    console_handler = None
    for h in clean_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
            console_handler = h
            break
    assert console_handler is not None, "Console handler to stdout not found"
    assert console_handler.formatter._fmt == custom_format

def test_setup_logging_no_file_provided(clean_logger):
    setup_logging(level=logging.INFO, log_file_name=None)
    assert clean_logger.level == logging.INFO
    
    num_console_handlers = 0
    num_file_handlers = 0
    has_stdout_handler = False
    for h in clean_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout:
            num_console_handlers += 1
            has_stdout_handler = True
        elif isinstance(h, logging.FileHandler):
            num_file_handlers += 1
            
    assert num_console_handlers >= 1, "Expected at least one console handler to stdout"
    assert has_stdout_handler, "Console handler to stdout not found"
    assert num_file_handlers == 0, "Expected no file handlers"
    # Ensure FileHandler was not attempted
    assert not any(isinstance(h, logging.FileHandler) for h in clean_logger.handlers)
    our_console_handler_found = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout and h.formatter._fmt == common_utils.DEFAULT_LOG_FORMAT
        for h in clean_logger.handlers
    )
    assert our_console_handler_found, "The specific console handler added by setup_logging was not found."

# --- Tests for Pydantic Models & Validation ---

VALID_MESSAGE_USER = {"role": "user", "content": "Hello"}
VALID_MESSAGE_ASSISTANT = {"role": "assistant", "content": "Hi there"}
INVALID_MESSAGE_ROLE = {"role": "customer", "content": "Query"}
VALID_TRACE_SIMPLE = {
    "trace_id": "trace_v1",
    "schema_version": "1.0",
    "messages": [VALID_MESSAGE_USER, VALID_MESSAGE_ASSISTANT],
    "completion": "Final answer."
}
INVALID_TRACE_BAD_MESSAGE = { # Invalid because of message role, not missing trace_id/version for this specific test
    "trace_id": "trace_inv_msg",
    "schema_version": "1.0",
    "messages": [VALID_MESSAGE_USER, INVALID_MESSAGE_ROLE],
    "completion": "Another answer."
}
# This test case is for when 'completion' is missing, which BasePipelineInput allows (Optional[str])
# To make it invalid for BasePipelineInput, we'd need to make trace_id or schema_version missing.
# Let's adjust this to test a missing mandatory field for BasePipelineInput.
INVALID_TRACE_MISSING_MANDATORY_FIELD = {
    "trace_id": "trace_missing_field",
    # "schema_version": "1.0", # Missing schema_version
    "messages": [VALID_MESSAGE_USER],
    "completion": "Completion present."
}


def test_message_model_valid():
    msg = Message(**VALID_MESSAGE_USER)
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_message_model_invalid_role():
    with pytest.raises(ValidationError):
        Message(**INVALID_MESSAGE_ROLE)

@patch('common_utils.logger')
def test_validate_dataset_all_valid_list(mock_logger):
    data_list = [VALID_TRACE_SIMPLE, VALID_TRACE_SIMPLE]
    valid_examples, invalid_examples = validate_dataset(data_list, BasePipelineInput, "TestStepListValid")
    
    assert len(valid_examples) == 2
    assert len(invalid_examples) == 0
    assert valid_examples == data_list # Ensure data is passed through
    mock_logger.info.assert_called_with("[TestStepListValid] Validation summary: All 2 examples are valid.")

@patch('common_utils.logger')
def test_validate_dataset_all_valid_hf_dataset(mock_logger):
    hf_dataset = Dataset.from_list([VALID_TRACE_SIMPLE, VALID_TRACE_SIMPLE])
    valid_examples, invalid_examples = validate_dataset(hf_dataset, BasePipelineInput, "TestStepHFValid")

    assert len(valid_examples) == 2
    assert len(invalid_examples) == 0
    # Convert back to list of dicts for comparison if needed, or check specific fields
    assert valid_examples[0]["completion"] == VALID_TRACE_SIMPLE["completion"]
    mock_logger.info.assert_called_with("[TestStepHFValid] Validation summary: All 2 examples are valid.")

@patch('common_utils.logger')
def test_validate_dataset_mixed_validity_list(mock_logger):
    data_list = [VALID_TRACE_SIMPLE, INVALID_TRACE_BAD_MESSAGE, VALID_TRACE_SIMPLE, INVALID_TRACE_MISSING_MANDATORY_FIELD]
    valid_examples, invalid_examples = validate_dataset(data_list, BasePipelineInput, "TestStepListMixed")

    assert len(valid_examples) == 2 # VALID_TRACE_SIMPLE, VALID_TRACE_SIMPLE
    assert len(invalid_examples) == 2 # INVALID_TRACE_BAD_MESSAGE, INVALID_TRACE_MISSING_MANDATORY_FIELD
    assert valid_examples[0]["completion"] == VALID_TRACE_SIMPLE["completion"]
    assert invalid_examples[0]["example_index"] == 1 # INVALID_TRACE_BAD_MESSAGE
    assert "errors" in invalid_examples[0]
    assert invalid_examples[1]["example_index"] == 3 # INVALID_TRACE_MISSING_MANDATORY_FIELD

    # Check warning logs more robustly
    found_error_bad_message = False
    found_error_missing_field = False
    for call_args in mock_logger.warning.call_args_list:
        log_message = call_args[0][0] 
        if "Validation failed for example 1" in log_message: # INVALID_TRACE_BAD_MESSAGE
            assert "'type': 'value_error'" in log_message 
            assert "'loc': ('messages', 1, 'role')" in log_message 
            assert "got 'customer'" in log_message
            found_error_bad_message = True
        elif "Validation failed for example 3" in log_message: # INVALID_TRACE_MISSING_MANDATORY_FIELD
            assert "'type': 'missing'" in log_message
            assert "'loc': ('schema_version',)" in log_message # Check for missing schema_version
            found_error_missing_field = True
        elif "[TestStepListMixed] Validation summary: 2 valid, 2 invalid examples." in log_message:
            pass # Summary log
            
    assert found_error_bad_message, "Expected log for validation failure on INVALID_TRACE_BAD_MESSAGE not found or incorrect."
    assert found_error_missing_field, "Expected log for validation failure on INVALID_TRACE_MISSING_MANDATORY_FIELD not found or incorrect."
    mock_logger.warning.assert_any_call("[TestStepListMixed] Validation summary: 2 valid, 2 invalid examples.")


@patch('common_utils.logger')
def test_validate_dataset_invalid_input_type(mock_logger):
    invalid_input = "this is not a dataset"
    valid_examples, invalid_examples = validate_dataset(invalid_input, BasePipelineInput, "TestStepInvalidType")

    assert len(valid_examples) == 0
    assert len(invalid_examples) == 1
    assert invalid_examples[0]["errors"] == "Input must be a Dataset or list of dicts"
    mock_logger.error.assert_called_with("[TestStepInvalidType] Validation error: Input must be a Hugging Face Dataset or a list of dictionaries.")

@patch('common_utils.BaseModel.model_validate') # Mock the actual validation call
@patch('common_utils.logger')
def test_validate_dataset_unexpected_error_during_validation(mock_logger, mock_model_validate):
    mock_model_validate.side_effect = RuntimeError("Unexpected boom!")
    # Ensure VALID_TRACE_SIMPLE is used here, which now includes trace_id and schema_version
    data_list = [VALID_TRACE_SIMPLE] 
    
    valid_examples, invalid_examples = validate_dataset(data_list, BasePipelineInput, "TestStepUnexpected")

    assert len(valid_examples) == 0
    assert len(invalid_examples) == 1
    assert invalid_examples[0]["example_index"] == 0
    assert invalid_examples[0]["errors"][0]["type"] == "unexpected_error"
    assert invalid_examples[0]["errors"][0]["msg"] == "Unexpected boom!"
    mock_logger.error.assert_called_with("[TestStepUnexpected] Unexpected error validating example 0: Unexpected boom!", exc_info=True)
    mock_logger.warning.assert_called_with("[TestStepUnexpected] Validation summary: 0 valid, 1 invalid examples.")
