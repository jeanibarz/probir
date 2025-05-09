import pytest
import os
import json
import logging
from unittest.mock import patch, mock_open

from datasets import Dataset
from src.step2_score_complexity import main as step2_main
from src.common_utils import BasePipelineInput # For creating valid test data

# Define a root directory for test outputs to make cleanup easier
TEST_OUTPUT_DIR = "tests/test_outputs/step2_score_complexity"
TEST_LOG_DIR = "logs" # Assuming logs are configured to go here by setup_logging

@pytest.fixture(scope="function", autouse=True)
def setup_teardown_test_dir():
    """Create and clean up the test output directory for each test."""
    if os.path.exists(TEST_OUTPUT_DIR):
        for f in os.listdir(TEST_OUTPUT_DIR):
            os.remove(os.path.join(TEST_OUTPUT_DIR, f))
    else:
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Ensure log directory exists for file logging tests
    os.makedirs(TEST_LOG_DIR, exist_ok=True)

    yield

    # Teardown: Optional - could remove files or keep for inspection
    # for f in os.listdir(TEST_OUTPUT_DIR):
    #     os.remove(os.path.join(TEST_OUTPUT_DIR, f))
    # if os.path.exists(TEST_OUTPUT_DIR):
    #     os.rmdir(TEST_OUTPUT_DIR)


def create_sample_input_file(file_path, data):
    """Helper to create a JSONL input file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl_output(file_path):
    """Helper to load a JSONL output file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

# Sample data for testing
# Ensure trace_id and schema_version are present as they are added by run_pipeline.py
# before any step script processes the data.
SAMPLE_DATA_SIMPLE = [
    BasePipelineInput(
        trace_id="simple_trace_1",
        schema_version="1.0",
        messages=[{"role": "user", "content": "Hello"}],
        completion="Hi there!"
    ).model_dump(exclude_none=True)
]

SAMPLE_DATA_COMPLEX = [
    BasePipelineInput(
        trace_id="complex_trace_1",
        schema_version="1.0",
        messages=[
            {"role": "user", "content": "Can you explain the algorithm for quicksort?"},
            {"role": "assistant", "content": "Sure, quicksort is a divide and conquer algorithm..."},
            {"role": "user", "content": "And how would I implement it in Python code?"}
        ],
        completion="Here's a Python implementation of quicksort: ... (long code example) ... and it's important for performance."
    ).model_dump(exclude_none=True)
]

SAMPLE_DATA_EDGE_CASES = [
    # Valid empty example, should result in 0 score
    BasePipelineInput(trace_id="edge_trace_1", schema_version="1.0", messages=[], completion="").model_dump(exclude_none=True),
    # Raw dicts to test robustness of the script's internal parsing.
    # These should also include schema_version if they are to be considered valid inputs at this stage.
    {"trace_id": "edge_trace_2", "schema_version": "1.0", "messages": None, "completion": None},
    {"trace_id": "edge_trace_3", "schema_version": "1.0", "messages": [{"role":"user", "content": 123}], "completion": 456}
]


def test_score_complexity_simple_case(caplog):
    """Test with a simple input, expecting low complexity."""
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "simple_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "simple_output.jsonl")
    log_file = "test_step2_simple.log" # Will be created in TEST_LOG_DIR by setup_logging

    create_sample_input_file(input_file, SAMPLE_DATA_SIMPLE)

    test_args = [
        "src/step2_score_complexity.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--log_level", "INFO",
        "--log_file_name", log_file
    ]

    with patch("sys.argv", test_args):
        step2_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert "complexity_score" in result
    assert "complexity_reasoning" in result
    assert isinstance(result["complexity_score"], (int, float))
    
    # Expected scores for SAMPLE_DATA_SIMPLE:
    # Turns: 1 message -> 1 * 2 = 2
    # MsgLen: len("Hello") = 5 -> 5 // 100 = 0
    # CompLen: len("Hi there!") = 9 -> 9 // 75 = 0
    # Keywords: 0
    # Total: 2 + 0 + 0 + 0 = 2
    assert result["complexity_score"] == 2.0
    assert "Turns: 2" in result["complexity_reasoning"]
    assert "MsgLen: 0" in result["complexity_reasoning"]
    assert "CompLen: 0" in result["complexity_reasoning"]
    assert "Keywords: 0" in result["complexity_reasoning"]

    # Check logs
    assert f"Input file: {input_file}, Output file: {output_file}" in caplog.text
    assert "Complexity scoring complete." in caplog.text
    assert "Saving scored dataset" in caplog.text
    assert "Processing complete." in caplog.text
    assert "Average: 2.00" in caplog.text # Summary stats
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


def test_score_complexity_complex_case(caplog):
    """Test with a more complex input, expecting higher complexity."""
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "complex_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "complex_output.jsonl")
    log_file = "test_step2_complex.log"

    create_sample_input_file(input_file, SAMPLE_DATA_COMPLEX)

    test_args = [
        "src/step2_score_complexity.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--log_level", "INFO",
        "--log_file_name", log_file
    ]

    with patch("sys.argv", test_args):
        step2_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert "complexity_score" in result
    assert "complexity_reasoning" in result

    # Expected scores for SAMPLE_DATA_COMPLEX:
    # Messages:
    #   "Can you explain the algorithm for quicksort?" (41 chars, "explain", "algorithm")
    #   "Sure, quicksort is a divide and conquer algorithm..." (50 chars, "quicksort", "divide", "conquer", "algorithm")
    #   "And how would I implement it in Python code?" (44 chars, "implement", "code")
    # Completion: "Here's a Python implementation of quicksort: ... (long code example) ... and it's important for performance." (100 chars, "Python", "implementation", "quicksort", "code", "performance")
    
    # Turns: 3 messages -> 3 * 2 = 6 (score_num_turns)
    # MsgLen: 41 + 50 + 44 = 135 -> 135 // 100 = 1 (score_message_length)
    # CompLen: 100 -> 100 // 75 = 1 (score_completion_length)
    # Keywords:
    #   explain (1), algorithm (2), quicksort (2), implement (1), code (2), performance (1) -> Total 9 keywords
    #   7 keywords * 2 = 14 (score_keywords)
    # Total: 6 + 1 + 1 + 14 = 22
    assert result["complexity_score"] == 22.0
    assert "Turns: 6" in result["complexity_reasoning"]
    assert "MsgLen: 1" in result["complexity_reasoning"]
    assert "CompLen: 1" in result["complexity_reasoning"]
    assert "Keywords: 14" in result["complexity_reasoning"]
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


def test_score_complexity_edge_cases(caplog):
    """Test with edge case inputs."""
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "edge_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "edge_output.jsonl")
    log_file = "test_step2_edge.log"

    create_sample_input_file(input_file, SAMPLE_DATA_EDGE_CASES)

    test_args = [
        "src/step2_score_complexity.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--log_level", "INFO",
        "--log_file_name", log_file
    ]

    with patch("sys.argv", test_args):
        step2_main()

    # The script should fail to load this dataset due to type inconsistency (string vs int for completion)
    # as per pyarrow's strictness in Dataset.from_json.
    assert not os.path.exists(output_file) 
    
    # Check for the specific error logs
    assert f"Error loading dataset from {input_file}" in caplog.text
    assert "An error occurred while generating the dataset" in caplog.text # General error from common_utils
    # More specific error from pyarrow via datasets
    assert "JSON parse error: Column(/completion) changed from string to number in row 2" in caplog.text
    
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))

def test_score_complexity_limit_argument(caplog):
    """Test the --limit argument."""
    caplog.set_level(logging.INFO)
    input_data = SAMPLE_DATA_SIMPLE * 5 # Create 5 identical simple examples
    input_file = os.path.join(TEST_OUTPUT_DIR, "limit_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "limit_output.jsonl")
    log_file = "test_step2_limit.log"

    create_sample_input_file(input_file, input_data)

    limit_value = 2
    test_args = [
        "src/step2_score_complexity.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--limit", str(limit_value),
        "--log_level", "INFO",
        "--log_file_name", log_file
    ]

    with patch("sys.argv", test_args):
        step2_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == limit_value # Check if limit was applied

    assert f"Processing limit: {limit_value} examples." in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))

def test_score_complexity_input_file_not_found(caplog):
    """Test behavior when input file does not exist."""
    caplog.set_level(logging.ERROR) # Expecting an error log
    input_file = os.path.join(TEST_OUTPUT_DIR, "non_existent_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "non_existent_output.jsonl")
    log_file = "test_step2_nonexist.log"

    test_args = [
        "src/step2_score_complexity.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--log_level", "INFO", # Script will log error, test captures it
        "--log_file_name", log_file
    ]

    with patch("sys.argv", test_args):
        step2_main() # Should not raise, but log an error and exit gracefully

    assert not os.path.exists(output_file) # Output file should not be created
    assert f"Error loading dataset from {input_file}" in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file)) # Log file should still be created

# TODO: Add test for invalid input JSONL format if not covered by common_utils tests
# TODO: Add test for permissions error on output file/directory (might be harder to mock reliably)
