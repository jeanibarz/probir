import pytest
import os
import json
import logging
from unittest.mock import patch, MagicMock

from step3_analyze_correction_patterns import main as step3_main
from common_utils import BasePipelineInput # Changed from BaseTrace
from step3_analyze_correction_patterns import CorrectionPatterns # Import for mock responses

TEST_OUTPUT_DIR = "tests/test_outputs/step3_analyze_correction_patterns"
TEST_LOG_DIR = "logs"

@pytest.fixture(scope="function", autouse=True)
def setup_teardown_test_dir():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    # Clean up files from previous test runs in TEST_OUTPUT_DIR
    for f in os.listdir(TEST_OUTPUT_DIR):
        os.remove(os.path.join(TEST_OUTPUT_DIR, f))
    yield
    # Optional: clean up TEST_OUTPUT_DIR after test
    # for f in os.listdir(TEST_OUTPUT_DIR):
    #     os.remove(os.path.join(TEST_OUTPUT_DIR, f))

def create_sample_input_file(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def load_jsonl_output(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

# Sample input data (conforming to BasePipelineInput)
SAMPLE_INPUT_DATA_1 = [
    BasePipelineInput(
        trace_id="trace_1",
        schema_version="1.0",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        completion="2+2 equals 4."
    ).model_dump(exclude_none=True)
]

SAMPLE_INPUT_DATA_2 = [
    BasePipelineInput(
        trace_id="trace_2",
        schema_version="1.0",
        messages=[
            {"role": "user", "content": "Tell me about cats."},
            {"role": "assistant", "content": "Cats are felines."},
            {"role": "user", "content": "No, I meant dogs."}
        ],
        completion="Ah, you're asking about dogs! Dogs are canines..."
    ).model_dump(exclude_none=True)
]


# --- Mock Ollama Client and Responses ---
def mock_ollama_chat_responses(mock_responses_iter):
    """
    Factory for creating a side_effect function for mocked ollama.Client.chat.
    `mock_responses_iter` should be an iterator or list of dictionaries
    representing the `response['message']['content']` to be returned by the LLM,
    or Exception instances to be raised.
    """
    responses = iter(mock_responses_iter)
    def side_effect_func(*args, **kwargs):
        next_response_content = next(responses)
        if isinstance(next_response_content, Exception):
            raise next_response_content
        
        # Simulate the structure of ollama client's chat response
        return {
            "model": "mock_model",
            "created_at": "2023-10-26T18:21:28.959Z",
            "message": {
                "role": "assistant",
                "content": next_response_content # This is where the JSON string goes
            },
            "done": True,
            # ... other fields ollama might return
        }
    return side_effect_func


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_no_patterns(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_no_corr.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_no_corr.jsonl")
    log_file = "test_step3_no_corr.log"

    create_sample_input_file(input_file, SAMPLE_INPUT_DATA_1)

    # Mock Ollama client instance and its methods
    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}] # Simulate successful connection test

    # Define mock LLM response (as a JSON string, like the LLM would output)
    llm_response_no_correction = CorrectionPatterns(
        is_user_feedback_on_error=False,
        is_assistant_self_correction=False,
        is_assistant_error_before_correction=False,
        correction_rationale=None
    ).model_dump_json()

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([llm_response_no_correction])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--ollama_model", "mock_model",
        "--log_level", "INFO",
        "--log_file_name", log_file,
        "--max_workers", "1" # Easier to debug sequentially first
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert result["is_user_feedback_on_error"] is False
    assert result["is_assistant_self_correction"] is False
    assert result["is_assistant_error_before_correction"] is False
    assert result["correction_rationale"] is None
    assert "correction_analysis_details" in result
    assert result["correction_analysis_details"]["llm_correction_raw_output"] == llm_response_no_correction
    
    assert "Examples with user feedback on error: 0" in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_user_feedback(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_user_feedback.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_user_feedback.jsonl")
    log_file = "test_step3_user_feedback.log"

    create_sample_input_file(input_file, SAMPLE_INPUT_DATA_2)

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    llm_response_user_feedback = CorrectionPatterns(
        is_user_feedback_on_error=True,
        is_assistant_self_correction=False, # Can be true if assistant also corrects
        is_assistant_error_before_correction=True,
        correction_rationale="User indicated 'cats' was wrong, meant 'dogs'."
    ).model_dump_json()

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([llm_response_user_feedback])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--ollama_model", "mock_model",
        "--log_level", "INFO",
        "--log_file_name", log_file,
        "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert result["is_user_feedback_on_error"] is True
    assert result["is_assistant_error_before_correction"] is True
    assert result["correction_rationale"] == "User indicated 'cats' was wrong, meant 'dogs'."
    
    assert "Examples with user feedback on error: 1" in caplog.text
    assert "Examples with assistant error before correction: 1" in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_assistant_self_correction(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_self_corr.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_self_corr.jsonl")
    log_file = "test_step3_self_corr.log"

    # Sample data where assistant might self-correct
    sample_data_self_correction = [
        BasePipelineInput(
            trace_id="trace_self_correct",
            schema_version="1.0",
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Berlin."},
                {"role": "assistant", "content": "Apologies, I misspoke. The capital of France is Paris."}
            ],
            completion="The capital of France is indeed Paris." # Final completion might reiterate
        ).model_dump(exclude_none=True)
    ]
    create_sample_input_file(input_file, sample_data_self_correction)

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    llm_response_self_correction = CorrectionPatterns(
        is_user_feedback_on_error=False,
        is_assistant_self_correction=True,
        is_assistant_error_before_correction=True, # Assistant made an error (Berlin)
        correction_rationale="Assistant corrected its statement about capital of France from Berlin to Paris."
    ).model_dump_json()

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([llm_response_self_correction])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--ollama_model", "mock_model",
        "--log_level", "INFO",
        "--log_file_name", log_file,
        "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert result["is_assistant_self_correction"] is True
    assert result["is_assistant_error_before_correction"] is True
    assert result["correction_rationale"] == "Assistant corrected its statement about capital of France from Berlin to Paris."
    
    assert "Examples with assistant self-correction: 1" in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_llm_malformed_json(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.ERROR) # Expecting error logs
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_malformed.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_malformed.jsonl")
    log_file = "test_step3_malformed.log"

    create_sample_input_file(input_file, SAMPLE_INPUT_DATA_1) # Use simple data

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    malformed_json_response = "{'is_user_feedback_on_error': false, 'is_assistant_self_correction': false," # Missing closing brace and quotes

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([malformed_json_response])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    # Check that default/error values are populated
    assert result["is_user_feedback_on_error"] is False # Default from CorrectionPatterns()
    assert "JSONDecodeError" in caplog.text
    assert "llm_correction_raw_output" in result["correction_analysis_details"]
    assert result["correction_analysis_details"]["llm_correction_raw_output"] == malformed_json_response
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_llm_validation_error(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.ERROR)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_validation_err.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_validation_err.jsonl")
    log_file = "test_step3_validation_err.log"

    create_sample_input_file(input_file, SAMPLE_INPUT_DATA_1)

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    # Valid JSON, but does not match CorrectionPatterns schema (e.g. wrong type)
    invalid_schema_response = json.dumps({"is_user_feedback_on_error": "not_a_boolean"})

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([invalid_schema_response])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert result["is_user_feedback_on_error"] is False # Default
    assert "Pydantic ValidationError" in caplog.text
    assert result["correction_analysis_details"]["llm_correction_raw_output"] == invalid_schema_response
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_ollama_api_error(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.ERROR)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_api_err.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_api_err.jsonl")
    log_file = "test_step3_api_err.log"

    create_sample_input_file(input_file, SAMPLE_INPUT_DATA_1)

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    # Simulate an ollama.ResponseError
    from src.step3_analyze_correction_patterns import ollama # Ensure ollama is imported for the exception type
    api_error = ollama.ResponseError("Mocked API error", status_code=500)

    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([api_error])
    
    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 1
    
    result = output_data[0]
    assert result["is_user_feedback_on_error"] is False # Default
    assert "Ollama API ResponseError" in caplog.text
    assert "Mocked API error" in result["correction_analysis_details"]["llm_correction_raw_output"]
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_empty_input(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_empty.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_empty.jsonl")
    log_file = "test_step3_empty.log"

    create_sample_input_file(input_file, []) # Empty data

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]
    
    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file) # Script should create an empty output file
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == 0
    
    assert "Input dataset is empty. Nothing to process further." in caplog.text
    assert "Empty output file saved" in caplog.text
    mock_client_instance.chat.assert_not_called() # Ollama should not be called for empty input
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_limit_argument(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.INFO)
    input_file = os.path.join(TEST_OUTPUT_DIR, "input_limit.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_limit.jsonl")
    log_file = "test_step3_limit.log"

    # Create more data than the limit
    input_data = SAMPLE_INPUT_DATA_1 * 3 
    create_sample_input_file(input_file, input_data)

    mock_client_instance = MagicMock()
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]

    # Mock response for two calls (due to limit=2)
    llm_response = CorrectionPatterns().model_dump_json()
    mock_client_instance.chat.side_effect = mock_ollama_chat_responses([llm_response, llm_response])
    
    limit_value = 2
    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, 
        "--max_workers", "1", "--limit", str(limit_value)
    ]

    with patch("sys.argv", test_args):
        step3_main()

    assert os.path.exists(output_file)
    output_data = load_jsonl_output(output_file)
    assert len(output_data) == limit_value
    assert mock_client_instance.chat.call_count == limit_value
    # Updated log message check based on common_utils.load_jsonl_dataset changes
    expected_log_msg_part = f"Returning {limit_value} examples (limited from {len(input_data)}, loaded via Dataset.from_json (schema inference))"
    found_log = False
    for record in caplog.records: # Iterate through actual log records
        if record.levelname == "INFO" and expected_log_msg_part in record.message and f"from {input_file}" in record.message:
            found_log = True
            break
    assert found_log, f"Expected log containing '{expected_log_msg_part} from {input_file}' not found in caplog.text:\n{caplog.text}"
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))


@patch('src.step3_analyze_correction_patterns.ollama.Client')
def test_analyze_corrections_input_file_not_found(mock_ollama_client_cls, caplog):
    caplog.set_level(logging.ERROR)
    input_file = os.path.join(TEST_OUTPUT_DIR, "non_existent_input.jsonl")
    output_file = os.path.join(TEST_OUTPUT_DIR, "output_non_existent.jsonl")
    log_file = "test_step3_file_not_found.log"

    # No input file created

    mock_client_instance = MagicMock() # Ollama client still needs to be mocked
    mock_ollama_client_cls.return_value = mock_client_instance
    mock_client_instance.list.return_value = [{"name": "mock_model"}]


    test_args = [
        "src/step3_analyze_correction_patterns.py", "--input_file", input_file, "--output_file", output_file,
        "--ollama_model", "mock_model", "--log_level", "INFO", "--log_file_name", log_file, "--max_workers", "1"
    ]

    with patch("sys.argv", test_args):
        step3_main() # Script should log error and exit

    assert not os.path.exists(output_file) # No output file should be created
    assert f"Input file not found: {input_file}" in caplog.text
    assert os.path.exists(os.path.join(TEST_LOG_DIR, log_file))
