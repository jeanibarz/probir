import pytest
import subprocess
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import ollama # Added for MagicMock spec

# Define paths for test data
TEST_DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = TEST_DATA_DIR / "sample_input_for_llm_anonymization.jsonl"
ACTUAL_OUTPUT_DIR = TEST_DATA_DIR / "actual_outputs"
ACTUAL_OUTPUT_FILE = ACTUAL_OUTPUT_DIR / "actual_output_llm_anonymized.jsonl"

# Ensure actual_outputs directory exists
os.makedirs(ACTUAL_OUTPUT_DIR, exist_ok=True)

# Sample data for input file
SAMPLE_INPUT_DATA_LLM = [
    {
        "messages": [{"role": "user", "content": "My name is John Doe and I live at 123 Main St."}],
        "completion": "My phone is 555-0100. Email: john.doe@example.com",
        "trace_id": "llm_anon_trace1",
        "session_id": "session_llm_001",
        "turn_in_session_id": 1,
        "original_messages": None, # Assume this is output from a previous step or not present
        "original_completion": None,
        "anonymization_details": {"regex_patterns_found": []} # Assume regex found nothing
    },
    {
        "messages": [{"role": "user", "content": "This text has no PII according to the LLM."}],
        "completion": "Indeed, it does not.",
        "trace_id": "llm_anon_trace2",
        "session_id": "session_llm_001",
        "turn_in_session_id": 2,
        "original_messages": None,
        "original_completion": None,
        "anonymization_details": {"regex_patterns_found": []}
    }
]

# Expected logic after LLM anonymization (mocked LLM responses)
# We will mock the LLM to return specific PII detections.
EXPECTED_LLM_ANONYMIZATION_LOGIC = [
    {
        "trace_id": "llm_anon_trace1",
        "anonymized_user_messages": [
            {"role": "user", "content": "My name is [PERSON_NAME_REDACTED] and I live at [ADDRESS_REDACTED]."}
        ],
        "anonymized_completion": "My phone is [PHONE_NUMBER_REDACTED]. Email: [EMAIL_ADDRESS_REDACTED]",
        "has_llm_anonymization_details": True,
        "llm_pii_types_found": ["PERSON_NAME", "ADDRESS", "PHONE_NUMBER", "EMAIL_ADDRESS"]
    },
    {
        "trace_id": "llm_anon_trace2",
        "anonymized_user_messages": [{"role": "user", "content": "This text has no PII according to the LLM."}],
        "anonymized_completion": "Indeed, it does not.",
        "has_llm_anonymization_details": False, # Or True with empty details
        "llm_pii_types_found": []
    }
]

@pytest.fixture(scope="module", autouse=True)
def create_test_data_files_llm_anonymize():
    with open(INPUT_FILE, "w") as f:
        for item in SAMPLE_INPUT_DATA_LLM:
            f.write(json.dumps(item) + "\n")
    
    if os.path.exists(ACTUAL_OUTPUT_FILE):
        os.remove(ACTUAL_OUTPUT_FILE)
    yield

# This function will be the mock target for ollama.Client.chat
# It needs to simulate the LLM's response structure.
def mock_ollama_chat_response(*args, **kwargs):
    # The 'messages' kwarg to ollama.chat contains the system prompt and user content.
    # We need to inspect the user content to decide which mock PII to return.
    # args[0] is usually the 'self' of the ollama.Client instance.
    # kwargs['messages'] is a list of dicts. The last one is typically the user message with the text to analyze.
    
    text_to_analyze = ""
    if 'messages' in kwargs and kwargs['messages']:
        # Assuming the text to analyze is in the last message's content
        text_to_analyze = kwargs['messages'][-1]['content']

    mock_response = {"message": {"content": ""}} # Default empty response

    if "John Doe" in text_to_analyze and "123 Main St" in text_to_analyze:
        # Corresponds to first part of trace1 messages
        pii_detections = [
            {"category": "PERSON_NAME", "value": "John Doe"},
            {"category": "ADDRESS", "value": "123 Main St"}
        ]
        mock_response["message"]["content"] = json.dumps({"pii_list": pii_detections})
    elif "555-0100" in text_to_analyze and "john.doe@example.com" in text_to_analyze:
        # Corresponds to trace1 completion
        pii_detections = [
            {"category": "PHONE_NUMBER", "value": "555-0100"},
            {"category": "EMAIL_ADDRESS", "value": "john.doe@example.com"}
        ]
        mock_response["message"]["content"] = json.dumps({"pii_list": pii_detections})
    else: # No PII for trace2 or other chunks
        mock_response["message"]["content"] = json.dumps({"pii_list": []})
        
    return mock_response


import sys # For sys.argv manipulation

# Import the main function from the script to be tested
from step1b_anonymize_llm import main as step1b_main # Assuming src is in pythonpath

def run_llm_anonymize_script_direct_call():
    # Original sys.argv
    original_argv = sys.argv
    
    # Simulate command-line arguments
    sys.argv = [
        "src/step1b_anonymize_llm.py", # Script name, as if called from command line
        "--input_file", str(INPUT_FILE),
        "--output_file", str(ACTUAL_OUTPUT_FILE),
        "--log_level", "DEBUG",
        "--ollama_model", "mock_model",
        "--max_workers", "1"
    ]

    mock_client_instance = MagicMock(spec=ollama.Client)
    mock_client_instance.chat.side_effect = mock_ollama_chat_response
    mock_client_instance.list.return_value = {
        "models": [
            {
                "name": "mock_model:latest",
                "modified_at": "2023-01-01T00:00:00Z",
                "size": 123456789,
                "digest": "abcdef123456",
                "details": {"family": "mockfamily", "parameter_size": "0B", "quantization_level": "Q0"}
            }
        ]
    }

    # The patch context manager will affect ollama.Client within the current process
    with patch('step1b_anonymize_llm.ollama.Client', return_value=mock_client_instance) as mock_ollama_client_constructor:
    # Note: The patch target string should be where 'ollama.Client' is looked up by the script under test.
    # If step1b_anonymize_llm.py does 'import ollama' and then 'ollama.Client()',
    # then 'step1b_anonymize_llm.ollama.Client' is correct.
    # If it does 'from ollama import Client' and then 'Client()',
    # then 'step1b_anonymize_llm.Client' would be the target.
    # Assuming 'import ollama' then 'ollama.Client()' based on typical usage.
        try:
            step1b_main() # Call the script's main function
        except SystemExit as e:
            # The script calls sys.exit(1) on error. We can catch this.
            # If e.code is 0 or None, it's a normal exit.
            if e.code != 0 and e.code is not None: 
                pytest.fail(f"LLM anonymize script exited with code {e.code}")
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
        # Verify ollama.Client was called
        mock_ollama_client_constructor.assert_called()
        # Verify chat was called
        assert mock_client_instance.chat.call_count > 0, "ollama.Client.chat was not called"


def test_llm_anonymize_data():
    run_llm_anonymize_script_direct_call() # Changed to direct call

    assert ACTUAL_OUTPUT_FILE.exists(), "Output file was not created by llm_anonymize script"

    actual_data = []
    with open(ACTUAL_OUTPUT_FILE, "r") as f:
        for line in f:
            actual_data.append(json.loads(line))
    
    assert len(actual_data) == len(SAMPLE_INPUT_DATA_LLM)

    for i, actual_item in enumerate(actual_data):
        original_input_item = SAMPLE_INPUT_DATA_LLM[i]
        expected_logic = next(e for e in EXPECTED_LLM_ANONYMIZATION_LOGIC if e["trace_id"] == actual_item["trace_id"])

        if expected_logic["has_llm_anonymization_details"]:
            assert actual_item.get("original_messages") is not None, f"original_messages missing for trace_id {actual_item['trace_id']}"
            assert actual_item["original_messages"] == original_input_item["messages"]
            
            assert actual_item.get("original_completion") is not None, f"original_completion missing for trace_id {actual_item['trace_id']}"
            assert actual_item["original_completion"] == original_input_item["completion"]
            
            assert "llm_anonymization_details" in actual_item
            details = actual_item["llm_anonymization_details"]
            assert isinstance(details, dict)
            assert "llm_detected_pii_items_messages" in details
            assert "llm_detected_pii_items_completion" in details
            
            # Check if the types of PII found match expectations (simplified check)
            all_found_types = set()
            for pii_item in details.get("llm_detected_pii_items_messages", []):
                all_found_types.add(pii_item["category"])
            for pii_item in details.get("llm_detected_pii_items_completion", []):
                all_found_types.add(pii_item["category"])
            
            assert sorted(list(all_found_types)) == sorted(expected_logic["llm_pii_types_found"])

        else:
            assert actual_item.get("original_messages") is None
            assert actual_item.get("original_completion") is None
            if "llm_anonymization_details" in actual_item and actual_item["llm_anonymization_details"] is not None:
                details = actual_item["llm_anonymization_details"]
                # Check that the lists of detected PII are empty or not present
                assert not details.get("llm_detected_pii_items_messages")
                assert not details.get("llm_detected_pii_items_completion")
                # Also check that the overall sensitive categories list is empty or not present if no PII
                assert not details.get("llm_sensitive_categories_found")


        # Check anonymized messages and completion based on expected placeholders
        # This requires knowing the exact placeholder format used by the LLM anonymization script
        # For this example, we assume placeholders like [PERSON_NAME], [ADDRESS] etc.
        assert len(actual_item["messages"]) == len(expected_logic["anonymized_user_messages"])
        for idx, msg_pair in enumerate(zip(actual_item["messages"], expected_logic["anonymized_user_messages"])):
            actual_msg, expected_msg = msg_pair
            assert actual_msg["role"] == expected_msg["role"]
            assert actual_msg["content"] == expected_msg["content"], \
                 f"Content mismatch in message {idx} for trace_id {actual_item['trace_id']}. Expected: '{expected_msg['content']}', Got: '{actual_msg['content']}'"

        assert actual_item["completion"] == expected_logic["anonymized_completion"], \
             f"Completion mismatch for trace_id {actual_item['trace_id']}. Expected: '{expected_logic['anonymized_completion']}', Got: '{actual_item['completion']}'"

        # Check other fields
        assert actual_item["trace_id"] == original_input_item["trace_id"]
        assert actual_item["session_id"] == original_input_item["session_id"]
        assert actual_item["turn_in_session_id"] == original_input_item["turn_in_session_id"]
