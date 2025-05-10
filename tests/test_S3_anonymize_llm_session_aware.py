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

# Simplified Sample data for input file - only llm_anon_trace1
SAMPLE_INPUT_DATA_LLM = [
    {
        "schema_version": "1.0",
        "messages": [{"role": "user", "content": "My name is John Doe and I live at 123 Main St."}],
        "completion": "My phone is 555-0100. Email: john.doe@example.com",
        "trace_id": "llm_anon_trace1",
        "session_id": "session_llm_001", # Single session, single turn
        "turn_in_session_id": 1,
        "original_messages": None,
        "original_completion": None,
        "anonymization_details": {"regex_patterns_found": []}
    }
]

# Simplified Expected logic for llm_anon_trace1 only
EXPECTED_LLM_ANONYMIZATION_LOGIC = [
    {
        "trace_id": "llm_anon_trace1",
        # Messages are from the latest (only) turn. LLM finds PERSON_NAME, ADDRESS.
        "anonymized_user_messages": [
            {"role": "user", "content": "My name is [PERSON_NAME_REDACTED] and I live at [ADDRESS_REDACTED]."}
        ],
        # Completion is anonymized individually. LLM finds PHONE_NUMBER, EMAIL_ADDRESS.
        "anonymized_completion": "My phone is [PHONE_NUMBER_REDACTED]. Email: [EMAIL_ADDRESS_REDACTED]",
        "has_llm_anonymization_details": True,
        # PII types found are from this turn's messages AND completion.
        "llm_pii_types_found": sorted(["PERSON_NAME", "ADDRESS", "PHONE_NUMBER", "EMAIL_ADDRESS"])
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
    # For the simplified step1c:
    # 1. LLM is called to analyze all messages within the single input trace.
    #    For llm_anon_trace1 messages: "My name is John Doe and I live at 123 Main St."
    #    -> LLM should return PERSON_NAME, ADDRESS.
    # 2. LLM is called to analyze the completion of that single input trace.
    #    For llm_anon_trace1 completion: "My phone is 555-0100. Email: john.doe@example.com"
    #    -> LLM should return PHONE_NUMBER, EMAIL_ADDRESS.

    text_to_analyze = ""
    if 'messages' in kwargs and kwargs['messages']:
        text_to_analyze = kwargs['messages'][-1]['content']

    mock_response = {"message": {"content": ""}}
    
    # Define the prompt template prefix used by the script
    prompt_prefix = "Please analyze the following text for PII:\n\n"

    # Define the raw chunk texts for the single turn (llm_anon_trace1)
    raw_chunk_trace1_messages = "My name is John Doe and I live at 123 Main St."
    raw_chunk_trace1_completion = "My phone is 555-0100. Email: john.doe@example.com"

    if text_to_analyze == f"{prompt_prefix}{raw_chunk_trace1_messages}": # Analysis of messages for the only turn
        pii_detections = [
            {"category": "PERSON_NAME", "value": "John Doe"},
            {"category": "ADDRESS", "value": "123 Main St"}
        ]
        mock_response["message"]["content"] = json.dumps({"pii_list": pii_detections})
    elif text_to_analyze == f"{prompt_prefix}{raw_chunk_trace1_completion}": # Analysis of completion for the only turn
        pii_detections = [
            {"category": "PHONE_NUMBER", "value": "555-0100"},
            {"category": "EMAIL_ADDRESS", "value": "john.doe@example.com"}
        ]
        mock_response["message"]["content"] = json.dumps({"pii_list": pii_detections})
    else:
        # If this branch is hit, it means the text_to_analyze didn't match any specific condition.
        # This would be unexpected for the current test data.
        pytest.fail(f"mock_ollama_chat_response received unexpected text for analysis: '{text_to_analyze}'")
        
    return mock_response


import sys # For sys.argv manipulation

# Import the main function from the script to be tested
from step1c_session_aware_llm_anonymize import main as step1c_main # Changed import

def run_llm_anonymize_script_direct_call():
    # Original sys.argv
    original_argv = sys.argv
    
    # Simulate command-line arguments
    sys.argv = [
        "src/step1c_session_aware_llm_anonymize.py", # Script name, as if called from command line
        "--input_file", str(INPUT_FILE),
        "--output_file", str(ACTUAL_OUTPUT_FILE),
        "--log_level", "DEBUG",
        "--ollama_model", "mock_model", # This will be used by the script
        "--ollama_host", "http://mockhost:11434", # Add host as step1c uses it
        "--max_workers", "1", # Keep low for predictable mocking
        "--chunk_size", "10000", # Add chunk_size as step1c uses it
        "--chunk_overlap", "1000" # Add chunk_overlap as step1c uses it
    ]

    mock_client_instance = MagicMock(spec=ollama.Client)
    mock_client_instance.chat.side_effect = mock_ollama_chat_response
    # Mock the 'list' method as well, as the script might call it (e.g. in get_llm_pii_analysis)
    mock_client_instance.list.return_value = {
        "models": [ # Ensure the mock_model is listed so the script doesn't fail model check
            {
                "name": "mock_model:latest", # Or whatever the script expects from --ollama_model
                "modified_at": "2023-01-01T00:00:00Z",
                "size": 123456789,
                "digest": "abcdef123456",
                "details": {"family": "mockfamily", "parameter_size": "0B", "quantization_level": "Q0"}
            }
        ]
    }

    # Patch target needs to be where ollama.Client is looked up by step1c_session_aware_llm_anonymize.py
    # Assuming it's 'step1c_session_aware_llm_anonymize.ollama.Client' if the script does 'import ollama'
    with patch('step1c_session_aware_llm_anonymize.ollama.Client', return_value=mock_client_instance) as mock_ollama_client_constructor:
        try:
            step1c_main() # Call the new script's main function
        except SystemExit as e:
            if e.code != 0 and e.code is not None:
                pytest.fail(f"LLM anonymize script (step1c) exited with code {e.code}")
        finally:
            sys.argv = original_argv
            
        mock_ollama_client_constructor.assert_called()
        # Check call count for the simplified single-turn scenario:
        # 1 call for messages of the only turn (which is also the latest)
        # 1 call for completion of the only turn
        assert mock_client_instance.chat.call_count == 2, \
            f"ollama.Client.chat was called {mock_client_instance.chat.call_count} times, expected 2 for single turn."


def test_llm_anonymize_data():
    run_llm_anonymize_script_direct_call()

    assert ACTUAL_OUTPUT_FILE.exists(), "Output file was not created by llm_anonymize script (step1c)"

    actual_data = []
    with open(ACTUAL_OUTPUT_FILE, "r") as f:
        for line in f:
            actual_data.append(json.loads(line))
    
    assert len(actual_data) == len(SAMPLE_INPUT_DATA_LLM)

    for i, actual_item in enumerate(actual_data):
        original_input_item = SAMPLE_INPUT_DATA_LLM[i]
        expected_logic = next(e for e in EXPECTED_LLM_ANONYMIZATION_LOGIC if e["trace_id"] == actual_item["trace_id"])

        # For step1c, check original_messages only if messages changed.
        if actual_item.get("messages") != original_input_item.get("messages"):
            assert actual_item.get("original_messages") is not None, f"original_messages missing for trace_id {actual_item['trace_id']} where messages changed"
            assert actual_item["original_messages"] == original_input_item["messages"]
        else:
            # If messages didn't change, original_messages should not be populated by step1c
            assert actual_item.get("original_messages") is None, f"original_messages present for trace_id {actual_item['trace_id']} where messages did not change"

        # Check original_completion only if completion changed.
        if actual_item.get("completion") != original_input_item.get("completion"):
            assert actual_item.get("original_completion") is not None, f"original_completion missing for trace_id {actual_item['trace_id']} where completion changed"
            assert actual_item["original_completion"] == original_input_item["completion"]
        else:
            # If completion didn't change, original_completion should not be populated
            assert actual_item.get("original_completion") is None, f"original_completion present for trace_id {actual_item['trace_id']} where completion did not change"
            
        assert "llm_anonymization_details" in actual_item
        details = actual_item["llm_anonymization_details"]
        assert isinstance(details, dict)

        # Check PII types found based on session-aware logic
        # step1c stores session message PII in "llm_detected_pii_items_messages_session"
        # and per-turn completion PII in "llm_detected_pii_items_completion_turn"
        # "llm_sensitive_categories_found_session" reflects categories from session messages (latest turn).
        # "llm_sensitive_categories_found_turn" reflects categories from the current turn's completion.

        # The expected_logic["llm_pii_types_found"] is a mix depending on the turn.
        # For the single turn (llm_anon_trace1), it's the latest.
        # The script's "llm_sensitive_categories_found" field in details should contain
        # all unique categories from this turn's messages and this turn's completion.

        # Directly get the overall categories found by the script for this turn's processing.
        # This key reflects all PII categories contributing to the anonymization of this turn,
        # considering session context for messages and individual analysis for completion.
        # In the single-turn test case, this should include PII from both messages and completion.
        final_actual_categories = details.get("llm_sensitive_categories_found", [])
        # This list is already sorted and unique by the script.
        
        assert final_actual_categories == sorted(expected_logic["llm_pii_types_found"]), \
            f"PII category mismatch for trace_id {actual_item['trace_id']}. Expected: {sorted(expected_logic['llm_pii_types_found'])}, Got: {final_actual_categories}"

        # Check anonymized messages and completion
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
