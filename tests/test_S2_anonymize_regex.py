import pytest
import subprocess
import json
import os
from pathlib import Path
from datasets import Dataset, load_dataset

# Define paths for test data
TEST_DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = TEST_DATA_DIR / "sample_input_for_regex_anonymization.jsonl"
# EXPECTED_OUTPUT_FILE = TEST_DATA_DIR / "expected_output_regex_anonymized.jsonl" # We'll compare field by field
ACTUAL_OUTPUT_DIR = TEST_DATA_DIR / "actual_outputs"
ACTUAL_OUTPUT_FILE = ACTUAL_OUTPUT_DIR / "actual_output_regex_anonymized.jsonl"

# Ensure actual_outputs directory exists
os.makedirs(ACTUAL_OUTPUT_DIR, exist_ok=True)

# Sample data for input file
SAMPLE_INPUT_DATA = [
    {
        "messages": [
            {"role": "user", "content": "My email is test@example.com and phone is 123-456-7890."},
            {"role": "assistant", "content": "Got it. Your IP is 192.168.1.1."}
        ],
        "completion": "Thanks! My credit card is 1234-5678-9012-3456.",
        "trace_id": "regex_anon_trace1",
        "session_id": "session_001",
        "turn_in_session_id": 1
    },
    {
        "messages": [{"role": "user", "content": "No PII here."}],
        "completion": "Okay, great.",
        "trace_id": "regex_anon_trace2",
        "session_id": "session_001",
        "turn_in_session_id": 2
    }
]

# Expected anonymized content (placeholders may vary based on script's exact replacement)
# We will check for the presence of "original_messages", "original_completion",
# and that PII in "messages" and "completion" is replaced.
# The exact replacement string (e.g., "[EMAIL_ADDRESS]") depends on the script's patterns.
# We'll assume common placeholders for this test.

EXPECTED_ANONYMIZATION_LOGIC = [
    {
        "trace_id": "regex_anon_trace1",
        "anonymized_user_messages": [
            {"role": "user", "content": "My email is [EMAIL_REDACTED] and phone is [PHONE_NUMBER_REDACTED]."},
            {"role": "assistant", "content": "Got it. Your IP is [IP_ADDRESS_REDACTED]."}
        ],
        "anonymized_completion": "Thanks! My credit card is [CREDIT_CARD_REDACTED].",
        "has_anonymization_details": True
    },
    {
        "trace_id": "regex_anon_trace2",
        "anonymized_user_messages": [{"role": "user", "content": "No PII here."}],
        "anonymized_completion": "Okay, great.",
        "has_anonymization_details": False # Or True with empty details, depends on script
    }
]


@pytest.fixture(scope="module", autouse=True)
def create_test_data_files_regex_anonymize():
    with open(INPUT_FILE, "w") as f:
        for item in SAMPLE_INPUT_DATA:
            f.write(json.dumps(item) + "\n")
    
    if os.path.exists(ACTUAL_OUTPUT_FILE):
        os.remove(ACTUAL_OUTPUT_FILE)
        
    yield

def run_regex_anonymize_script():
    script_path = Path(__file__).parent.parent / "src" / "step1_anonymize_data.py"
    cmd = [
        "python", str(script_path),
        "--input_file", str(INPUT_FILE),
        "--output_file", str(ACTUAL_OUTPUT_FILE),
        "--log_level", "DEBUG"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        print("Script stdout (regex_anonymize):", result.stdout)
        if result.stderr:
            print("Script stderr (regex_anonymize):", result.stderr)
    except subprocess.CalledProcessError as e:
        print("CalledProcessError stdout (regex_anonymize):", e.stdout)
        print("CalledProcessError stderr (regex_anonymize):", e.stderr)
        pytest.fail(f"Regex anonymize script execution failed: {e}")
    except subprocess.TimeoutExpired as e:
        print("TimeoutExpired stdout (regex_anonymize):", e.stdout)
        print("TimeoutExpired stderr (regex_anonymize):", e.stderr)
        pytest.fail(f"Regex anonymize script execution timed out: {e}")

def test_regex_anonymize_data():
    run_regex_anonymize_script()

    assert ACTUAL_OUTPUT_FILE.exists(), "Output file was not created by regex_anonymize script"

    actual_data = []
    with open(ACTUAL_OUTPUT_FILE, "r") as f:
        for line in f:
            actual_data.append(json.loads(line))
    
    assert len(actual_data) == len(SAMPLE_INPUT_DATA), \
        f"Number of records in output ({len(actual_data)}) does not match input ({len(SAMPLE_INPUT_DATA)}) for regex_anonymize"

    for i, actual_item in enumerate(actual_data):
        original_input_item = SAMPLE_INPUT_DATA[i]
        expected_logic = next(e for e in EXPECTED_ANONYMIZATION_LOGIC if e["trace_id"] == actual_item["trace_id"])

        # Check that original fields are preserved if anonymization occurred
        if expected_logic["has_anonymization_details"]:
            assert "original_messages" in actual_item, f"original_messages missing for trace_id {actual_item['trace_id']}"
            assert actual_item["original_messages"] == original_input_item["messages"], f"original_messages mismatch for trace_id {actual_item['trace_id']}"
            
            assert "original_completion" in actual_item, f"original_completion missing for trace_id {actual_item['trace_id']}"
            assert actual_item["original_completion"] == original_input_item["completion"], f"original_completion mismatch for trace_id {actual_item['trace_id']}"
            
            assert "anonymization_details" in actual_item, f"anonymization_details missing for trace_id {actual_item['trace_id']}"
            assert isinstance(actual_item["anonymization_details"], dict), "anonymization_details is not a dict"
            # Check that regex_patterns_found is not empty
            assert "regex_patterns_found" in actual_item["anonymization_details"], f"regex_patterns_found missing in anonymization_details for trace_id {actual_item['trace_id']}"
            assert len(actual_item["anonymization_details"]["regex_patterns_found"]) > 0, \
                   f"anonymization_details.regex_patterns_found should not be empty for trace_id {actual_item['trace_id']}"

        else: # No PII expected, so original fields should not be present
            assert actual_item.get("original_messages") is None, f"original_messages should be None for trace_id {actual_item['trace_id']} (no PII)"
            assert actual_item.get("original_completion") is None, f"original_completion should be None for trace_id {actual_item['trace_id']} (no PII)"
            # anonymization_details will always be present, check if regex_patterns_found is empty
            assert "anonymization_details" in actual_item, f"anonymization_details missing for trace_id {actual_item['trace_id']} (no PII case)"
            assert isinstance(actual_item["anonymization_details"], dict), "anonymization_details is not a dict for no PII case"
            assert "regex_patterns_found" in actual_item["anonymization_details"], f"regex_patterns_found missing in anonymization_details for trace_id {actual_item['trace_id']} (no PII case)"
            assert len(actual_item["anonymization_details"]["regex_patterns_found"]) == 0, \
                   f"anonymization_details.regex_patterns_found should be empty for trace_id {actual_item['trace_id']} (no PII)"


        # Check anonymized messages
        assert len(actual_item["messages"]) == len(expected_logic["anonymized_user_messages"])
        for idx, msg_pair in enumerate(zip(actual_item["messages"], expected_logic["anonymized_user_messages"])):
            actual_msg, expected_msg = msg_pair
            assert actual_msg["role"] == expected_msg["role"]
            # This is a simplified check. Real check needs to confirm PII is replaced.
            # For now, we check if the content matches the expected anonymized string.
            # A more robust test would check that original PII is NOT in actual_msg["content"]
            # and that placeholders ARE in actual_msg["content"] if PII was present.
            assert actual_msg["content"] == expected_msg["content"], \
                f"Content mismatch in message {idx} for trace_id {actual_item['trace_id']}. Expected: '{expected_msg['content']}', Got: '{actual_msg['content']}'"

        # Check anonymized completion
        assert actual_item["completion"] == expected_logic["anonymized_completion"], \
             f"Completion mismatch for trace_id {actual_item['trace_id']}. Expected: '{expected_logic['anonymized_completion']}', Got: '{actual_item['completion']}'"

        # Check other fields are preserved
        assert actual_item["trace_id"] == original_input_item["trace_id"]
        assert actual_item["session_id"] == original_input_item["session_id"]
        assert actual_item["turn_in_session_id"] == original_input_item["turn_in_session_id"]
