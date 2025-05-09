import pytest
import subprocess
import json
import os
from pathlib import Path
from datasets import Dataset, load_dataset

# Define paths for test data
TEST_DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = TEST_DATA_DIR / "sample_input_for_sessionization.jsonl"
EXPECTED_OUTPUT_FILE = TEST_DATA_DIR / "expected_output_sessionized.jsonl"
ACTUAL_OUTPUT_DIR = TEST_DATA_DIR / "actual_outputs"
ACTUAL_OUTPUT_FILE = ACTUAL_OUTPUT_DIR / "actual_output_sessionized.jsonl"

# Ensure actual_outputs directory exists
os.makedirs(ACTUAL_OUTPUT_DIR, exist_ok=True)

# Sample data for input file
# This data should represent various scenarios for session identification
# Each "messages" field is the full prompt for its "completion"
SAMPLE_INPUT_DATA = [
    { # trace1: User starts, Assistant responds
        "messages": [{"role": "user", "content": "Hello"}],
        "completion": "Hi there!",
        "trace_id": "trace1",
    },
    { # trace2: User continues after Assistant's last response from trace1
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How can I help?"}
        ],
        "completion": "I need help with my account.",
        "trace_id": "trace2"
    },
    { # trace3: User starts a new topic (new session)
        "messages": [{"role": "user", "content": "Another question."}],
        "completion": "Sure, what is it?",
        "trace_id": "trace3"
    },
    { # trace4: User starts another new topic (new session)
        "messages": [{"role": "user", "content": "A third topic."}],
        "completion": "Okay.",
        "trace_id": "trace4"
    },
    { # trace5: User continues after Assistant's last response from trace4
        "messages": [
            {"role": "user", "content": "A third topic."},
            {"role": "assistant", "content": "Okay."},
            {"role": "user", "content": "Follow up on third topic."}
        ],
        "completion": "Thanks!",
        "trace_id": "trace5"
    }
]

# Expected output data after sessionization
# The script assigns session_id (uuid4) and turn_in_session_id
# Since session_id is random, we can't hardcode it.
# We will check for presence of session_id, turn_in_session_id, and their correct sequencing.
# For this test, we'll focus on the turn_in_session_id logic and that session_ids change when expected.

EXPECTED_TURN_LOGIC = [
    {"trace_id": "trace1", "turn_in_session_id": 1}, # New session (1-indexed)
    {"trace_id": "trace2", "turn_in_session_id": 2}, # Continues session
    {"trace_id": "trace3", "turn_in_session_id": 1}, # New session
    {"trace_id": "trace4", "turn_in_session_id": 1}, # New session
    {"trace_id": "trace5", "turn_in_session_id": 2}  # Continues session
]


@pytest.fixture(scope="module", autouse=True)
def create_test_data_files():
    # Create sample input file
    with open(INPUT_FILE, "w") as f:
        for item in SAMPLE_INPUT_DATA:
            f.write(json.dumps(item) + "\n")
    
    # Clean up actual output file if it exists from previous runs
    if os.path.exists(ACTUAL_OUTPUT_FILE):
        os.remove(ACTUAL_OUTPUT_FILE)
        
    yield # Test runs here

    # Optional: cleanup files after tests if needed, though often useful to inspect them
    # os.remove(INPUT_FILE)
    # if os.path.exists(ACTUAL_OUTPUT_FILE):
    #     os.remove(ACTUAL_OUTPUT_FILE)


def run_session_id_script():
    script_path = Path(__file__).parent.parent / "src" / "step2b_identify_sessions.py"
    cmd = [
        "python", str(script_path),
        "--input_file", str(INPUT_FILE),
        "--output_file", str(ACTUAL_OUTPUT_FILE),
        "--log_level", "DEBUG" # For more detailed logs if needed during debugging
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        print("Script stdout:", result.stdout)
        if result.stderr:
            print("Script stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("CalledProcessError stdout:", e.stdout)
        print("CalledProcessError stderr:", e.stderr)
        pytest.fail(f"Script execution failed: {e}")
    except subprocess.TimeoutExpired as e:
        print("TimeoutExpired stdout:", e.stdout)
        print("TimeoutExpired stderr:", e.stderr)
        pytest.fail(f"Script execution timed out: {e}")


def test_identify_sessions():
    run_session_id_script()

    assert ACTUAL_OUTPUT_FILE.exists(), "Output file was not created"

    actual_data = []
    with open(ACTUAL_OUTPUT_FILE, "r") as f:
        for line in f:
            actual_data.append(json.loads(line))
    
    assert len(actual_data) == len(SAMPLE_INPUT_DATA), \
        f"Number of records in output ({len(actual_data)}) does not match input ({len(SAMPLE_INPUT_DATA)})"

    # Verify session_id and turn_in_session_id logic
    # We check that session_id is present and turn_in_session_id matches expected logic.
    # We also check that session_id changes when a new session is expected.
    
    previous_session_id = None
    expected_session_index_change = [False, False, True, True, False] # True if session_id should change from previous

    for i, actual_item in enumerate(actual_data):
        expected_item_logic = next(t for t in EXPECTED_TURN_LOGIC if t["trace_id"] == actual_item["trace_id"])

        assert "session_id" in actual_item, f"session_id missing in output item {i}: {actual_item}"
        assert isinstance(actual_item["session_id"], str), f"session_id is not a string in item {i}"
        assert len(actual_item["session_id"]) > 0, f"session_id is empty in item {i}"
        
        assert "turn_in_session_id" in actual_item, f"turn_in_session_id missing in output item {i}"
        assert actual_item["turn_in_session_id"] == expected_item_logic["turn_in_session_id"], \
            f"Mismatch in turn_in_session_id for trace_id {actual_item['trace_id']}. Expected {expected_item_logic['turn_in_session_id']}, got {actual_item['turn_in_session_id']}"

        if i > 0:
            if expected_session_index_change[i]:
                assert actual_item["session_id"] != previous_session_id, \
                    f"session_id should have changed for trace_id {actual_item['trace_id']} (item {i}), but it remained {actual_item['session_id']}"
            else:
                assert actual_item["session_id"] == previous_session_id, \
                    f"session_id should NOT have changed for trace_id {actual_item['trace_id']} (item {i}), but it changed from {previous_session_id} to {actual_item['session_id']}"
        
        previous_session_id = actual_item["session_id"]

    # Further check: count distinct session_ids. Expected 3 sessions.
    distinct_session_ids = set(item["session_id"] for item in actual_data)
    assert len(distinct_session_ids) == 3, \
        f"Expected 3 distinct session_ids, but found {len(distinct_session_ids)}: {distinct_session_ids}"
