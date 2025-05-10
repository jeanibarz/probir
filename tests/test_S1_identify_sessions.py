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
# The script now outputs only the longest trace from each session.
# We expect 3 sessions from the sample data.
# Session 1: trace1, trace2 -> trace2 is longest (messages: 3)
# Session 2: trace3 -> trace3 is longest (messages: 1)
# Session 3: trace4, trace5 -> trace5 is longest (messages: 3)

EXPECTED_OUTPUT_CONTENT = [
    {
        "trace_id": "trace2", 
        "expected_session_id": "session_001", # First session identified
        "expected_turn_in_session_id": 2,
        "expected_messages_length": 3
    },
    {
        "trace_id": "trace3",
        "expected_session_id": "session_002", # Second session
        "expected_turn_in_session_id": 1,
        "expected_messages_length": 1
    },
    {
        "trace_id": "trace5",
        "expected_session_id": "session_003", # Third session
        "expected_turn_in_session_id": 2,
        "expected_messages_length": 3
    }
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
    
    assert len(actual_data) == len(EXPECTED_OUTPUT_CONTENT), \
        f"Number of records in output ({len(actual_data)}) does not match expected ({len(EXPECTED_OUTPUT_CONTENT)})"

    # Verify the content of the output records
    # The order of records in actual_data should match the order of sessions identified
    
    actual_data_map_by_trace_id = {item["trace_id"]: item for item in actual_data}

    for expected_item in EXPECTED_OUTPUT_CONTENT:
        actual_item = actual_data_map_by_trace_id.get(expected_item["trace_id"])
        assert actual_item is not None, f"Expected trace_id {expected_item['trace_id']} not found in output."

        assert "session_id" in actual_item, f"session_id missing for trace_id {expected_item['trace_id']}"
        assert actual_item["session_id"] == expected_item["expected_session_id"], \
            f"Mismatch in session_id for trace_id {expected_item['trace_id']}. Expected {expected_item['expected_session_id']}, got {actual_item['session_id']}"
        
        assert "turn_in_session_id" in actual_item, f"turn_in_session_id missing for trace_id {expected_item['trace_id']}"
        assert actual_item["turn_in_session_id"] == expected_item["expected_turn_in_session_id"], \
            f"Mismatch in turn_in_session_id for trace_id {expected_item['trace_id']}. Expected {expected_item['expected_turn_in_session_id']}, got {actual_item['turn_in_session_id']}"

        assert "messages" in actual_item, f"messages field missing for trace_id {expected_item['trace_id']}"
        assert len(actual_item["messages"]) == expected_item["expected_messages_length"], \
            f"Mismatch in messages length for trace_id {expected_item['trace_id']}. Expected {expected_item['expected_messages_length']}, got {len(actual_item['messages'])}"

    # Further check: count distinct session_ids. Expected 3 sessions.
    distinct_session_ids = set(item["session_id"] for item in actual_data)
    assert len(distinct_session_ids) == 3, \
        f"Expected 3 distinct session_ids, but found {len(distinct_session_ids)}: {distinct_session_ids}"
    
    # Check that the session IDs found are the ones we expect
    expected_session_id_set = set(e["expected_session_id"] for e in EXPECTED_OUTPUT_CONTENT)
    assert distinct_session_ids == expected_session_id_set, \
        f"Distinct session IDs found ({distinct_session_ids}) do not match expected set ({expected_session_id_set})"
