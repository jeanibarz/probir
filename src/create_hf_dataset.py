import sqlite3
import json
import argparse # Added
import logging # Keep for direct logger usage if any, or for log level constants
from datasets import Dataset
from common_utils import setup_logging, save_jsonl_dataset, create_default_arg_parser # Added

# DB_PATH, TABLE_NAME, OUTPUT_FILE will be handled by argparse
SYSTEM_PROMPT = "You are a helpful assistant."

# Logger instance will be configured by setup_logging
logger = logging.getLogger(__name__)

def fetch_traces_from_db(db_path: str, table_name: str, limit: int = None):
    """Fetches rows from the specified table in the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        cursor = conn.cursor()
        query = f"SELECT request_full_json, llm_response_text FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        logger.info(f"Successfully fetched {len(rows)} rows from {table_name} in {db_path}.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"Database error while fetching traces: {e}")
        return []

def create_sft_example(request_full_json_str: str, llm_response_text: str):
    """
    Creates a dictionary with "messages" (prompt context) and "completion" (final assistant response).
    """
    try:
        request_data = json.loads(request_full_json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Skipping row due to JSON parsing error in request_full_json: {e}")
        return None

    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Assuming request_data['contents'] follows the Google Generative AI API structure
    # [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]
    # These 'contents' form the prompt leading up to the llm_response_text
    if 'contents' in request_data and isinstance(request_data['contents'], list):
        for turn in request_data['contents']:
            role = turn.get('role')
            parts = turn.get('parts')
            if role and parts and isinstance(parts, list) and parts[0].get('text'):
                text = parts[0]['text']
                # Map 'model' role from API to 'assistant' for SFT standard
                message_role = 'assistant' if role == 'model' else role
                prompt_messages.append({"role": message_role, "content": text})
            else:
                logger.warning(f"Skipping a turn in prompt_messages due to missing role, parts, or text: {turn}")
    else:
        logger.warning(f"Could not find 'contents' in request_full_json or it's not a list. Prompt will only contain system message. Request data: {request_data}")

    if not llm_response_text:
        logger.warning("Missing llm_response_text for a row. Completion will be empty. This example might be problematic for training.")
        # Consider returning None or handling this case more strictly if empty completions are not allowed
        # For now, we allow it but log a warning.
        final_completion_text = ""
    else:
        final_completion_text = llm_response_text
        
    return {"messages": prompt_messages, "completion": final_completion_text}

def main():
    parser = create_default_arg_parser(
        description="Create a Hugging Face SFT dataset from LLM traces in an SQLite database."
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default="valid_llm_traces",
        help="Name of the table in the SQLite database to fetch traces from (default: valid_llm_traces).",
    )
    # --input_file from create_default_arg_parser will be used for db_path
    # --output_file from create_default_arg_parser will be used for output_jsonl_path
    # --limit from create_default_arg_parser will be used for limiting traces
    # --log_level from create_default_arg_parser will be used for logging
    
    args = parser.parse_args()

    # Setup logging based on the command-line argument
    # Convert log_level string (e.g., "INFO") to logging constant (e.g., logging.INFO)
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name) # Pass log_file_name
    
    logger.info("Starting Hugging Face dataset creation process.")
    logger.info(f"Input DB: {args.input_file}, Table: {args.table_name}, Output: {args.output_file}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit} examples.")

    traces = fetch_traces_from_db(args.input_file, args.table_name, args.limit)
    if not traces:
        logger.error("No traces fetched. Exiting.")
        return

    processed_examples = []
    for i, trace in enumerate(traces):
        request_json_str = trace['request_full_json']
        response_text = trace['llm_response_text']
        
        if not request_json_str:
            logger.warning(f"Skipping trace {i+1} due to missing request_full_json.")
            continue
        # Empty response_text is handled in create_sft_example with a warning

        example_data = create_sft_example(request_json_str, response_text)
        if example_data:
            processed_examples.append(example_data)
        else:
            logger.warning(f"Skipped trace {i+1} as example creation failed.")

    if not processed_examples:
        logger.error("No SFT examples could be created. Exiting.")
        return

    logger.info(f"Successfully processed {len(processed_examples)} examples for SFT dataset.")

    # Create Hugging Face Dataset
    try:
        hf_dataset = Dataset.from_list(processed_examples)
        logger.info("Successfully created Hugging Face Dataset object.")
    except Exception as e:
        logger.error(f"Failed to create Hugging Face Dataset object: {e}", exc_info=True)
        logger.error("Please ensure the 'datasets' library is installed ('pip install datasets').")
        return

    # Save dataset to .jsonl file using common_utils
    try:
        save_jsonl_dataset(hf_dataset, args.output_file, force_ascii=False)
        # The save_jsonl_dataset function already logs success.
    except Exception as e:
        logger.error(f"Failed to save dataset to {args.output_file}: {e}", exc_info=True)

if __name__ == '__main__':
    main()
