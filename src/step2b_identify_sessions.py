import json # Still needed for json.dumps in debug log
import logging
from tqdm import tqdm # Keep tqdm for progress if iterating over a list
from datasets import Dataset # To create Dataset from list
from common_utils import (
    setup_logging, 
    create_default_arg_parser,
    load_jsonl_dataset, # Added
    save_jsonl_dataset, # Added
    BaseTrace,          # Added
    validate_dataset    # Added
)

# Logger will be configured in main() by setup_logging
logger = logging.getLogger(__name__)

def are_messages_equal(messages1, messages2):
    """
    Compares two lists of message dictionaries.
    """
    if not messages1 and not messages2: # Both empty
        return True
    if not messages1 or not messages2: # One empty, one not
        return False
    if len(messages1) != len(messages2):
        return False
    for msg1, msg2 in zip(messages1, messages2):
        if not isinstance(msg1, dict) or not isinstance(msg2, dict):
            # This indicates a data quality issue if it happens.
            logger.warning(f"Encountered non-dict message during comparison: {msg1}, {msg2}")
            return False
        # Comparing role and content. Add other relevant fields if necessary.
        if msg1.get('role') != msg2.get('role') or msg1.get('content') != msg2.get('content'):
            return False
    return True

def main():
    parser = create_default_arg_parser(description="Phase 2b: Identify conversation sessions in the dataset.")
    # Add any script-specific arguments here if needed in the future
    # Example: parser.add_argument("--specific_param", type=int, default=10)
    args = parser.parse_args()

    # Setup logging using the common utility
    # Convert log_level string to logging level constant
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)
    
    # Re-initialize logger for this module after setup_logging has run
    # This ensures it uses the new configuration.
    global logger
    logger = logging.getLogger(__name__)

    logger.info(f"Starting session identification for {args.input_file}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit} examples.")
    if args.log_file_name:
        logger.info(f"General logs for this script will also be saved to logs/{args.log_file_name}")


    current_session_id = 0
    turn_in_session_id = 0
    
    # Store the 'messages' and 'completion' of the previous trace
    # Ensure 'messages' is a list of dicts, 'completion' is a string.
    previous_trace_messages = None
    previous_trace_completion = None
    
    processed_lines = 0
    
    # First pass to count lines for tqdm
    total_lines = 0
    try:
        # Load dataset using common_utils
        # The limit is handled by load_jsonl_dataset
        input_dataset = load_jsonl_dataset(args.input_file, limit=args.limit)
        logger.info(f"Loaded {len(input_dataset)} examples to process.")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return
    except Exception as e:
        logger.error(f"Failed to load input dataset: {e}", exc_info=True)
        return

    # Validate input dataset (optional, but good practice)
    # We expect the input to conform to at least the basic parts of BaseTrace
    # (messages, completion)
    valid_input_examples, invalid_input_examples = validate_dataset(input_dataset, BaseTrace, "InputValidation_Step2b")
    if invalid_input_examples:
        logger.warning(f"Found {len(invalid_input_examples)} invalid examples in the input. These will be skipped or may cause errors.")
        # Depending on strictness, one might choose to filter `valid_input_examples`
        # For now, we proceed with `input_dataset` which might contain them.
        # A stricter approach: input_dataset = Dataset.from_list(valid_input_examples)

    processed_examples = [] # To store successfully processed examples as dicts
    
    debug_log_file_path = "logs/session_debug_log.txt" # Keep debug log as is
    with open(debug_log_file_path, 'w', encoding='utf-8') as debug_file:
        logger.info(f"Detailed session logic debug logs will be written to: {debug_log_file_path}")
        debug_file.write(f"Session Identification Debug Log for {args.input_file}\n\n")

        try:
            # Iterate over the loaded dataset (which is a Dataset object)
            for example_idx, current_trace_data in enumerate(tqdm(input_dataset, desc="Identifying Sessions")):
                # current_trace_data is already a dict here from the Dataset iteration
                
                # Basic validation of expected fields (already somewhat covered by BaseTrace if input was validated)
                current_messages = current_trace_data.get("messages", [])
                current_completion = current_trace_data.get("completion", "")

                if not isinstance(current_messages, list) or not all(isinstance(m, dict) for m in current_messages):
                    logger.error(f"Skipping example {example_idx} due to malformed 'messages' field: {current_messages}")
                    debug_file.write(f"Example {example_idx}: ERROR - Malformed 'messages' field.\n---\n")
                    continue
                
                if not isinstance(current_completion, str):
                    logger.error(f"Skipping example {example_idx} due to malformed 'completion' field (not a string): {current_completion}")
                    debug_file.write(f"Example {example_idx}: ERROR - Malformed 'completion' field (not a string).\n---\n")
                    continue

                debug_file.write(f"  Current messages: {json.dumps(current_messages)}\n")
                debug_file.write(f"  Current completion: {json.dumps(current_completion)}\n")

                is_new_session = True # Assume new session by default

                if previous_trace_messages is not None and previous_trace_completion is not None:
                    expected_prefix_if_continuation = previous_trace_messages + [{"role": "assistant", "content": previous_trace_completion}]
                    current_messages_context = current_messages[:-1] if len(current_messages) > 0 else []

                    debug_file.write(f"  Previous messages: {json.dumps(previous_trace_messages)}\n")
                    debug_file.write(f"  Previous completion: {json.dumps(previous_trace_completion)}\n")
                    debug_file.write(f"  Expected prefix if continuation: {json.dumps(expected_prefix_if_continuation)}\n")
                    debug_file.write(f"  Current messages context (current_messages[:-1]): {json.dumps(current_messages_context)}\n")

                    if are_messages_equal(current_messages_context, expected_prefix_if_continuation):
                        is_new_session = False
                        turn_in_session_id += 1
                        debug_file.write(f"  Comparison: MATCH - Continuing session {current_session_id}, Turn {turn_in_session_id}.\n")
                    else:
                        debug_file.write(f"  Comparison: NO MATCH - Starting new session.\n")
                else:
                    debug_file.write(f"  No previous trace, starting new session.\n")

                if is_new_session:
                    current_session_id += 1
                    turn_in_session_id = 1
                    debug_file.write(f"  New Session: ID {current_session_id}, Turn {turn_in_session_id}.\n")
                
                # Add session info to the current trace data (which is a dict)
                current_trace_data["session_id"] = f"session_{current_session_id:03d}"
                current_trace_data["turn_in_session_id"] = turn_in_session_id
                
                processed_examples.append(current_trace_data) # Collect processed examples
                
                previous_trace_messages = current_messages
                previous_trace_completion = current_completion
                
                processed_lines += 1 # Keep track of successfully processed examples
                debug_file.write("---\n")
        
        except Exception as e:
            logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
            debug_file.write(f"FATAL ERROR: {e}\n")

    logger.info(f"Session identification complete. Processed {processed_lines} examples.")
    # Removed total_lines comparison as it's not directly comparable with dataset iteration easily
    logger.info(f"Total sessions identified: {current_session_id}")

    if not processed_examples:
        logger.warning("No examples were processed successfully. Output file will be empty or not created if saving fails.")
        # Create an empty dataset to ensure the output file is created as per pipeline expectations
        output_dataset = Dataset.from_list([])
    else:
        # Create Hugging Face Dataset from processed examples
        try:
            output_dataset = Dataset.from_list(processed_examples)
            logger.info(f"Successfully created output dataset with {len(output_dataset)} examples.")
        except Exception as e:
            logger.error(f"Failed to create Hugging Face Dataset from processed examples: {e}", exc_info=True)
            return # Exit if dataset creation fails

    # Validate the output dataset before saving (good practice)
    valid_output_examples, invalid_output_examples = validate_dataset(output_dataset, BaseTrace, "OutputValidation_Step2b")
    if invalid_output_examples:
        logger.warning(f"Output validation found {len(invalid_output_examples)} invalid examples. These will still be saved.")
        # Further actions could be taken here, e.g., saving invalid ones separately or halting.
        # For now, we just log. The run_pipeline.py orchestrator will also log and save them.

    # Save the output dataset using common_utils
    try:
        save_jsonl_dataset(output_dataset, args.output_file)
        # save_jsonl_dataset logs its own success message
    except Exception as e:
        logger.error(f"Failed to save output dataset to {args.output_file}: {e}", exc_info=True)

    logger.info(f"Debug log saved to: {debug_log_file_path}")

if __name__ == "__main__":
    main()
