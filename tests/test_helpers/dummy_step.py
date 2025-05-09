import argparse
import json
import os
import logging
import sys

# Basic logger for the dummy step
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Dummy pipeline step for testing.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--step_name", default="UnknownStep", help="Name of the step for logging.")
    parser.add_argument("--add_field_name", help="Optional: Name of a field to add.")
    parser.add_argument("--add_field_value", help="Optional: Value of the field to add.")
    parser.add_argument("--simulate_failure", action="store_true", help="Simulate a step failure.")
    parser.add_argument("--produce_invalid_data", action="store_true", help="Produce data missing the 'id' field.")
    parser.add_argument("--log_file_name", help="Log file name (passed by orchestrator, used for file logging if needed).") # Standard arg

    args = parser.parse_args()

    # Setup file logging if log_file_name is provided (mimicking common_utils.setup_logging behavior partially)
    if args.log_file_name:
        log_dir = "logs" # Assume orchestrator ensures this exists or script handles it
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, args.log_file_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info(f"Dummy step '{args.step_name}' starting. Input: {args.input_file}, Output: {args.output_file}")

    if args.simulate_failure:
        logger.error(f"Simulating failure for step '{args.step_name}'.")
        sys.exit(1)

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    output_data = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    record = json.loads(line.strip())
                    original_id_for_log = record.get("id", "unknown_id") # For logging, if id exists
                    trace_id_for_log = record.get("trace_id", "unknown_trace_id") # For logging

                    if args.produce_invalid_data:
                        if "trace_id" in record:
                            del record["trace_id"] # Remove mandatory field 'trace_id' for BasePipelineInput
                        logger.info(f"Intentionally removed 'trace_id' field for record (original id: '{original_id_for_log}', trace_id: '{trace_id_for_log}') to make data invalid for BasePipelineInput.")
                    
                    if args.add_field_name and args.add_field_value:
                        record[args.add_field_name] = args.add_field_value
                    record[f"{args.step_name}_processed"] = True # Mark as processed by this step
                    output_data.append(record)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line: {line.strip()}")
                    continue
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            for record in output_data:
                outfile.write(json.dumps(record) + "\n")
        
        logger.info(f"Dummy step '{args.step_name}' completed. Processed {len(output_data)} records.")

    except Exception as e:
        logger.error(f"Error in dummy step '{args.step_name}': {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
