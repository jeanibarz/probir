import re
import copy
import logging
from datasets import Dataset # Keep for type hints if needed, or remove if not directly used
from common_utils import (
    setup_logging, 
    load_jsonl_dataset, 
    save_jsonl_dataset, 
    create_default_arg_parser
    # BaseTrace,      # Removed
    # validate_dataset # Removed
)

logger = logging.getLogger(__name__)

# Define regex patterns, placeholders, and tags
# Each item is a tuple: (pattern_name, regex_pattern, placeholder)
REGEX_PATTERNS = [
    ("EMAIL", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL_REDACTED]"),
    ("GENERIC_API_KEY", r'\b((sk|pk|rk)-[a-zA-Z0-9]{20,})\b', "[API_KEY_REDACTED]"),
    ("HF_TOKEN", r'\b(hf_[a-zA-Z0-9]{30,})\b', "[HF_TOKEN_REDACTED]"),
    ("GITHUB_TOKEN", r'\b(ghp_[a-zA-Z0-9]{30,}|gho_[a-zA-Z0-9]{30,}|ghu_[a-zA-Z0-9]{30,}|ghs_[a-zA-Z0-9]{30,}|ghr_[a-zA-Z0-9]{30,})\b', "[GITHUB_TOKEN_REDACTED]"),
    ("AWS_ACCESS_KEY_ID", r'\b(AKIA[0-9A-Z]{16})\b', "[AWS_ACCESS_KEY_ID_REDACTED]"),
    # Caution: This AWS Secret Key pattern is broad and might have false positives.
    ("POTENTIAL_AWS_SECRET_KEY", r'\b([A-Za-z0-9/+=]{40})\b', "[POTENTIAL_SECRET_KEY_REDACTED]"),
    ("JWT", r'\b(ey[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+)\b', "[JWT_REDACTED]"),
    # This pattern for URL credentials replaces the user:pass@ part.
    ("URL_CREDENTIALS", r'https?://(?:[^:]+:[^@]+@)', "[URL_CREDENTIALS_REDACTED_PREFIX]"),
    # Basic pattern for 16-digit credit card numbers (e.g., XXXX-XXXX-XXXX-XXXX or XXXXXXXXXXXXXXXX)
    ("CREDIT_CARD", r'\b(?:\d{4}[- ]?){3}\d{4}\b', "[CREDIT_CARD_REDACTED]"),
    # Basic pattern for phone numbers (North American style, very generic)
    ("PHONE_NUMBER", r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b', "[PHONE_NUMBER_REDACTED]"),
    # Basic IP Address v4
    ("IP_ADDRESS_V4", r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', "[IP_ADDRESS_REDACTED]"),
]

def anonymize_text(text_to_anonymize: str, patterns_found_in_current_text: set):
    """
    Applies all regex patterns to a single text string and replaces matches.
    Returns the anonymized text and updates the set of patterns found.
    """
    if not isinstance(text_to_anonymize, str): # Handle cases where content might not be a string
        logger.debug(f"Skipping anonymization for non-string content: {type(text_to_anonymize)}")
        return text_to_anonymize

    anonymized_text = text_to_anonymize
    for name, pattern, placeholder in REGEX_PATTERNS:
        # For URL_CREDENTIALS, we want to replace the prefix but keep the rest of the URL if possible.
        # However, the current regex matches the part to be replaced directly.
        # A simple sub is fine here. If the regex was more complex, a custom repl func might be needed.
        
        # Check if pattern exists before replacing to correctly log found patterns
        if re.search(pattern, anonymized_text):
            patterns_found_in_current_text.add(name)
            anonymized_text = re.sub(pattern, placeholder, anonymized_text)
            
    return anonymized_text

def anonymize_example(example: dict):
    """
    Processes a single example (data point) from the dataset.
    Anonymizes 'content' in 'messages' and the 'completion' field.
    Updates the example dictionary in place to fit BaseTrace model.
    """
    # Make a deep copy to avoid modifying the input dict if it's part of a larger structure
    # that might be reused, though dataset.map usually handles this.
    processed_example = copy.deepcopy(example)
    
    # Ensure original_ fields are initialized to None so they are always present in the output schema
    processed_example["original_messages"] = None
    processed_example["original_completion"] = None
    
    patterns_found_in_example = set()
    # original_messages_saved and original_completion_saved are still useful to avoid overwriting
    # original_messages/completion if multiple PII instances are found in the same field.
    # However, the primary goal is to ensure the keys exist.
    # Let's simplify: if any change is made, we store the original.
    # The deepcopy of `example` at the start means `processed_example` initially has the originals.
    # We only need to populate `original_messages` and `original_completion` if a change occurs.

    # Anonymize messages
    if "messages" in processed_example and isinstance(processed_example["messages"], list):
        # Store original messages if any part of them gets anonymized
        # We need to compare the full list of messages before and after anonymization
        original_messages_for_comparison = copy.deepcopy(processed_example["messages"])
        
        anonymized_messages_list = []
        made_change_to_messages = False
        for message_dict in original_messages_for_comparison: # Iterate over a copy
            if isinstance(message_dict, dict) and "content" in message_dict:
                original_content = message_dict["content"]
                anonymized_content = anonymize_text(original_content, patterns_found_in_example)
                
                msg_copy = copy.deepcopy(message_dict) # Work on a copy of the message
                msg_copy["content"] = anonymized_content
                anonymized_messages_list.append(msg_copy)
                
                if anonymized_content != original_content:
                    made_change_to_messages = True
            else:
                anonymized_messages_list.append(copy.deepcopy(message_dict))
        
        if made_change_to_messages:
            processed_example["original_messages"] = original_messages_for_comparison
        processed_example["messages"] = anonymized_messages_list # Update with potentially anonymized messages
    
    # Anonymize completion
    if "completion" in processed_example and isinstance(processed_example["completion"], str):
        original_completion_text = processed_example["completion"] # This is the original from the input example
        anonymized_completion_text = anonymize_text(original_completion_text, patterns_found_in_example)
        
        if anonymized_completion_text != original_completion_text:
            processed_example["original_completion"] = original_completion_text # Store the original
        processed_example["completion"] = anonymized_completion_text # Update with potentially anonymized completion

    # Store anonymization details
    # Ensure this field is always present, even if empty
    processed_example["anonymization_details"] = {
        "regex_patterns_found": sorted(list(patterns_found_in_example))
    }
    
    # Remove old keys if they existed from a previous version of this script
    # This ensures the output strictly matches BaseTrace for these fields.
    if "anonymized_messages" in processed_example:
        del processed_example["anonymized_messages"]
    if "anonymized_completion" in processed_example:
        del processed_example["anonymized_completion"]
    if "sensitive_patterns_found" in processed_example: # old key name
        del processed_example["sensitive_patterns_found"]

    return processed_example

def main():
    parser = create_default_arg_parser(
        description="Anonymize sensitive data in a Hugging Face dataset using regex."
    )
    # --input_file, --output_file, --limit, --log_level are provided by create_default_arg_parser
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name) # Pass log_file_name

    # The module-level logger will now use the new configuration.
    # No need to re-assign or use global here.

    logger.info(f"Starting regex-based anonymization process.")
    if args.log_file_name:
        logger.info(f"Logs for this script will also be saved to logs/{args.log_file_name}")
    logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit} examples.")

    try:
        dataset = load_jsonl_dataset(args.input_file, limit=args.limit)
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_file}: {e}", exc_info=True)
        return

    logger.info("Starting anonymization mapping...")

    # Apply the anonymization function to each example
    # The .map() function will modify the examples and return a new dataset
    # Ensure remove_columns is used if the input dataset has columns not in BaseTrace that we don't want.
    # However, BaseTrace is designed to be additive, so unknown columns are usually ignored by Pydantic.
    # For now, let's assume the input columns are either in BaseTrace or fine to pass through if not touched.
    processed_dataset = dataset.map(anonymize_example, num_proc=1) # num_proc=1 for easier debugging

    logger.info("Anonymization mapping complete.")

    # Output validation is now handled by the pipeline orchestrator (run_pipeline.py)
    # using the specific output model for this step (RegexAnonymizationOutput).
    # Removing self-validation from the script itself.
    # valid_output_examples, invalid_output_examples = validate_dataset(processed_dataset, BaseTrace, "OutputValidation_Step1")
    # if invalid_output_examples:
    #     logger.warning(f"Output validation found {len(invalid_output_examples)} invalid examples after regex anonymization. These will still be saved.")
    
    try:
        save_jsonl_dataset(processed_dataset, args.output_file, force_ascii=False)
    except Exception as e:
        logger.error(f"Error saving anonymized dataset to {args.output_file}: {e}", exc_info=True)
        return

    # Optional: Print a summary of found patterns
    all_found_patterns_summary = set()
    total_examples_with_any_regex_pattern = 0
    for ex in processed_dataset: # Iterate over the Dataset object directly
        # Access the patterns from the new structure
        patterns_in_ex = ex.get("anonymization_details", {}).get("regex_patterns_found", [])
        if patterns_in_ex:
            total_examples_with_any_regex_pattern +=1
            for p_name in patterns_in_ex:
                all_found_patterns_summary.add(p_name)
    
    if all_found_patterns_summary:
        logger.info(f"\nSummary of sensitive pattern types found (Regex) across {total_examples_with_any_regex_pattern} examples:")
        for p_name in sorted(list(all_found_patterns_summary)):
            # Counting requires iterating again or a more complex aggregation during the map
            # For simplicity, just list found pattern types here. Detailed counts per pattern can be added if needed.
            # Example of counting (can be slow for large datasets):
            count = sum(1 for ex_count in processed_dataset if p_name in ex_count.get("anonymization_details", {}).get("regex_patterns_found", []))
            logger.info(f"- {p_name}: found in {count} examples")
    else:
        logger.info("\nNo sensitive patterns (based on defined regex) found in the dataset.")

    logger.info("Regex-based anonymization process finished.")

if __name__ == "__main__":
    main()
