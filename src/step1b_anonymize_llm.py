import ollama
import json
import argparse # Will be removed later by create_llm_arg_parser
import copy
import concurrent.futures
import logging
import sys
from datasets import Dataset # Keep for type hints if needed
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Tuple, Dict, Any # Added Dict, Any
from common_utils import (
    setup_logging,
    load_jsonl_dataset,
    save_jsonl_dataset,
    create_llm_arg_parser,
    chunk_text
    # BaseTrace,      # Removed
    # validate_dataset # Removed
)

logger = logging.getLogger(__name__)

# --- Pydantic Model Definition ---
class PiiItem(BaseModel):
    value: str = Field(description="The exact sensitive text string found.")
    category: str = Field(description="One of the predefined PII category names (e.g., PERSON_NAME, EMAIL_ADDRESS).")

# Define a wrapper model for the response
class PiiResponse(BaseModel):
    pii_list: List[PiiItem] = Field(description="A list of found PII items. Should be an empty list if no PII is found.")

# Get the schema for the wrapper model
PII_SCHEMA = PiiResponse.model_json_schema()

# --- LLM Prompt Configuration ---
PII_CATEGORIES_PROMPT_LIST = [
    "PERSON_NAME: Full names of individuals.",
    "ORGANIZATION_NAME: Names of companies, institutions, or other organizations.",
    "LOCATION_ADDRESS: Specific street addresses, or city/state/country if highly specific and sensitive in context.",
    "PHONE_NUMBER: Telephone numbers.",
    "EMAIL_ADDRESS: Email addresses.",
    "USER_ID_USERNAME: Usernames, screen names, or other unique identifiers.",
    "PASSWORD_CREDENTIAL: Passwords or sensitive access credentials.",
    "API_KEY_SECRET: API keys, secret tokens, or other programmatic access keys.",
    "FINANCIAL_INFO: Credit card numbers, bank account details, or other financial data.",
    "HEALTH_INFO: Medical conditions, treatments, or other personal health information.",
    "PROJECT_CODENAME_INTERNAL: Internal project names or codenames that are considered sensitive.",
    "OTHER_SENSITIVE_CONTEXTUAL: Any other information that appears sensitive in the given context and should be redacted."
]

pii_list_formatted_for_prompt = '\n- '.join(PII_CATEGORIES_PROMPT_LIST)
# Updated system prompt to include the JSON schema
SYSTEM_PROMPT_TEMPLATE = f"""You are an expert PII (Personally Identifiable Information) detection system. Your task is to analyze the provided text and identify any sensitive information belonging to the following categories:
- {pii_list_formatted_for_prompt}

You MUST analyze the user's text and return ONLY a valid JSON object that strictly conforms to the following JSON schema:
```json
{json.dumps(PII_SCHEMA, indent=2)}
```
The JSON object MUST have a single key "pii_list". The value of "pii_list" MUST be a list containing objects, where each object represents a found piece of PII and has the keys "value" (the exact sensitive text string) and "category" (one of the PII category names).

Example of valid JSON output if PII is found:
```json
{{
  "pii_list": [
    {{"value": "John Doe", "category": "PERSON_NAME"}},
    {{"value": "test@example.com", "category": "EMAIL_ADDRESS"}}
  ]
}}
```

If you find no PII in the text, you MUST return a JSON object with an empty list for "pii_list":
```json
{{
  "pii_list": []
}}
```

Do not provide any explanation, preamble, or commentary outside of the JSON object itself. Only output the JSON object conforming to the schema.
"""

# OLLAMA_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_WORKERS 
# will be handled by argparse via create_llm_arg_parser later in main().
# The local chunk_text function will be removed in a subsequent step, 
# as it's now imported from common_utils.

# Removing local chunk_text function as it's imported from common_utils.py

def _analyze_chunk_ollama(
    chunk_text: str, # Changed: was chunk_text_with_offset: Tuple[str, int]
    client: ollama.Client, 
    ollama_model: str, # Added ollama_model parameter
    pii_schema_str: str # This is SYSTEM_PROMPT_TEMPLATE
) -> List[dict]:
    """
    Sends a single text chunk to Ollama for PII analysis.
    PII item coordinates from LLM will be relative to the chunk.
    """
    # chunk_text, _ = chunk_text_with_offset # Original line, offset not used

    if not chunk_text.strip():
        return []

    messages = [
        {"role": "system", "content": pii_schema_str}, # pii_schema_str is SYSTEM_PROMPT_TEMPLATE
        {"role": "user", "content": f"Please analyze the following text for PII:\n\n{chunk_text}"}
    ]
    try:
        response = client.chat(
            model=ollama_model, # Use passed ollama_model
            messages=messages,
            format="json"
        )
        response_content = response.get("message", {}).get("content", "")

        if not response_content.strip():
            logger.warning(f"LLM returned empty content for chunk: '{chunk_text[:100]}...'")
            return []
        
        try:
            # Ensure PiiResponse is defined or imported if used here
            parsed_response = PiiResponse.model_validate_json(response_content)
            return [item.model_dump() for item in parsed_response.pii_list]
        except ValidationError as e:
            logger.error(f"Pydantic Validation Error for chunk. Response: '{response_content}'. Error: {e}", exc_info=True)
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from LLM for chunk. Response: '{response_content}'. Error: {e}", exc_info=True)
            return []
            
    except Exception as e:
        logger.error(f"Error interacting with Ollama for chunk: {e}", exc_info=True)
        return []

def get_llm_pii_analysis(
    text_to_analyze: str, 
    client: ollama.Client,
    ollama_model: str, # Added
    chunk_size: int, 
    chunk_overlap: int, 
    max_workers: int, 
    pii_schema_str: str = SYSTEM_PROMPT_TEMPLATE 
) -> List[dict]:
    """
    Sends text to Ollama model for PII analysis, handling chunking and parallel processing.
    Validates against Pydantic schema, aggregates results, and returns a list of unique PII dictionaries.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return []

    # chunk_text from common_utils returns List[str]
    text_chunks = chunk_text(text_to_analyze, chunk_size, chunk_overlap)
    if not text_chunks:
        return []

    all_pii_items_raw = []
    
    # Use ThreadPoolExecutor for parallel processing of chunks
    effective_max_workers = max(1, max_workers) # max_workers will be an int from args

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        future_to_chunk_text = { # Store chunk_text for error logging if needed
            executor.submit(
                _analyze_chunk_ollama, 
                text_chunk_item, # Pass the string directly
                client, 
                ollama_model, 
                pii_schema_str
            ): text_chunk_item
            for text_chunk_item in text_chunks # Iterate over list of strings
        }
        for future in concurrent.futures.as_completed(future_to_chunk_text):
            original_chunk_text = future_to_chunk_text[future]
            try:
                pii_items_from_chunk = future.result()
                if pii_items_from_chunk:
                    all_pii_items_raw.extend(pii_items_from_chunk)
            except Exception as exc:
                # Log with part of the chunk text for context
                logger.error(f"Chunk starting with '{original_chunk_text[:50]}...' generated an exception: {exc}", exc_info=True)
                
    # Deduplicate PII items based on (value, category)
    # This is important as overlapping chunks might detect the same PII
    unique_pii_items_map = {}
    for item in all_pii_items_raw:
        # Create a hashable key for deduplication
        key = (item.get("value"), item.get("category"))
        if key not in unique_pii_items_map:
            unique_pii_items_map[key] = item
    
    # The order might be lost here, but anonymize_text_with_llm_results sorts by length anyway.
    # If original order of first appearance is important, a more complex deduplication is needed.
    return list(unique_pii_items_map.values())


def anonymize_text_with_llm_results(text_to_anonymize: str, pii_list: List[dict], llm_categories_found_in_text: set):
    """
    Replaces PII found by LLM in the text with placeholders.
    """
    if not isinstance(text_to_anonymize, str):
        logger.debug(f"Skipping LLM anonymization for non-string content: {type(text_to_anonymize)}")
        return text_to_anonymize
        
    anonymized_text = text_to_anonymize
    # Sort PII by length descending to replace longer matches first (e.g. "John Doe" before "John")
    
    pii_to_replace = sorted([(item["value"], item["category"]) for item in pii_list], key=lambda x: len(x[0]), reverse=True)

    for value, category in pii_to_replace:
        if value in anonymized_text: # Ensure the value is actually in the current state of the text
            placeholder = f"[{category.upper()}_REDACTED]"
            anonymized_text = anonymized_text.replace(value, placeholder)
            llm_categories_found_in_text.add(category)
            
    return anonymized_text

def process_example_with_llm(
    example: dict, 
    idx: int, 
    ollama_client: ollama.Client, 
    ollama_model: str, # Added
    chunk_size: int, 
    chunk_overlap: int, 
    max_workers: int
):
    """
    Processes a single example using LLM for PII detection and anonymization.
    Operates on the fields produced by the regex anonymization step.
    Dynamically injects a fake token into the first example (idx=0) for testing.
    Uses chunking for LLM analysis.
    Modifies the 'example' dict in place to conform to BaseTrace.
    """
    processed_example = copy.deepcopy(example) # Work on a copy
    
    llm_pii_categories_found_overall = set()
    llm_detected_pii_items_messages = []
    llm_detected_pii_items_completion = []
    fake_token_injected = False

    # --- Process messages ---
    # The input 'messages' field should be the one from the previous step (regex anonymized)
    # or original if regex step was skipped.
    current_messages_for_processing = processed_example.get("messages", [])
    
    if current_messages_for_processing and isinstance(current_messages_for_processing, list):
        # Create a temporary version of messages for analysis if injection is needed
        messages_to_analyze_for_llm = copy.deepcopy(current_messages_for_processing)

        # --- Dynamic Injection for the first example (idx=0) for testing LLM detection ---
        if idx == 0:
            for i, msg_dict in enumerate(messages_to_analyze_for_llm):
                if msg_dict.get("role") == "user" and isinstance(msg_dict.get("content"), str):
                    original_content = msg_dict["content"]
                    msg_dict["content"] = original_content + " My secret token is FAKE_TOKEN_12345abcdef."
                    fake_token_injected = True
                    logger.debug(f"Example {idx}: Injected fake token for LLM analysis.")
                    break 
        # --- End Injection ---

        anonymized_messages_list_for_output = []
        # Check if original_messages was already populated (e.g., by a previous regex step)
        # and is not None.
        original_messages_already_captured = processed_example.get("original_messages") is not None

        for i, message_obj_from_input in enumerate(current_messages_for_processing):
            # message_obj_from_input is from the actual input (e.g. regex-anonymized)
            # text_to_analyze_for_llm is from the (potentially) fake-token-injected version
            text_to_analyze_for_llm = messages_to_analyze_for_llm[i].get("content", "")
            
            pii_results_msg = []
            if isinstance(text_to_analyze_for_llm, str) and text_to_analyze_for_llm.strip():
                pii_results_msg = get_llm_pii_analysis(
                    text_to_analyze_for_llm, ollama_client, ollama_model,
                    chunk_size, chunk_overlap, max_workers
                )
            
            # Store all detected PII items for this message
            if pii_results_msg:
                llm_detected_pii_items_messages.extend(pii_results_msg)
                if fake_token_injected and any(item.get("value") == "FAKE_TOKEN_12345abcdef" for item in pii_results_msg):
                     logger.debug(f"Fake token detected by LLM in example {idx} messages (Category: {next((item['category'] for item in pii_results_msg if item.get('value') == 'FAKE_TOKEN_12345abcdef'), 'N/A')})")

            # Anonymize the *original* content of this message from the input example
            # using the PII found in the (potentially injected) text_to_analyze_for_llm
            message_copy_for_output = copy.deepcopy(message_obj_from_input)
            original_content_of_this_message = message_obj_from_input.get("content", "")

            if isinstance(original_content_of_this_message, str) and pii_results_msg:
                anonymized_content_this_message = anonymize_text_with_llm_results(
                    original_content_of_this_message, pii_results_msg, llm_pii_categories_found_overall
                )
                if anonymized_content_this_message != original_content_of_this_message:
                    if not original_messages_already_captured:
                        # If original_messages haven't been captured by a prior step (e.g. regex),
                        # and LLM makes a change, then current_messages_for_processing are the "originals" for this field.
                        processed_example["original_messages"] = copy.deepcopy(current_messages_for_processing)
                        original_messages_already_captured = True # Mark as captured for subsequent messages in this example
                message_copy_for_output["content"] = anonymized_content_this_message
            
            anonymized_messages_list_for_output.append(message_copy_for_output)
        
        processed_example["messages"] = anonymized_messages_list_for_output

    # --- Process completion ---
    current_completion_for_processing = processed_example.get("completion", "")
    original_completion_already_captured = processed_example.get("original_completion") is not None

    if isinstance(current_completion_for_processing, str) and current_completion_for_processing.strip():
        pii_results_compl = get_llm_pii_analysis(
            current_completion_for_processing, ollama_client, ollama_model,
            chunk_size, chunk_overlap, max_workers
        )
        if pii_results_compl:
            llm_detected_pii_items_completion.extend(pii_results_compl)
            anonymized_completion_text = anonymize_text_with_llm_results(
                current_completion_for_processing, pii_results_compl, llm_pii_categories_found_overall
            )
            if anonymized_completion_text != current_completion_for_processing:
                if not original_completion_already_captured:
                     # If original_completion hasn't been captured by a prior step,
                     # and LLM makes a change, then current_completion_for_processing is the "original" for this field.
                    processed_example["original_completion"] = current_completion_for_processing
                    # No need to set original_completion_already_captured = True here as it's the end of completion processing
            processed_example["completion"] = anonymized_completion_text

    # Store LLM anonymization details
    processed_example["llm_anonymization_details"] = {
        "llm_sensitive_categories_found": sorted(list(llm_pii_categories_found_overall)),
        "llm_detected_pii_items_messages": llm_detected_pii_items_messages, # Store raw detections
        "llm_detected_pii_items_completion": llm_detected_pii_items_completion # Store raw detections
    }

    # Clean up old keys if they were somehow present from a different version
    for old_key in ["final_anonymized_messages", "final_anonymized_completion", "llm_sensitive_categories_found"]:
        if old_key in processed_example:
            del processed_example[old_key]

    # Ensure optional fields managed by this step are present, defaulting to None if not set.
    # This helps maintain a consistent schema for the Hugging Face datasets library.
    if "original_messages" not in processed_example:
        processed_example["original_messages"] = None
    if "original_completion" not in processed_example:
        processed_example["original_completion"] = None
            
    return processed_example

def main():
    parser = create_llm_arg_parser(
        description="Anonymize sensitive data in a Hugging Face dataset using an Ollama LLM."
    )
    # --input_file, --output_file, --limit, --log_level from create_default_arg_parser
    # --ollama_host, --ollama_model, --chunk_size, --chunk_overlap, --max_workers from create_llm_arg_parser
    args = parser.parse_args()

    # Resolve config values using the new helper
    # common_utils.load_config_value will handle CLI > ENV > .env > default hierarchy
    # Ensure load_config_value is imported or defined if this script is run standalone without full project context
    # For now, assuming it's available via common_utils
    from common_utils import load_config_value 
    
    resolved_ollama_model = load_config_value("OLLAMA_MODEL", args.ollama_model, "phi3:mini")
    resolved_ollama_host = load_config_value("OLLAMA_HOST", args.ollama_host, "http://localhost:11434")


    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name) # Pass log_file_name

    # Module-level logger will use the new config.

    logger.info(f"Starting LLM-based anonymization process.")
    if args.log_file_name:
        logger.info(f"Logs for this script will also be saved to logs/{args.log_file_name}")
    logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")
    logger.info(f"Ollama Model: {resolved_ollama_model} (Resolved), Host: {resolved_ollama_host} (Resolved)")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}, Max workers: {args.max_workers}")
    if args.limit:
        logger.info(f"Processing limit: {args.limit} examples.")

    try:
        ollama_client_args = {}
        if resolved_ollama_host: # Use resolved value
            ollama_client_args['host'] = resolved_ollama_host
        ollama_client = ollama.Client(**ollama_client_args)
        # Test connection by listing models, using the resolved model name if needed for specific checks (not strictly necessary for .list())
        ollama_client.list() 
        logger.info(f"Ollama client initialized successfully for host: {resolved_ollama_host or 'default'}.")
    except Exception as e:
        logger.error(f"Error initializing Ollama client or listing models for host '{resolved_ollama_host}': {e}", exc_info=True)
        logger.error("Please ensure Ollama is running and the specified model is available.")
        sys.exit(1) # Changed to exit with error code

    try:
        dataset_to_process = load_jsonl_dataset(args.input_file, limit=args.limit)
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_file}: {e}", exc_info=True)
        return

    logger.info(f"Processing {len(dataset_to_process)} examples...")
    logger.info("Starting LLM-based anonymization mapping (this might take a while)...")

    # dataset.map creates a new dataset with the results of the function
    anonymized_dataset_llm = dataset_to_process.map(
        lambda example, idx: process_example_with_llm(
            example, 
            idx, 
            ollama_client,
            resolved_ollama_model, # Use resolved value
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_workers=args.max_workers
        ),
        with_indices=True, # Pass index to the map function
        num_proc=1 # Keep num_proc=1 for easier debugging with LLM calls
    )

    logger.info("LLM-based anonymization mapping complete.")

    # Output validation is now handled by the pipeline orchestrator (run_pipeline.py)
    # using the specific output model for this step (LlmAnonymizationOutput).
    # Removing self-validation from the script itself.
    # valid_output_examples, invalid_output_examples = validate_dataset(anonymized_dataset_llm, BaseTrace, "OutputValidation_Step1b")
    # if invalid_output_examples:
    #     logger.warning(f"Output validation found {len(invalid_output_examples)} invalid examples after LLM anonymization. These will still be saved.")
    
    try:
        save_jsonl_dataset(anonymized_dataset_llm, args.output_file, force_ascii=False)
    except Exception as e:
        logger.error(f"Error saving LLM-anonymized dataset to {args.output_file}: {e}", exc_info=True)
        return
    
    # Summarize findings
    total_examples_with_llm_pii = 0
    all_llm_pii_categories_summary = set()
    for ex in anonymized_dataset_llm:
        details = ex.get("llm_anonymization_details", {})
        categories = details.get("llm_sensitive_categories_found", [])
        if categories:
            total_examples_with_llm_pii += 1
            for cat in categories:
                all_llm_pii_categories_summary.add(cat)
                
    if all_llm_pii_categories_summary:
        logger.info(f"\nSummary of LLM-detected sensitive PII categories (found in {total_examples_with_llm_pii} examples):")
        for p_name in sorted(list(all_llm_pii_categories_summary)):
            # Count examples where this specific category was found
            count = sum(1 for ex_count in anonymized_dataset_llm 
                        if p_name in ex_count.get("llm_anonymization_details", {}).get("llm_sensitive_categories_found", []))
            logger.info(f"- {p_name}: found by LLM in {count} examples")
    else:
        logger.info("\nNo PII (based on LLM analysis) found in the dataset.")
    
    logger.info("LLM-based anonymization process finished.")

if __name__ == "__main__":
    main()
