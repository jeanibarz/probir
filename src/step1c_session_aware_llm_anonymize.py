import ollama
import json
import copy
import concurrent.futures
import logging
import sys
from datasets import Dataset, load_dataset # Removed Features, Value, Sequence as they will come from common_utils Features object
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Tuple, Dict, Any, DefaultDict
from collections import defaultdict

from common_utils import (
    setup_logging,
    load_jsonl_dataset,
    save_jsonl_dataset,
    create_llm_arg_parser,
    chunk_text,
    load_config_value,
    # Import the Pydantic model for this script's output
    LlmAnonymizationOutput,
    # Import the Features object for this script's output
    llm_anonymization_output_features
)

logger = logging.getLogger(__name__)

# --- Pydantic Model Definition (Copied from step1b) ---
class PiiItem(BaseModel):
    value: str = Field(description="The exact sensitive text string found.")
    category: str = Field(description="One of the predefined PII category names (e.g., PERSON_NAME, EMAIL_ADDRESS).")

class PiiResponse(BaseModel):
    pii_list: List[PiiItem] = Field(description="A list of found PII items. Should be an empty list if no PII is found.")

PII_SCHEMA = PiiResponse.model_json_schema()

# --- LLM Prompt Configuration (Copied from step1b) ---
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

# --- Core LLM Interaction and Anonymization Functions (Copied from step1b) ---
def _analyze_chunk_ollama(
    chunk_text_str: str, 
    client: ollama.Client, 
    ollama_model: str,
    system_prompt: str 
) -> List[dict]:
    if not chunk_text_str.strip():
        return []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please analyze the following text for PII:\n\n{chunk_text_str}"}
    ]
    try:
        response = client.chat(model=ollama_model, messages=messages, format="json")
        response_content = response.get("message", {}).get("content", "")
        logger.debug(f"_analyze_chunk_ollama in step1c received response_content: {response_content}") # ADDED DEBUG
        if not response_content.strip():
            logger.warning(f"LLM returned empty content for chunk: '{chunk_text_str[:100]}...'")
            return []
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
    ollama_model: str,
    chunk_size: int, 
    chunk_overlap: int, 
    max_workers: int, 
    system_prompt: str = SYSTEM_PROMPT_TEMPLATE 
) -> List[dict]:
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return []
    text_chunks = chunk_text(text_to_analyze, chunk_size, chunk_overlap)
    if not text_chunks:
        return []
    all_pii_items_raw = []
    effective_max_workers = max(1, max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
        future_to_chunk_text = {
            executor.submit(_analyze_chunk_ollama, text_chunk_item, client, ollama_model, system_prompt): text_chunk_item
            for text_chunk_item in text_chunks
        }
        for future in concurrent.futures.as_completed(future_to_chunk_text):
            original_chunk_text = future_to_chunk_text[future]
            try:
                pii_items_from_chunk = future.result()
                if pii_items_from_chunk:
                    all_pii_items_raw.extend(pii_items_from_chunk)
            except Exception as exc:
                logger.error(f"Chunk starting with '{original_chunk_text[:50]}...' generated an exception: {exc}", exc_info=True)
    unique_pii_items_map = {}
    for item in all_pii_items_raw:
        key = (item.get("value"), item.get("category"))
        if key not in unique_pii_items_map:
            unique_pii_items_map[key] = item
    return list(unique_pii_items_map.values())

def anonymize_text_with_llm_results(text_to_anonymize: str, pii_list: List[dict], llm_categories_found_in_text_set: set):
    if not isinstance(text_to_anonymize, str):
        logger.debug(f"Skipping LLM anonymization for non-string content: {type(text_to_anonymize)}")
        return text_to_anonymize
    anonymized_text = text_to_anonymize
    pii_to_replace = sorted([(item["value"], item["category"]) for item in pii_list], key=lambda x: len(x[0]), reverse=True)
    for value, category in pii_to_replace:
        if value in anonymized_text:
            placeholder = f"[{category.upper()}_REDACTED]"
            anonymized_text = anonymized_text.replace(value, placeholder)
            llm_categories_found_in_text_set.add(category)
    return anonymized_text

# --- Simplified Per-Example Processing Logic ---
def process_example_llm_anonymization(
    example_data: Dict[str, Any],
    ollama_client: ollama.Client,
    ollama_model: str,
    chunk_size: int,
    chunk_overlap: int,
    max_workers: int
) -> Optional[Dict[str, Any]]: # Return Optional in case of validation failure

    processed_example = copy.deepcopy(example_data)
    trace_id = processed_example.get("trace_id", "N/A") # For logging

    # --- Step 1: Collect PII from all messages and completion in this single trace ---
    all_pii_items_for_trace_messages_list = []
    all_pii_items_for_trace_completion_list = []
    overall_pii_categories_found_in_trace = set()

    # Analyze messages
    current_messages = processed_example.get("messages", [])
    if isinstance(current_messages, list):
        for msg_idx, message_obj in enumerate(current_messages):
            original_msg_content = message_obj.get("content")
            if isinstance(original_msg_content, str) and original_msg_content.strip():
                logger.debug(f"Trace {trace_id}, Msg Idx {msg_idx}: Analyzing content for PII: '{original_msg_content[:100]}...'")
                pii_for_this_content = get_llm_pii_analysis(
                    original_msg_content, ollama_client, ollama_model,
                    chunk_size, chunk_overlap, max_workers
                )
                if pii_for_this_content:
                    all_pii_items_for_trace_messages_list.extend(pii_for_this_content)
                    for pii_item in pii_for_this_content:
                        overall_pii_categories_found_in_trace.add(pii_item.get("category"))
            # else: content is not string or is empty, no PII analysis needed

    # Analyze completion
    current_completion = processed_example.get("completion", "")
    if isinstance(current_completion, str) and current_completion.strip():
        logger.debug(f"Trace {trace_id}: Analyzing completion for PII: '{current_completion[:100]}...'")
        pii_for_completion = get_llm_pii_analysis(
            current_completion, ollama_client, ollama_model,
            chunk_size, chunk_overlap, max_workers
        )
        if pii_for_completion:
            all_pii_items_for_trace_completion_list.extend(pii_for_completion)
            for pii_item in pii_for_completion:
                overall_pii_categories_found_in_trace.add(pii_item.get("category"))

    # Combine all PII found in this trace (messages + completion) and make unique for anonymization
    combined_pii_for_anonymization_map = {}
    for item_list in [all_pii_items_for_trace_messages_list, all_pii_items_for_trace_completion_list]:
        for item in item_list:
            key = (item.get("value"), item.get("category"))
            if key not in combined_pii_for_anonymization_map:
                combined_pii_for_anonymization_map[key] = item
    
    master_pii_list_for_trace_anonymization = list(combined_pii_for_anonymization_map.values())

    # --- Step 2: Apply Anonymization using the master PII list for this trace ---
    
    # Anonymize messages
    anonymized_trace_messages_list = []
    original_messages_captured = processed_example.get("original_messages") is not None # Check if already captured
    made_change_to_messages = False

    if isinstance(current_messages, list):
        for message_obj in current_messages: # Iterate original messages from processed_example
            message_copy_for_output = copy.deepcopy(message_obj)
            original_content_of_this_message = message_obj.get("content")
            
            if isinstance(original_content_of_this_message, str) and master_pii_list_for_trace_anonymization:
                temp_categories_found_in_this_text = set() # This set is updated by anonymize_text_with_llm_results
                anonymized_content_this_message = anonymize_text_with_llm_results(
                    original_content_of_this_message, 
                    master_pii_list_for_trace_anonymization, 
                    temp_categories_found_in_this_text 
                )
                if anonymized_content_this_message != original_content_of_this_message:
                    made_change_to_messages = True
                message_copy_for_output["content"] = anonymized_content_this_message
            
            anonymized_trace_messages_list.append(message_copy_for_output)

        if made_change_to_messages and not original_messages_captured:
            processed_example["original_messages"] = copy.deepcopy(current_messages) # Save original before overwriting
        
        processed_example["messages"] = anonymized_trace_messages_list

    # Anonymize completion
    original_completion_captured = processed_example.get("original_completion") is not None
    
    if isinstance(current_completion, str) and master_pii_list_for_trace_anonymization:
        temp_categories_found_in_completion_text = set()
        anonymized_completion_text = anonymize_text_with_llm_results(
            current_completion, master_pii_list_for_trace_anonymization, temp_categories_found_in_completion_text
        )
        if anonymized_completion_text != current_completion:
            if not original_completion_captured:
                processed_example["original_completion"] = current_completion # Save original
        processed_example["completion"] = anonymized_completion_text
        
    # --- Step 3: Store LLM Anonymization Details ---
    # Deduplicate PII items for reporting (messages and completion separately)
    unique_trace_messages_pii_map = {}
    for item in all_pii_items_for_trace_messages_list: # Use the list of PII found *only* in messages
        key = (item.get("value"), item.get("category"))
        if key not in unique_trace_messages_pii_map:
            unique_trace_messages_pii_map[key] = item
    
    unique_trace_completion_pii_map = {}
    for item in all_pii_items_for_trace_completion_list: # Use the list of PII found *only* in completion
        key = (item.get("value"), item.get("category"))
        if key not in unique_trace_completion_pii_map:
             unique_trace_completion_pii_map[key] = item

    processed_example["llm_anonymization_details"] = {
        "llm_sensitive_categories_found": sorted(list(overall_pii_categories_found_in_trace)),
        "llm_detected_pii_items_messages": list(unique_trace_messages_pii_map.values()),
        "llm_detected_pii_items_completion": list(unique_trace_completion_pii_map.values())
    }

    # --- Step 4: Ensure schema consistency and validate ---
    # (Copied defensive checks and Pydantic validation from original process_session_with_llm)
    if processed_example.get("session_id") is None:
        processed_example["session_id"] = ""
    if processed_example.get("completion") is None:
        processed_example["completion"] = ""
    
    current_messages_list_for_cleanup = processed_example.get("messages")
    if isinstance(current_messages_list_for_cleanup, list):
        for msg_obj in current_messages_list_for_cleanup:
            if isinstance(msg_obj, dict):
                if msg_obj.get("role") is None: msg_obj["role"] = ""
                if msg_obj.get("content") is None: msg_obj["content"] = ""
    elif current_messages_list_for_cleanup is None:
         processed_example["messages"] = []
    
    if "original_messages" not in processed_example or processed_example.get("original_messages") is None:
        processed_example["original_messages"] = []
    
    if "original_completion" not in processed_example or processed_example.get("original_completion") is None:
        processed_example["original_completion"] = ""

    if "llm_anonymization_details" in processed_example:
        details_dict = processed_example["llm_anonymization_details"]
        if not isinstance(details_dict.get("llm_detected_pii_items_messages"), list):
            details_dict["llm_detected_pii_items_messages"] = []
        if not isinstance(details_dict.get("llm_detected_pii_items_completion"), list):
            details_dict["llm_detected_pii_items_completion"] = []

    try:
        model_instance = LlmAnonymizationOutput.model_validate(processed_example)
        final_dict_for_dataset = model_instance.model_dump(mode='json')
        return final_dict_for_dataset
    except ValidationError as e_val:
        logger.error(f"Pydantic validation failed for trace {trace_id} after LLM anonymization: {e_val}. Record will be excluded.")
        return None


def main():
    parser = create_llm_arg_parser(
        description="LLM-based anonymization of sensitive data (simplified per-trace processing)." # Updated description
    )
    args = parser.parse_args()

    resolved_ollama_model = load_config_value("OLLAMA_MODEL", args.ollama_model, "phi3:mini")
    resolved_ollama_host = load_config_value("OLLAMA_HOST", args.ollama_host, "http://localhost:11434")

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)

    logger.info(f"Starting LLM-based anonymization process (per-trace).") # Updated log
    if args.log_file_name:
        logger.info(f"Logs for this script will also be saved to logs/{args.log_file_name}")
    logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")
    logger.info(f"Ollama Model: {resolved_ollama_model}, Host: {resolved_ollama_host}")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}, Max workers: {args.max_workers}")
    if args.limit:
        logger.info(f"Processing limit (applied to initial dataset load): {args.limit} examples.")

    try:
        ollama_client_args = {}
        if resolved_ollama_host:
            ollama_client_args['host'] = resolved_ollama_host
        ollama_client = ollama.Client(**ollama_client_args)
        ollama_client.list()
        logger.info(f"Ollama client initialized successfully for host: {resolved_ollama_host or 'default'}.")
    except Exception as e:
        logger.error(f"Error initializing Ollama client for host '{resolved_ollama_host}': {e}", exc_info=True)
        sys.exit(1)

    try:
        input_dataset = load_jsonl_dataset(args.input_file, limit=args.limit)
        logger.info(f"Loaded {len(input_dataset)} examples to process.")
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_file}: {e}", exc_info=True)
        return

    # Removed session grouping logic. Process each example independently.
    all_processed_examples = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_trace_id = {
            executor.submit(
                process_example_llm_anonymization,
                example, # Pass the example dict directly
                ollama_client,
                resolved_ollama_model,
                args.chunk_size,
                args.chunk_overlap,
                args.max_workers # This max_workers is for sub-chunk processing within get_llm_pii_analysis
            ): example.get("trace_id", f"index_{i}") 
            for i, example in enumerate(input_dataset)
        }

        for future in concurrent.futures.as_completed(future_to_trace_id):
            trace_id_for_log = future_to_trace_id[future]
            try:
                processed_example_result = future.result()
                if processed_example_result: # Check if not None (due to validation failure)
                    all_processed_examples.append(processed_example_result)
            except Exception as exc:
                logger.error(f"Trace {trace_id_for_log} generated an exception during processing: {exc}", exc_info=True)
                # Optionally, could add a placeholder or the original example if needed for error tracking

    logger.info(f"LLM anonymization complete. Successfully processed {len(all_processed_examples)} examples.") # Updated log

    # Manual save to ensure List[Dict] structure for messages/original_messages
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for record in all_processed_examples:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Successfully saved {len(all_processed_examples)} records to {args.output_file} via manual JSON dump.")
    except Exception as e:
        logger.error(f"Error saving LLM-anonymized dataset to {args.output_file} via manual JSON dump: {e}", exc_info=True)
        return

    # For summary, create the Dataset directly from the processed examples in memory.
    summary_dataset: Dataset
    if all_processed_examples:
        try:
            # Use the imported llm_anonymization_output_features for schema consistency
            summary_dataset = Dataset.from_list(all_processed_examples, features=llm_anonymization_output_features)
        except Exception as e:
            logger.error(f"Error creating Dataset from_list for summary: {e}", exc_info=True)
            # Fallback to creating dataset without explicit features if schema fails, for basic summary
            summary_dataset = Dataset.from_list(all_processed_examples) if all_processed_examples else Dataset.from_list([])
    else:
        logger.warning("No examples were successfully processed for summary.")
        summary_dataset = Dataset.from_list([], features=llm_anonymization_output_features)


    # Summarize findings (similar to step1b)
    total_examples_with_llm_pii = 0
    all_llm_pii_categories_summary = set()
    
    # Ensure summary_dataset is not empty and has the expected column
    if summary_dataset and "llm_anonymization_details" in summary_dataset.column_names:
        for ex in summary_dataset:
            details = ex.get("llm_anonymization_details", {})
            categories = details.get("llm_sensitive_categories_found", [])
            if categories: 
                total_examples_with_llm_pii += 1
            for cat in categories: 
                all_llm_pii_categories_summary.add(cat)
    elif summary_dataset: # Dataset exists but maybe not the details column (e.g. if features failed)
        logger.warning("Could not perform detailed PII summary as 'llm_anonymization_details' column might be missing or features object failed.")


    if all_llm_pii_categories_summary: 
        logger.info(f"\nSummary of LLM-detected sensitive PII categories (found in {total_examples_with_llm_pii} examples with PII):")
        for p_name in sorted(list(all_llm_pii_categories_summary)):
            count = 0
            if summary_dataset and "llm_anonymization_details" in summary_dataset.column_names:
                 count = sum(1 for ex_count in summary_dataset
                            if p_name in ex_count.get("llm_anonymization_details", {}).get("llm_sensitive_categories_found", []))
            logger.info(f"- {p_name}: found by LLM in {count} examples")
    else:
        logger.info("\nNo PII (based on LLM analysis) found in the dataset.")
    
    logger.info("LLM-based anonymization process finished.") # Updated log

if __name__ == "__main__":
    main()
