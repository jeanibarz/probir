import argparse
import json
import ollama
from datasets import load_dataset, Dataset
from tqdm import tqdm
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import concurrent.futures
import logging
import re
import copy # Added for deepcopy
from common_utils import (
    setup_logging,
    load_jsonl_dataset,
    save_jsonl_dataset,
    create_llm_arg_parser,
    chunk_text, # Import from common_utils
    BaseTrace,      # Added
    validate_dataset # Added
)

# Configuration constants can remain if they are specific defaults for this script,
# but create_llm_arg_parser will provide defaults for many of these.
# DEFAULT_OLLAMA_MODEL will be handled by create_llm_arg_parser's --ollama_model
# DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_MAX_WORKERS will also be handled.

# Logger will be configured by setup_logging in main
logger = logging.getLogger(__name__)

# --- Pydantic Models for LLM Output ---
class CorrectionPatterns(BaseModel):
    is_user_feedback_on_error: bool = Field(False, description="True if a user turn explicitly or implicitly points out an error, inadequacy, or misunderstanding in a PREVIOUS assistant turn.")
    is_assistant_self_correction: bool = Field(False, description="True if an assistant turn corrects an error, inadequacy, or misunderstanding it made in a PREVIOUS turn OF ITS OWN, often without direct user prompting in the immediately preceding turn for that specific correction.")
    is_assistant_error_before_correction: bool = Field(False, description="True if the assistant made an identifiable error (e.g., factual, logical, tool use error, hallucination) that is later subject to user feedback or self-correction within the provided conversation.")
    correction_rationale: Optional[str] = Field(None, description="Brief rationale if any correction pattern (user feedback or assistant self-correction) is detected. Explain what was corrected or pointed out.")

# --- System Prompt ---
SYSTEM_PROMPT_INSTRUCTION_PART = """
You are an expert AI assistant specializing in analyzing conversational dynamics. Your task is to identify specific feedback and correction patterns within a given conversational trace.
The trace consists of 'messages' (a list of turns with 'role' and 'content') and a 'final_assistant_completion' which is the last response from the assistant in this trace.

Analyze the entire conversational flow provided.

Based on your analysis, determine the following:
1.  `is_user_feedback_on_error`: Is there any turn where the 'user' provides feedback (explicitly or implicitly) on an error, mistake, misunderstanding, or inadequacy in a PREVIOUS assistant's turn?
    Examples: "No, that's not what I meant.", "That's incorrect, it should be X.", "You missed the part about Y."
2.  `is_assistant_self_correction`: Does the assistant, in any of its turns (including the 'final_assistant_completion'), correct an error, mistake, or inadequacy it made in one of ITS OWN PREVIOUS turns? This might happen after a user points out an error, or sometimes the assistant might realize its own mistake without direct immediate prompting for that specific correction.
    Examples: "Apologies, I previously stated X, but the correct information is Y.", "I seem to have misunderstood. Let me try again."
3.  `is_assistant_error_before_correction`: Did the assistant make any identifiable error (factual, logical, misinterpretation, incorrect tool use, hallucination) at any point in the conversation that was later subject to either user feedback or assistant self-correction?
4.  `correction_rationale`: If `is_user_feedback_on_error` OR `is_assistant_self_correction` is true, provide a brief explanation of what was being corrected or what feedback was given.

Provide your response as a single JSON object conforming to the following schema:
{
  "is_user_feedback_on_error": <boolean>,
  "is_assistant_self_correction": <boolean>,
  "is_assistant_error_before_correction": <boolean>,
  "correction_rationale": "<string, or null if no correction patterns found>"
}
"""

USER_PROMPT_DATA_TEMPLATE = """
Conversational Data:
MESSAGES (Input context):
{messages_json_str}

FINAL ASSISTANT COMPLETION (Target response for this trace):
{completion_str}
"""

# Local chunk_text removed, using common_utils.chunk_text

# --- LLM Interaction ---
def get_llm_correction_analysis_for_chunk(text_chunk: str, client: ollama.Client, model_name: str) -> Optional[CorrectionPatterns]:
    try:
        response = client.chat(
            model=model_name,
            messages=[{'role': 'system', 'content': text_chunk}],
            format="json"
        )
        response_content = response['message']['content']
        # Attempt to repair common JSON issues from LLMs
        response_content = response_content.strip().replace("```json", "").replace("```", "")
        
        parsed_json = json.loads(response_content)
        return CorrectionPatterns(**parsed_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError for model {model_name} (chunk): {e}. Response: {response_content[:500]}")
    except ValidationError as e:
        logger.error(f"Pydantic ValidationError for model {model_name} (chunk): {e}. Response: {response_content[:500]}")
    except ollama.ResponseError as e:
        logger.error(f"Ollama API ResponseError for model {model_name} (chunk): {e.status_code} - {e.error}")
    except Exception as e:
        logger.error(f"Unexpected error during LLM call for model {model_name} (chunk): {e}")
    return None

def get_llm_correction_analysis(
    example: Dict[str, Any],
    client: ollama.Client,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    max_workers_chunks: int
) -> Dict[str, Any]:
    """
    Analyzes a single example for correction patterns using an LLM.
    Handles chunking for the combined text of messages and completion.
    Returns a dictionary of fields to be added to the example.
    """
    # Use 'messages' and 'completion' as per BaseTrace and previous steps' outputs
    messages = example.get('messages', [])
    completion = example.get('completion', "")

    if not isinstance(messages, list): messages = []
    if not isinstance(completion, str): completion = ""

    messages_str = json.dumps(messages, indent=2)
    
    # The prompt itself will contain the structure, so we pass the raw strings
    # The LLM prompt template will be formatted with these.
    # For chunking, we might consider if the LLM can handle the full prompt + data,
    # or if the data part of the prompt needs chunking.
    # For this task, the LLM analyzes the *relationship* between turns, so chunking
    # the conversational data itself is tricky.
    # Let's try sending the full context within the prompt first. If it's too long,
    # the Ollama client/server might error out.
    # The prompt structure is fixed, only messages_json_str and completion_str vary.

    system_message_content = SYSTEM_PROMPT_INSTRUCTION_PART
    user_message_content = USER_PROMPT_DATA_TEMPLATE.format(
        messages_json_str=messages_str,
        completion_str=completion
    )
    
    # We are not chunking the input data for this specific LLM task, as the relationship
    # between turns is key. We rely on the LLM to handle the context length.
    # If context length becomes an issue, a more sophisticated approach would be needed,
    # perhaps analyzing pairs/triplets of turns or summarizing previous turns.

    llm_raw_output = None
    analysis_result = CorrectionPatterns() # Default empty result
    response = None

    try:
        response = client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_message_content},
                {'role': 'user', 'content': user_message_content}
            ],
            format="json"
        )
        llm_raw_output = response['message']['content']
        
        # Attempt to repair common JSON issues from LLMs
        llm_raw_output_cleaned = llm_raw_output.strip().replace("```json", "").replace("```", "")
        
        parsed_json = json.loads(llm_raw_output_cleaned)
        analysis_result = CorrectionPatterns(**parsed_json)

    except ollama.ResponseError as e:
        logger.error(f"Ollama API ResponseError during client.chat for model {model_name}: Status: {e.status_code}, Error: {e.error}. User message length: {len(user_message_content)}")
        # llm_raw_output might not be set or be partial here, depending on when ResponseError is raised.
        # If response object exists and has content, log it.
        if response and response.get('message') and response['message'].get('content'):
             logger.error(f"Ollama ResponseError - Partial/Error response content: {response['message']['content']}")
        llm_raw_output = f"Ollama ResponseError: {e.error}" # Store error for the output column
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError for model {model_name}: {e.msg}. Position: {e.pos}. Line: {e.lineno}, Col: {e.colno}. FULL RAW Response from LLM: {llm_raw_output}")
    except ValidationError as e:
        logger.error(f"Pydantic ValidationError for model {model_name}. Errors: {e.errors()}. FULL RAW Response from LLM: {llm_raw_output}")
    except Exception as e: # Catch any other exceptions, including those from client.chat() if not ResponseError
        logger.error(f"Unexpected error during LLM analysis or client.chat() for model {model_name}: {type(e).__name__} - {e}. User message length: {len(user_message_content)}")
        llm_raw_output = f"Unexpected error: {type(e).__name__} - {e}" # Store error for the output column


    # Prepare fields for BaseTrace
    output_fields = {
        "is_user_feedback_on_error": analysis_result.is_user_feedback_on_error,
        "is_assistant_self_correction": analysis_result.is_assistant_self_correction,
        "is_assistant_error_before_correction": analysis_result.is_assistant_error_before_correction,
        "correction_rationale": analysis_result.correction_rationale,
        "correction_analysis_details": { # Store raw output and other details here
            "llm_correction_raw_output": llm_raw_output,
            "model_used": model_name 
        }
    }
    return output_fields


def process_example_for_correction(example_with_index, client, model_name, chunk_size, chunk_overlap, max_workers_chunks):
    index, example_data = example_with_index # example_data is a dict
    # logger.debug(f"Processing example {index + 1} for correction patterns...") # Too verbose for info
    
    # Make a deep copy to avoid modifying the original dict from the dataset during processing
    # if it's shared across threads or if get_llm_correction_analysis modifies it (it shouldn't).
    current_example_copy = copy.deepcopy(example_data)

    correction_data_fields = get_llm_correction_analysis(
        current_example_copy, # Pass the copy
        client, 
        model_name, 
        chunk_size, 
        chunk_overlap, 
        max_workers_chunks
    )
    
    # Merge the new fields into the copied example
    for key, value in correction_data_fields.items():
        current_example_copy[key] = value
        
    return current_example_copy


def main():
    parser = create_llm_arg_parser(
        description="Phase 3: Analyze conversational traces for feedback/correction patterns using an LLM."
    )
    parser.set_defaults(
        input_file="sft_dataset_complexity_scored.jsonl",
        output_file="sft_dataset_correction_analyzed.jsonl"
        # Note: DEFAULT_OLLAMA_MODEL, DEFAULT_CHUNK_SIZE, etc. are effectively superseded by create_llm_arg_parser defaults
    )
    # Specific argument for max_workers for examples, if different from general max_workers for chunks
    parser.add_argument("--max_workers_examples", type=int, default=None, 
                        help="Max concurrent examples to process (defaults to general --max_workers if not set).")
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)
    
    # Module-level logger will pick up the new configuration.

    logger.info("Starting correction pattern analysis process.")
    if args.log_file_name:
        logger.info(f"Logs for this script will also be saved to logs/{args.log_file_name}")
    logger.info(f"Input file: {args.input_file}, Output file: {args.output_file}")
    logger.info(f"Ollama Model: {args.ollama_model}, Host: {args.ollama_host or 'Default (localhost or OLLAMA_HOST env)'}")
    logger.info(f"Chunk size (for LLM data if chunked by common_utils.chunk_text): {args.chunk_size}, Overlap: {args.chunk_overlap}")
    
    max_workers_for_examples = args.max_workers_examples if args.max_workers_examples is not None else args.max_workers
    logger.info(f"Max concurrent examples: {max_workers_for_examples}")
    # args.max_workers from create_llm_arg_parser can be used for max_workers_chunks if needed.

    try:
        ollama_client_args = {}
        if args.ollama_host:
            ollama_client_args['host'] = args.ollama_host
        client = ollama.Client(**ollama_client_args)
        client.list() # Test connection
        logger.info("Ollama client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Ollama client: {e}", exc_info=True)
        return

    try:
        dataset = load_jsonl_dataset(args.input_file, limit=args.limit)
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_file}: {e}", exc_info=True)
        return

    if not dataset or len(dataset) == 0:
        logger.info("Input dataset is empty. Nothing to process.")
        try:
            # Ensure an empty list is passed to Dataset.from_list for empty dataset creation
            empty_hf_dataset = Dataset.from_list([])
            save_jsonl_dataset(empty_hf_dataset, args.output_file, force_ascii=False)
            logger.info(f"Empty output file saved to {args.output_file}")
        except Exception as e:
            logger.error(f"Error saving empty dataset to {args.output_file}: {e}", exc_info=True)
        return
        
    logger.info(f"Analyzing correction patterns for {len(dataset)} examples using model {args.ollama_model}...")

    processed_examples = []
    
    # Using ThreadPoolExecutor for parallel processing of examples
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_for_examples) as executor:
        # Prepare arguments for each task
        # The 6th argument to process_example_for_correction is max_workers_chunks, which should be args.max_workers
        tasks_with_args = [
            ((idx, example), client, args.ollama_model, args.chunk_size, args.chunk_overlap, args.max_workers)
            for idx, example in enumerate(dataset)
        ]
        
        # Use a wrapper lambda to unpack arguments for map
        future_to_example_idx = {
            executor.submit(process_example_for_correction, *task_args): task_args[0][0]
            for task_args in tasks_with_args
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_example_idx), total=len(dataset), desc="Analyzing Corrections"):
            example_idx = future_to_example_idx[future]
            try:
                processed_example = future.result()
                processed_examples.append(processed_example)
            except Exception as exc:
                logger.error(f"Example {example_idx + 1} generated an exception: {exc}. Check LLM raw output if available in logs for this example.")
                # Optionally, append original example or a version with error flags
                original_example = tasks_with_args[example_idx][0][1]
                processed_examples.append({
                    **original_example,
                    "is_user_feedback_on_error": None,
                    "is_assistant_self_correction": None,
                    "is_assistant_error_before_correction": None,
                    "correction_rationale": f"Error during processing: {exc}",
                    "llm_correction_raw_output": None
                })
    
    # Sort by original index to maintain order if parallel execution shuffles them (though list.append should be fine)
    # This is more critical if results were collected in a set or dict without order.
    # For simplicity, assuming ThreadPoolExecutor + list.append preserves order of submission for results if collected sequentially.
    # However, as_completed does not guarantee order, so if order is critical, sort afterwards.
    # For now, we'll rely on the collection order from as_completed, which might not be original order.
    # If strict order is needed, results should be stored in a list of correct size and populated by index.
    # Example:
    # results = [None] * len(dataset)
    # for future in tqdm(concurrent.futures.as_completed(future_to_example_idx), total=len(dataset), desc="Analyzing Corrections"):
    #     idx = future_to_example_idx[future]
    #     try:
    #         results[idx] = future.result()
    #     except Exception as exc:
    #         logger.error(f"Example {idx + 1} generated an exception: {exc}. Storing error placeholder.")
    #         original_example = tasks_with_args[idx][0][1] # Get original example
    #         results[idx] = {
    #             **original_example,
    #             "is_user_feedback_on_error": None,
    #             "is_assistant_self_correction": None,
    #             "is_assistant_error_before_correction": None,
    #             "correction_rationale": f"Error during processing: {exc}",
    #             "llm_correction_raw_output": None
    #         }
    # processed_examples = [res for res in results if res is not None] # Filter out any potential Nones if not all futures complete

    if not processed_examples:
        logger.error("No examples were processed successfully. Exiting.")
        # Save an empty dataset if no examples were processed
        try:
            empty_hf_dataset = Dataset.from_list([])
            save_jsonl_dataset(empty_hf_dataset, args.output_file, force_ascii=False)
            logger.info(f"Empty output file saved to {args.output_file} as no examples were processed.")
        except Exception as e:
            logger.error(f"Error saving empty dataset to {args.output_file}: {e}", exc_info=True)
        return

    analyzed_dataset = Dataset.from_list(processed_examples)

    logger.info("Correction pattern analysis mapping complete.")

    # Validate the output dataset
    valid_output_examples, invalid_output_examples = validate_dataset(analyzed_dataset, BaseTrace, "OutputValidation_Step3")
    if invalid_output_examples:
        logger.warning(f"Output validation found {len(invalid_output_examples)} invalid examples after correction analysis. These will still be saved.")
        # Orchestrator will handle saving these.

    logger.info(f"Saving analyzed dataset to: {args.output_file}")
    try:
        save_jsonl_dataset(analyzed_dataset, args.output_file, force_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save analyzed dataset: {e}", exc_info=True)
        return # Stop if saving fails
        
    logger.info("Processing complete.")

    # Summary statistics
    user_feedback_count = sum(1 for ex in analyzed_dataset if ex.get("is_user_feedback_on_error") is True)
    assistant_self_correction_count = sum(1 for ex in analyzed_dataset if ex.get("is_assistant_self_correction") is True)
    assistant_error_count = sum(1 for ex in analyzed_dataset if ex.get("is_assistant_error_before_correction") is True)

    logger.info("\nCorrection Pattern Analysis Stats:")
    logger.info(f"  Total examples processed: {len(analyzed_dataset)}")
    logger.info(f"  Examples with user feedback on error: {user_feedback_count}")
    logger.info(f"  Examples with assistant self-correction: {assistant_self_correction_count}")
    logger.info(f"  Examples with assistant error before correction: {assistant_error_count}")

if __name__ == "__main__":
    main()
