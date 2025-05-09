import json
# import sqlite3 # Not used directly in this script after refactor
from datasets import Dataset
from tqdm import tqdm
import re
import logging
import copy # Added for deepcopy if modifying example in place
from common_utils import (
    setup_logging,
    load_jsonl_dataset,
    save_jsonl_dataset,
    create_default_arg_parser
    # BaseTrace,      # Removed
    # validate_dataset # Removed
)

# Placeholder for ollama and pydantic if we use LLM-based scoring later
# import ollama
# from pydantic import BaseModel, Field
# from typing import List, Optional

logger = logging.getLogger(__name__) # Added

# --- Configuration for Heuristic Scoring ---
COMPLEXITY_KEYWORDS = [
    "explain", "debug", "code", "implement", "algorithm", "complex", "step-by-step",
    "error", "fix", "troubleshoot", "analyze", "refactor", "optimize", "integrate",
    "configure", "deploy", "architecture", "database", "security", "performance"
]

# --- Heuristic Scoring Functions ---

def calculate_heuristic_score(example_data: dict):
    """
    Calculates a heuristic complexity score for a single example.
    Uses 'messages' and 'completion' fields from the example_data.
    """
    # Use 'messages' and 'completion' as per BaseTrace and previous steps' outputs
    messages = example_data.get('messages', []) 
    completion = example_data.get('completion', "")

    if not isinstance(messages, list):
        messages = []
    if not isinstance(completion, str):
        completion = ""

    num_turns = len(messages)
    total_message_length = sum(len(msg.get('content', '')) for msg in messages if isinstance(msg, dict))
    completion_length = len(completion)

    # Keyword scoring
    keyword_score = 0
    full_text_for_keywords = " ".join(msg.get('content', '') for msg in messages if isinstance(msg, dict)) + " " + completion
    for keyword in COMPLEXITY_KEYWORDS:
        keyword_score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', full_text_for_keywords, re.IGNORECASE))

    # Basic weighting (can be refined)
    # Max score for each component can be capped, e.g. at 20-30 points each for a total of 100
    # For now, let's use raw values and normalize/scale later if needed.
    
    # Score components:
    # 1. Number of turns (e.g., 1 point per turn, cap at 20)
    # 2. Total message length (e.g., 1 point per 100 chars, cap at 20)
    # 3. Completion length (e.g., 1 point per 50 chars, cap at 20)
    # 4. Keyword count (e.g., 2 points per keyword, cap at 40)

    score_num_turns = min(num_turns * 2, 25) # Max 25 points
    score_message_length = min(total_message_length // 100, 25) # Max 25 points
    score_completion_length = min(completion_length // 75, 25) # Max 25 points
    score_keywords = min(keyword_score * 2, 25) # Max 25 points
    
    heuristic_score = score_num_turns + score_message_length + score_completion_length + score_keywords
    
    return {
        "heuristic_score_num_turns": score_num_turns,
        "heuristic_score_message_length": score_message_length,
        "heuristic_score_completion_length": score_completion_length,
        "heuristic_score_keywords": score_keywords,
        "heuristic_complexity_score": heuristic_score
    }

# --- LLM-based Scoring (Placeholder) ---

# class LLMComplexityResponse(BaseModel):
#     llm_complexity_score: int = Field(..., description="Complexity score from 1 (low) to 10 (high)")
#     rationale: str = Field(..., description="Brief rationale for the score")

# SYSTEM_PROMPT_LLM_COMPLEXITY = """
# You are an expert AI assistant tasked with evaluating the complexity of a given conversational exchange.
# The exchange consists of a series of messages (user, assistant, system prompts) and a final assistant completion.
# Your goal is to assign a complexity score from 1 (very simple, e.g., basic Q&A) to 10 (very complex, e.g., multi-step reasoning, coding, detailed explanation of intricate topics).
# Consider factors like:
# - Number of turns and length of the conversation.
# - Depth of knowledge required.
# - Presence of coding, debugging, or complex instructions.
# - Multi-step reasoning or problem-solving.
# - Ambiguity or underspecification in user requests.
# - Sophistication of the assistant's response.

# Analyze the following conversational data:
# MESSAGES:
# {messages_json_str}

# FINAL ASSISTANT COMPLETION:
# {completion_str}

# Provide your response as a JSON object with the following schema:
# {
#   "llm_complexity_score": <integer between 1 and 10>,
#   "rationale": "<string, your brief explanation for the score>"
# }
# """

# def get_llm_complexity_score(example, client, model_name):
#     """
#     (Placeholder) Gets complexity score from an LLM.
#     """
#     # This function would format the input, call the LLM, parse the response,
#     # and handle errors.
#     # For now, it returns a placeholder.
#     return {
#         "llm_complexity_score": 0, # Placeholder
#         "llm_complexity_rationale": "LLM scoring not yet implemented."
#     }


def process_example(example):
    """
    Applies all scoring methods to a single example.
    Modifies the example dict to include complexity_score and complexity_reasoning.
    """
    processed_example = copy.deepcopy(example) # Work on a copy

    heuristic_scores_details = calculate_heuristic_score(processed_example)
    
    # Align with BaseTrace model fields
    processed_example["complexity_score"] = heuristic_scores_details.get("heuristic_complexity_score", 0.0)
    
    # Construct a reasoning string
    reasoning_parts = [
        f"Turns: {heuristic_scores_details.get('heuristic_score_num_turns', 0)}",
        f"MsgLen: {heuristic_scores_details.get('heuristic_score_message_length', 0)}",
        f"CompLen: {heuristic_scores_details.get('heuristic_score_completion_length', 0)}",
        f"Keywords: {heuristic_scores_details.get('heuristic_score_keywords', 0)}"
    ]
    processed_example["complexity_reasoning"] = f"Heuristic - ({'; '.join(reasoning_parts)})"

    # Remove intermediate heuristic score fields if they were added to example directly
    # and are not part of BaseTrace, or keep them if desired for detailed output.
    # For now, let's assume they are not part of BaseTrace and should not be in the final output example
    # unless explicitly added to BaseTrace as optional fields.
    # The current calculate_heuristic_score returns a dict, which is then merged.
    # We should merge selectively or ensure BaseTrace has these.
    # Let's adjust to only add the final score and reasoning to the example.
    
    # Clean up intermediate scores from the example if they were merged directly
    # This is safer if process_example was `return {**example, **heuristic_scores_details}`
    # Since we are modifying `processed_example` and only adding specific keys, this is less critical.
    # However, if `calculate_heuristic_score` modified `example` in place, this would be important.
    # For clarity, let's ensure only BaseTrace fields are in the final output from this function.
    
    # Placeholder for LLM scoring - if implemented, it would add its own fields
    # For now, we can add placeholder fields if they are in BaseTrace
    if "llm_complexity_score" not in processed_example: # Check if already added by a previous (hypothetical) step
         processed_example["llm_complexity_score"] = None # Or 0.0, depending on BaseTrace definition
    if "llm_complexity_rationale" not in processed_example:
         processed_example["llm_complexity_rationale"] = None # Or "LLM scoring not implemented"

    return processed_example

def main():
    parser = create_default_arg_parser(description="Phase 2: Score complexity of conversational traces.")
    # Add any LLM-specific args here if LLM scoring is re-enabled
    # Example: parser.add_argument("--ollama_model", type=str, help="Ollama model for LLM-based scoring.")
    # The default input/output files from create_default_arg_parser might be overridden if needed,
    # or this script can just use the defaults provided by create_default_arg_parser.
    # For now, we assume the defaults from create_default_arg_parser are sufficient or will be overridden by user.
    # If specific defaults are needed for *this script only* that differ from common_utils,
    # they can be set after calling create_default_arg_parser:
    # parser.set_defaults(input_file="sft_dataset_anonymized_llm.jsonl", 
    #                     output_file="sft_dataset_complexity_scored.jsonl")
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)

    # The module-level logger will now use the new configuration.
    # No need to re-assign or use global here.

    logger.info("Starting complexity scoring process.")
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

    logger.info("Scoring complexity for each example...")
    
    # Using map for processing is generally good, but tqdm with manual loop is fine for visibility
    # For consistency with other scripts that might use .map(), let's switch to it.
    # If tqdm is desired with .map(), it can sometimes be integrated, or log progress periodically.
    # For simplicity here, we'll use .map and log start/end.
    
    # Use .map for processing
    scored_dataset = dataset.map(process_example, num_proc=1) # num_proc=1 for easier debugging

    logger.info("Complexity scoring complete.")

    # Output validation is now handled by the pipeline orchestrator (run_pipeline.py)
    # using the specific output model for this step (ComplexityScoringOutput).
    # Removing self-validation from the script itself.
    # valid_output_examples, invalid_output_examples = validate_dataset(scored_dataset, BaseTrace, "OutputValidation_Step2")
    # if invalid_output_examples:
    #     logger.warning(f"Output validation found {len(invalid_output_examples)} invalid examples after complexity scoring. These will still be saved.")

    logger.info(f"Saving scored dataset to: {args.output_file}")
    try:
        save_jsonl_dataset(scored_dataset, args.output_file, force_ascii=False)
    except Exception as e:
        logger.error(f"Error saving scored dataset to {args.output_file}: {e}", exc_info=True)
        return
    
    logger.info("Processing complete.")

    # Log summary statistics using the 'complexity_score' field
    complexity_scores = [ex.get('complexity_score', 0.0) for ex in scored_dataset if ex.get('complexity_score') is not None]
    if complexity_scores:
        avg_complexity_score = sum(complexity_scores) / len(complexity_scores)
        min_complexity_score = min(complexity_scores)
        max_complexity_score = max(complexity_scores)
        logger.info(f"\nComplexity Score Stats:")
        logger.info(f"  Average: {avg_complexity_score:.2f}")
        logger.info(f"  Min: {min_complexity_score}")
        logger.info(f"  Max: {max_complexity_score}")
    else:
        logger.info("\nNo complexity scores found in the processed dataset for summary.")

    # The detailed heuristic component scores are no longer directly in the final example,
    # but within the reasoning string. If detailed stats on components are needed,
    # `process_example` would need to return them or they'd need to be parsed from reasoning.
    # For now, this summary is based on the final 'complexity_score'.

if __name__ == "__main__":
    main()
