from datasets import load_dataset
import argparse
import logging
from common_utils import setup_logging # Assuming setup_logging is in common_utils

# Logger will be configured by setup_logging in main
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Push a local JSONL dataset to the Hugging Face Hub.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="sft_dataset.jsonl",
        help="Path to the local JSONL dataset file to push (default: sft_dataset.jsonl)."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="JeanIbarz/cline_dataset", # Consider making this required or prompt if not given
        help="Hugging Face Hub repository ID (e.g., YourUsername/YourDatasetName)."
    )
    parser.add_argument(
        "--private",
        action='store_true', # Default is False if not present
        help="Set the repository to private on the Hub. If not set, it defaults to public unless repo exists and is private."
    )
    parser.add_argument(
        "--public",
        action='store_true', # Mutually exclusive with private for clarity, though push_to_hub handles it
        help="Set the repository to public on the Hub. Overrides --private if both are set."
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    parser.add_argument(
        "--log_file_name",
        type=str,
        default=None,
        help="Optional: Name of the log file (e.g., 'push_hf.log'). If not provided, logs only to console."
    )
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)
    
    global logger # Re-initialize logger for this module
    logger = logging.getLogger(__name__)

    if args.log_file_name:
        logger.info(f"Logs for this script will also be saved to logs/{args.log_file_name}")

    # Determine privacy setting
    # push_to_hub default for private is False. If --private is set, make it True.
    # If --public is explicitly set, it should be public.
    is_private = args.private
    if args.public:
        is_private = False # Explicit public flag overrides private

    logger.info(f"Loading dataset from {args.input_file}...")
    try:
        dataset = load_dataset("json", data_files=args.input_file, split="train")
        logger.info(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_file}: {e}")
        return

    logger.info(f"Preparing to push dataset to Hugging Face Hub: {args.repo_id} (Private: {is_private})")

    try:
        logger.info("Pushing dataset to Hub (this may take a while)...")
        dataset.push_to_hub(args.repo_id, private=is_private)
        logger.info(f"Dataset successfully pushed to {args.repo_id} on the Hugging Face Hub.")
        logger.info(f"You can view it at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        logger.error(f"Error pushing dataset to Hub: {e}")
        logger.error("Please ensure you are logged in (`huggingface-cli login`) and have the correct permissions for the repository.")

if __name__ == "__main__":
    main()
