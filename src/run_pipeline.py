import yaml
import subprocess
import logging
import os
import argparse
import shlex
import json # Added for checkpointing
from typing import Optional, List, Dict, Any # Added List, Dict, Any
from common_utils import setup_logging, load_jsonl_dataset, save_jsonl_dataset, BaseTrace, validate_dataset, ensure_dir_exists, Dataset # Added validation imports and Dataset for saving invalid

# Logger will be configured by setup_logging in main
logger = logging.getLogger(__name__)

# --- Checkpoint Configuration ---
CHECKPOINT_DIR = "logs"
CHECKPOINT_FILE_NAME = "pipeline_checkpoint.json"

def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Loads checkpoint data from the given path."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from checkpoint file: {checkpoint_path}. Ignoring checkpoint.")
        except Exception as e:
            logger.error(f"Error loading checkpoint file {checkpoint_path}: {e}. Ignoring checkpoint.")
    else:
        logger.info(f"No checkpoint file found at {checkpoint_path}.")
    return None

def save_checkpoint(checkpoint_path: str, step_name: str, output_file: str, pipeline_config_path: str) -> None:
    """Saves pipeline progress to a checkpoint file."""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_data = {
            "last_completed_step_name": step_name,
            "last_output_file": output_file,
            "pipeline_config_path": os.path.abspath(pipeline_config_path) # Store absolute path
        }
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=4)
        logger.info(f"Saved checkpoint for step '{step_name}' to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint to {checkpoint_path}: {e}")

def clear_checkpoint(checkpoint_path: str) -> None:
    """Clears the checkpoint file."""
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint file {checkpoint_path} cleared.")
    except Exception as e:
        logger.error(f"Error clearing checkpoint file {checkpoint_path}: {e}")

def parse_steps_to_run_arg(steps_arg, all_step_configs):
    """
    Parses the --steps-to-run argument (comma-separated names or 1-based indices)
    and returns a list of actual step config objects to run.
    """
    if not steps_arg:
        return None # Indicates all enabled steps should run

    selected_steps_to_execute = []
    step_identifiers = [s.strip() for s in steps_arg.split(',')]

    step_map_by_name = {step_conf.get("name"): (idx, step_conf) for idx, step_conf in enumerate(all_step_configs)}
    step_map_by_index = {str(idx + 1): (idx, step_conf) for idx, step_conf in enumerate(all_step_configs)}

    parsed_identifier_tuples = [] 

    for identifier in step_identifiers:
        found_step_tuple = None
        if identifier in step_map_by_name:
            found_step_tuple = step_map_by_name[identifier]
        elif identifier in step_map_by_index:
            found_step_tuple = step_map_by_index[identifier]
        else:
            logger.warning(f"Step identifier '{identifier}' not found in pipeline.yaml. Ignoring.")
            continue
        
        if found_step_tuple not in parsed_identifier_tuples:
             parsed_identifier_tuples.append(found_step_tuple)

    parsed_identifier_tuples.sort(key=lambda x: x[0])
    selected_steps_to_execute = [step_conf for _, step_conf in parsed_identifier_tuples]
    
    return selected_steps_to_execute


def run_pipeline(config_path="pipeline.yaml", steps_to_run_arg=None, resume=False, force_rerun=False):
    """
    Runs the data processing pipeline defined in the YAML configuration file.
    Can run all enabled steps or a specific subset if steps_to_run_arg is provided.
    Supports resuming from a checkpoint.
    """
    checkpoint_file_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILE_NAME)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Pipeline configuration file not found: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        return

    pipeline_name = config.get("pipeline_name", "Unnamed Pipeline")
    default_base_input = config.get("default_base_input")
    all_defined_steps = config.get("steps", [])

    logger.info(f"Starting pipeline: {pipeline_name}")
    if force_rerun:
        logger.info("Force rerun enabled. Checkpoint will be ignored if 'resume' is also active.")
        clear_checkpoint(checkpoint_file_path) # Clear checkpoint on force rerun

    if not default_base_input:
        logger.error("`default_base_input` not defined in pipeline configuration.")
        return
    
    # Initial determination of steps user wants to run (or all enabled if not specified)
    steps_user_intends_to_run_configs = parse_steps_to_run_arg(steps_to_run_arg, all_defined_steps)
    if steps_user_intends_to_run_configs is None: 
        steps_user_intends_to_run_configs = [s for s in all_defined_steps if s.get("enabled", True)]
    elif not steps_user_intends_to_run_configs:
        logger.info("No valid steps selected by --steps-to-run argument. Exiting.")
        return

    # --- Checkpoint and Resume Logic ---
    last_completed_step_name_from_checkpoint = None
    output_from_last_completed_checkpointed_step = None
    
    if resume and not force_rerun:
        checkpoint = load_checkpoint(checkpoint_file_path)
        if checkpoint and checkpoint.get("pipeline_config_path") == os.path.abspath(config_path):
            last_completed_step_name_from_checkpoint = checkpoint.get("last_completed_step_name")
            output_from_last_completed_checkpointed_step = checkpoint.get("last_output_file")
            
            if last_completed_step_name_from_checkpoint and output_from_last_completed_checkpointed_step:
                logger.info(f"Attempting to resume pipeline after step: '{last_completed_step_name_from_checkpoint}'.")
                logger.info(f"Output from last completed step was: {output_from_last_completed_checkpointed_step}")

                # If no specific steps were requested via CLI, adjust steps_user_intends_to_run_configs
                if steps_to_run_arg is None:
                    try:
                        # Find the index of the last completed step in the *full list of defined steps*
                        last_completed_idx_in_all_steps = -1
                        for idx, step_conf in enumerate(all_defined_steps):
                            if step_conf.get("name") == last_completed_step_name_from_checkpoint:
                                last_completed_idx_in_all_steps = idx
                                break
                        
                        if last_completed_idx_in_all_steps != -1:
                            # Filter steps_user_intends_to_run_configs to only include those *after* the last completed one
                            # This requires knowing the original index in all_defined_steps for each config
                            # Rebuild steps_user_intends_to_run_configs based on all_defined_steps
                            temp_steps_to_run = []
                            for idx, step_conf in enumerate(all_defined_steps):
                                if idx > last_completed_idx_in_all_steps and step_conf in steps_user_intends_to_run_configs:
                                    temp_steps_to_run.append(step_conf)
                            steps_user_intends_to_run_configs = temp_steps_to_run
                            
                            if not steps_user_intends_to_run_configs:
                                logger.info("Pipeline was previously completed or no further enabled steps after resume point. Exiting.")
                                return
                            else:
                                resumed_step_names = [s.get('name', 'Unnamed') for s in steps_user_intends_to_run_configs]
                                logger.info(f"Resuming. Steps to execute: {', '.join(resumed_step_names)}")
                        else:
                            logger.warning(f"Last completed step '{last_completed_step_name_from_checkpoint}' from checkpoint not found in current pipeline.yaml. Running from beginning of specified/enabled steps.")
                            output_from_last_completed_checkpointed_step = None # Invalidate, as context is lost
                    except ValueError:
                        logger.warning(f"Could not find last completed step '{last_completed_step_name_from_checkpoint}' in current pipeline definition. Running from beginning of specified/enabled steps.")
                        output_from_last_completed_checkpointed_step = None # Invalidate
                else: # --steps-to-run was provided
                    logger.info(f"Resuming with specific steps defined by --steps-to-run. Checkpoint output '{output_from_last_completed_checkpointed_step}' will be used if first targeted step needs {{prev_output}}.")
            else:
                logger.info("Checkpoint data incomplete. Starting pipeline from beginning of specified/enabled steps.")
                output_from_last_completed_checkpointed_step = None # Invalidate
        elif checkpoint: # Checkpoint exists but for different pipeline config
            logger.warning(f"Checkpoint found at {checkpoint_file_path} is for a different pipeline configuration ('{checkpoint.get('pipeline_config_path')}' vs '{os.path.abspath(config_path)}'). Ignoring checkpoint.")
            output_from_last_completed_checkpointed_step = None # Invalidate
        # If no checkpoint or invalid, output_from_last_completed_checkpointed_step remains None

    if steps_to_run_arg and not resume: # Only log this if not resuming, to avoid redundant logs
        selected_names = [s.get('name', 'Unnamed') for s in steps_user_intends_to_run_configs]
        logger.info(f"Targeting specific steps: {', '.join(selected_names)}")
    
    # This map is used to resolve {prev_output} if a step is skipped (not run in this invocation)
    # but its output is needed by a later step that *is* run.
    defined_outputs_map = {}
    for i, step_conf in enumerate(all_defined_steps):
        step_name = step_conf.get("name", f"Defined Step {i+1}")
        outputs_conf = step_conf.get("outputs", {})
        main_output = outputs_conf.get("main")
        if main_output:
            defined_outputs_map[step_name] = main_output
            defined_outputs_map[f"__index__{i}"] = main_output

    # This tracks the output of the most recent *actually executed* step in this run.
    # Initialize with checkpointed output if we resumed and it's relevant.
    last_dynamically_generated_output = output_from_last_completed_checkpointed_step
    
    accumulated_validation_results: Dict[str, Dict[str, int]] = {} # To store validation counts per step

    for current_step_idx_in_yaml, step_config_from_all_steps in enumerate(all_defined_steps):
        step_name = step_config_from_all_steps.get("name", f"Unnamed Step (YAML index {current_step_idx_in_yaml+1})")

        # Determine if this step should actually be executed in this run
        if step_config_from_all_steps not in steps_user_intends_to_run_configs:
            # This step is not targeted for execution in this run (either disabled, not selected, or successfully completed in a resumed run).
            # However, its defined output might be needed if a *later* step uses {prev_output}
            # and this current step was its conceptual predecessor.
            # We update last_dynamically_generated_output to this step's *defined* output
            # if no dynamic output has been generated yet by a *run* step that conceptually
            # comes after this skipped one but before the one needing {prev_output}.
            # This is tricky. The current logic for {prev_output} already consults defined_outputs_map
            # if last_dynamically_generated_output is None from a *run* step.
            # For simplicity, if a step is skipped, we ensure its defined output is known for the map.
            # The crucial part is that `last_dynamically_generated_output` should only be updated by *executed* steps.
            # If we are skipping steps due to resume, `last_dynamically_generated_output` is already primed.
            # If skipping for other reasons (not in `steps_user_intends_to_run_configs`),
            # then `last_dynamically_generated_output` should persist from the last *executed* step.
            logger.debug(f"Step '{step_name}' is not in the execution list for this run. Skipping actual execution.")
            continue # Skip to the next step in all_defined_steps

        # --- From here, we are processing a step that IS in steps_user_intends_to_run_configs ---
        
        script = step_config_from_all_steps.get("script")
        inputs_conf = step_config_from_all_steps.get("inputs", {})
        outputs_conf = step_config_from_all_steps.get("outputs", {})
        additional_args = step_config_from_all_steps.get("args", [])
        description = step_config_from_all_steps.get("description", "")

        logger.info(f"--- Evaluating Step for Execution: {step_name} ---")
        if description:
            logger.info(f"Description: {description}")

        if not script:
            logger.error(f"No script defined for step '{step_name}'. Skipping.")
            logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
            continue
        
        step_input_source_tag = inputs_conf.get("main", "")
        actual_input_file = ""

        if step_input_source_tag == "{base}":
            actual_input_file = default_base_input
        elif step_input_source_tag == "{prev_output}":
            if last_dynamically_generated_output: # Output from a previously *run* step (or checkpoint)
                actual_input_file = last_dynamically_generated_output
            elif current_step_idx_in_yaml > 0: # Fallback to defined output of conceptual predecessor
                preceding_step_in_yaml_config = all_defined_steps[current_step_idx_in_yaml - 1]
                preceding_step_name_in_yaml = preceding_step_in_yaml_config.get("name", f"Unnamed Step (YAML index {current_step_idx_in_yaml})")
                
                conceptual_prev_output_file = defined_outputs_map.get(preceding_step_name_in_yaml)
                if not conceptual_prev_output_file:
                    conceptual_prev_output_file = defined_outputs_map.get(f"__index__{current_step_idx_in_yaml - 1}")

                if conceptual_prev_output_file:
                    actual_input_file = conceptual_prev_output_file
                    logger.info(f"Step '{step_name}' uses '{{prev_output}}'. No prior step run in this invocation or no dynamic output. "
                                f"Using defined output of '{preceding_step_name_in_yaml}': {actual_input_file}")
                else:
                    logger.error(f"Cannot determine conceptual previous output for '{{prev_output}}' for step '{step_name}'. Skipping.")
                    logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
                    continue
            else: 
                logger.error(f"Step '{step_name}' is the first defined step and cannot use '{{prev_output}}' if no prior dynamic output (e.g. from resume). Skipping.")
                logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
                continue
        else: 
            actual_input_file = step_input_source_tag
        
        if not actual_input_file:
             logger.error(f"Input file for step '{step_name}' could not be determined. Skipping.")
             logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
             continue

        if not os.path.exists(actual_input_file):
            logger.error(f"Input file '{actual_input_file}' for step '{step_name}' does not exist. Skipping.")
            logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
            continue
            
        step_output_target = outputs_conf.get("main")
        if not step_output_target:
            logger.error(f"No 'outputs.main' defined for step '{step_name}'. Cannot proceed. Skipping.")
            logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
            continue
        actual_output_file = step_output_target

        command = ["python", script]
        command.extend(["--input_file", actual_input_file])
        command.extend(["--output_file", actual_output_file])
        
        # Add log_file_name argument for the individual script if not already present in step_config args
        # Construct a unique log file name for the step
        step_log_file_name = f"step_{step_name.lower().replace(' ', '_').replace('.', '')}.log"
        
        has_log_file_arg = any(isinstance(arg, str) and arg == "--log_file_name" for arg in additional_args)
        if not has_log_file_arg:
            command.extend(["--log_file_name", step_log_file_name])
        
        command.extend([str(arg) for arg in additional_args])


        logger.info(f"Executing command: {' '.join(shlex.quote(str(c)) for c in command)}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"Step '{step_name}' completed successfully.")
            if result.stdout: logger.debug(f"STDOUT:\n{result.stdout}")
            if result.stderr: logger.warning(f"STDERR:\n{result.stderr}")
            
            last_dynamically_generated_output = actual_output_file # Update with the output of this successfully run step
            
            # --- Data Validation after successful step execution ---
            if os.path.exists(actual_output_file):
                logger.info(f"Validating output of step '{step_name}' from file: {actual_output_file}")
                try:
                    # Load the dataset produced by the step
                    produced_dataset = load_jsonl_dataset(actual_output_file) 
                    
                    valid_examples, invalid_examples_with_errors = validate_dataset(
                        produced_dataset, 
                        BaseTrace, # Using the comprehensive BaseTrace model
                        step_name
                    )
                    
                    accumulated_validation_results[step_name] = {
                        "valid": len(valid_examples),
                        "invalid": len(invalid_examples_with_errors)
                    }

                    if invalid_examples_with_errors:
                        invalid_output_dir = "data/invalid_examples"
                        ensure_dir_exists(invalid_output_dir) # Ensure directory exists
                        
                        # Sanitize step_name for filename
                        safe_step_name_for_file = step_name.replace(' ', '_').replace('.', '_').lower()
                        invalid_output_filename = os.path.join(invalid_output_dir, f"{safe_step_name_for_file}_invalid_examples.jsonl")
                        
                        if invalid_examples_with_errors: 
                            try:
                                invalid_data_to_save = [item['example_data'] for item in invalid_examples_with_errors if 'example_data' in item]
                                if invalid_data_to_save:
                                    invalid_dataset_to_save = Dataset.from_list(invalid_data_to_save)
                                    save_jsonl_dataset(invalid_dataset_to_save, invalid_output_filename)
                                    logger.warning(f"Saved {len(invalid_examples_with_errors)} invalid examples from step '{step_name}' to {invalid_output_filename}")
                                else:
                                    logger.warning(f"Step '{step_name}' had invalid entries, but no 'example_data' to save.")
                            except Exception as save_err:
                                logger.error(f"Could not save invalid examples for step '{step_name}' to {invalid_output_filename}: {save_err}", exc_info=True)
                        
                        logger.warning(f"Step '{step_name}' produced {len(invalid_examples_with_errors)} invalid records. See logs and {invalid_output_filename}.")

                    # Save checkpoint *after* successful execution AND validation attempt
                    save_checkpoint(checkpoint_file_path, step_name, actual_output_file, config_path)

                except Exception as val_err:
                    logger.error(f"Error during validation of step '{step_name}' output: {val_err}", exc_info=True)
                    logger.error(f"Pipeline will continue, but output of '{step_name}' may be compromised. Checkpoint NOT saved for this state if validation failed.")
                    # Halt pipeline if validation itself fails catastrophically
                    logger.error("Pipeline execution halted due to validation error.")
                    return
            else:
                logger.warning(f"Output file {actual_output_file} for step '{step_name}' not found after execution. Cannot validate.")
                logger.error(f"Pipeline execution halted because output file for step '{step_name}' was not created.")
                return

        except subprocess.CalledProcessError as e:
            logger.error(f"Step '{step_name}' failed with return code {e.returncode}.")
            if e.stdout: logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr: logger.error(f"STDERR:\n{e.stderr}")
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to step failure.")
            return 
        except FileNotFoundError:
            logger.error(f"Script '{script}' not found for step '{step_name}'. Make sure it's in the correct path or Python's PATH.")
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to missing script.")
            return
        except Exception as e:
            logger.error(f"An unexpected error occurred while running step '{step_name}': {e}", exc_info=True)
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to unexpected error.")
            return

        logger.info(f"--- Finished Step: {step_name} --- \n")

    logger.info(f"Pipeline '{pipeline_name}' finished all targeted steps successfully.")
    # Optionally clear checkpoint on full successful run of *all originally intended steps*
    # For now, let the last successful step's checkpoint persist.
    # If all steps_user_intends_to_run_configs completed, and this list wasn't shortened by resume,
    # it implies a full run of what was asked.
    # A more robust "all complete" might involve checking if the last executed step
    # is also the last step in all_defined_steps (among enabled ones).
    # For now, this is sufficient.

    # --- Generate Final Report ---
    if accumulated_validation_results:
        logger.info("--- Pipeline Validation Summary ---")
        total_validated_across_all_steps = 0
        total_invalid_across_all_steps = 0
        report_lines = ["# Pipeline Execution Report", f"## Pipeline: {pipeline_name}", "## Validation Summary"]

        for step_name_report, counts in accumulated_validation_results.items():
            msg = f"Step '{step_name_report}': {counts['valid']} valid, {counts['invalid']} invalid examples."
            logger.info(msg)
            report_lines.append(f"- **{step_name_report}**: {counts['valid']} valid, {counts['invalid']} invalid.")
            total_validated_across_all_steps += counts['valid']
            total_invalid_across_all_steps += counts['invalid']
        
        if total_invalid_across_all_steps > 0:
            warning_msg = f"Total invalid examples across all validated steps: {total_invalid_across_all_steps}. Check logs and 'data/invalid_examples/' directory."
            logger.warning(warning_msg)
            report_lines.append(f"\n**WARNING:** {warning_msg}")
        else:
            success_msg = "All examples in all validated steps were valid."
            logger.info(success_msg)
            report_lines.append(f"\n**SUCCESS:** {success_msg}")
        logger.info("--- End of Pipeline Validation Summary ---")
        
        # Save report to a Markdown file
        report_path = os.path.join(CHECKPOINT_DIR, "pipeline_execution_report.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f_report:
                f_report.write("\n".join(report_lines))
            logger.info(f"Pipeline execution report saved to {report_path}")
        except Exception as report_e:
            logger.error(f"Failed to save pipeline execution report: {report_e}")
            
    else:
        logger.info("No validation results accumulated during this pipeline run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a data processing pipeline from a YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline.yaml",
        help="Path to the pipeline YAML configuration file (default: pipeline.yaml)"
    )
    parser.add_argument(
        "--steps-to-run",
        type=str,
        default=None,
        help="Comma-separated list of step names or 1-based indices to run (e.g., \"1,3\", \"Step Name A,Step Name C\"). If not provided, runs all enabled steps."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt to resume pipeline from the last successful checkpoint."
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force pipeline to run from the beginning, ignoring any checkpoint. Clears existing checkpoint."
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
        default="run_pipeline_orchestrator.log", # Default log file for orchestrator
        help="Optional: Name of the log file for the pipeline orchestrator (e.g., 'pipeline_run.log'). If not provided, logs only to console."
    )
    args = parser.parse_args()

    # Setup logging for the orchestrator itself
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    # Use a specific log file name for the orchestrator, or None if user specified empty string
    orchestrator_log_file = args.log_file_name if args.log_file_name else None 
    setup_logging(log_level_int, orchestrator_log_file)


    if orchestrator_log_file:
        logger.info(f"Orchestrator logs will also be saved to logs/{orchestrator_log_file}")

    run_pipeline(
        config_path=args.config, 
        steps_to_run_arg=args.steps_to_run,
        resume=args.resume,
        force_rerun=args.force_rerun
    )
