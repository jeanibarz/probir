import yaml
import subprocess
import logging
import os
import sys # Added for sys.exit
import argparse
import shlex
import json # Added for checkpointing
import uuid # Added for trace_id generation
from typing import Optional, List, Dict, Any, Type # Added List, Dict, Any, Type
from pydantic import ValidationError, BaseModel # Added for Pydantic config validation
from common_utils import (
    setup_logging,
    load_jsonl_dataset,
    save_jsonl_dataset,
    validate_dataset,
    ensure_dir_exists,
    Dataset,
    PipelineConfig,
    StepConfig,
    # Import new per-step Pydantic models
    BasePipelineInput,
    SessionIdentificationOutput,
    RegexAnonymizationOutput,
    LlmAnonymizationOutput,
    ComplexityScoringOutput,
    CorrectionAnalysisOutput,
    Message, # Message is used by the models
    # Import the Features object for step1c output
    llm_anonymization_output_features
)

# Logger will be configured by setup_logging in main
# Using "probir_pipeline" as the logger name to be consistent with setup_logging default
logger = logging.getLogger("probir_pipeline")

# --- Checkpoint Configuration ---
CHECKPOINT_DIR = "logs"
CHECKPOINT_FILE_NAME = "pipeline_checkpoint.json"

def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Loads checkpoint data from the given path."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data_dict = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data_dict # Return dict directly
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
    and returns a list of actual StepConfig objects to run.
    `all_step_configs` is a list of StepConfig objects.
    """
    if not steps_arg:
        return None # Indicates all enabled steps should run

    selected_steps_to_execute: List[StepConfig] = []
    step_identifiers = [s.strip() for s in steps_arg.split(',')]

    # Create maps using StepConfig attributes
    step_map_by_name = {step_conf.name: (idx, step_conf) for idx, step_conf in enumerate(all_step_configs)}
    step_map_by_index = {str(idx + 1): (idx, step_conf) for idx, step_conf in enumerate(all_step_configs)}

    parsed_identifier_tuples: List[Tuple[int, StepConfig]] = []

    for identifier in step_identifiers:
        found_step_tuple: Optional[Tuple[int, StepConfig]] = None
        if identifier in step_map_by_name:
            found_step_tuple = step_map_by_name[identifier]
        elif identifier in step_map_by_index:
            found_step_tuple = step_map_by_index[identifier]
        else:
            logger.warning(f"Step identifier '{identifier}' not found in pipeline.yaml. Ignoring.")
            continue
        
        if found_step_tuple not in parsed_identifier_tuples:
             parsed_identifier_tuples.append(found_step_tuple)

    # Sort by original index in the pipeline.yaml
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
            raw_config = yaml.safe_load(f)
        pipeline_config = PipelineConfig.model_validate(raw_config)
    except FileNotFoundError:
        logger.error(f"Pipeline configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Error validating pipeline configuration from {config_path}: {e}")
        sys.exit(1)

    logger.info(f"Starting pipeline: {pipeline_config.pipeline_name}")
    if force_rerun:
        logger.info("Force rerun enabled. Checkpoint will be ignored if 'resume' is also active.")
        clear_checkpoint(checkpoint_file_path) # Clear checkpoint on force rerun

    if not pipeline_config.default_base_input: # Should be caught by Pydantic if required
        logger.error("`default_base_input` not defined in pipeline configuration.")
        sys.exit(1)
    
    # Initial determination of steps user wants to run (or all enabled if not specified)
    # all_defined_steps is now pipeline_config.steps (List[StepConfig])
    steps_user_intends_to_run_configs = parse_steps_to_run_arg(steps_to_run_arg, pipeline_config.steps)
    if steps_user_intends_to_run_configs is None:
        steps_user_intends_to_run_configs = [s for s in pipeline_config.steps if s.enabled]
    elif not steps_user_intends_to_run_configs: # Empty list after parsing
        logger.info("No valid steps selected by --steps-to-run argument. Exiting.")
        sys.exit(0) # Normal exit if no valid steps selected by user.

    # --- Define mapping from step script to its output Pydantic model ---
    STEP_SCRIPT_TO_OUTPUT_MODEL: Dict[str, Type[BaseModel]] = {
        "src/step2b_identify_sessions.py": SessionIdentificationOutput,
        "src/step1_anonymize_data.py": RegexAnonymizationOutput,
        "src/step1b_anonymize_llm.py": LlmAnonymizationOutput, # Keep for old pipeline.yaml compatibility if needed
        "src/step1c_session_aware_llm_anonymize.py": LlmAnonymizationOutput, # New script uses same output model
        "src/step2_score_complexity.py": ComplexityScoringOutput,
        "src/step3_analyze_correction_patterns.py": CorrectionAnalysisOutput,
        "tests/test_helpers/dummy_step.py": BasePipelineInput, # Added for test helper script
    }

    # --- Define mapping from step script to its DatasetFeatures object ---
    # This allows explicit schema definition when loading step outputs for validation.
    STEP_SCRIPT_TO_FEATURES: Dict[str, Any] = { # Using Any for DatasetFeatures type from datasets
        "src/step1c_session_aware_llm_anonymize.py": llm_anonymization_output_features,
        # Add other steps here if they have predefined Features objects
    }

    # --- Checkpoint and Resume Logic ---
    last_completed_step_name_from_checkpoint = None
    output_from_last_completed_checkpointed_step = None
    
    if resume and not force_rerun:
        checkpoint_dict = load_checkpoint(checkpoint_file_path) # Returns a dict or None
        if checkpoint_dict and checkpoint_dict.get("pipeline_config_path") == os.path.abspath(config_path):
            last_completed_step_name_from_checkpoint = checkpoint_dict.get("last_completed_step_name")
            output_from_last_completed_checkpointed_step = checkpoint_dict.get("last_output_file")
            
            if last_completed_step_name_from_checkpoint and output_from_last_completed_checkpointed_step:
                logger.info(f"Attempting to resume pipeline after step: '{last_completed_step_name_from_checkpoint}'.")
                logger.info(f"Output from last completed step was: {output_from_last_completed_checkpointed_step}")

                if steps_to_run_arg is None: # If no specific steps were requested via CLI
                    try:
                        last_completed_idx_in_all_steps = -1
                        for idx, step_conf_obj in enumerate(pipeline_config.steps):
                            if step_conf_obj.name == last_completed_step_name_from_checkpoint:
                                last_completed_idx_in_all_steps = idx
                                break
                        
                        if last_completed_idx_in_all_steps != -1:
                            temp_steps_to_run: List[StepConfig] = []
                            for idx, step_conf_obj in enumerate(pipeline_config.steps):
                                # Check if this step_conf_obj is in the list of steps the user intends to run
                                # (which at this point, if steps_to_run_arg was None, means all enabled steps)
                                if idx > last_completed_idx_in_all_steps and step_conf_obj in steps_user_intends_to_run_configs:
                                    temp_steps_to_run.append(step_conf_obj)
                            steps_user_intends_to_run_configs = temp_steps_to_run
                            
                            if not steps_user_intends_to_run_configs:
                                logger.info("Pipeline was previously completed or no further enabled steps after resume point. Exiting.")
                                sys.exit(0)
                            else:
                                resumed_step_names = [s.name for s in steps_user_intends_to_run_configs]
                                logger.info(f"Resuming. Steps to execute: {', '.join(resumed_step_names)}")
                        else: # Last completed step from checkpoint not found in current config
                            logger.warning(f"Last completed step '{last_completed_step_name_from_checkpoint}' from checkpoint not found in current pipeline.yaml. Running from beginning of specified/enabled steps.")
                            output_from_last_completed_checkpointed_step = None 
                    except ValueError: # Should not happen if names are unique and present
                        logger.warning(f"Error finding last completed step '{last_completed_step_name_from_checkpoint}'. Running from beginning.")
                        output_from_last_completed_checkpointed_step = None
                else: # --steps-to-run was provided
                    logger.info(f"Resuming with specific steps defined by --steps-to-run. Checkpoint output '{output_from_last_completed_checkpointed_step}' will be used if first targeted step needs {{prev_output}}.")
            else: # Checkpoint data incomplete
                logger.info("Checkpoint data incomplete. Starting pipeline from beginning of specified/enabled steps.")
                output_from_last_completed_checkpointed_step = None
        elif checkpoint_dict: # Checkpoint exists but for different pipeline config
            logger.warning(f"Checkpoint found at {checkpoint_file_path} is for a different pipeline configuration ('{checkpoint_dict.get('pipeline_config_path')}' vs '{os.path.abspath(config_path)}'). Ignoring checkpoint.")
            output_from_last_completed_checkpointed_step = None
    
    if steps_to_run_arg and not resume:
        selected_names = [s.name for s in steps_user_intends_to_run_configs]
        logger.info(f"Targeting specific steps: {', '.join(selected_names)}")

    defined_outputs_map = {}
    for i, step_conf_obj in enumerate(pipeline_config.steps):
        # step_conf_obj is a StepConfig instance
        defined_outputs_map[step_conf_obj.name] = step_conf_obj.outputs.main
        defined_outputs_map[f"__index__{i}"] = step_conf_obj.outputs.main # For index-based fallback

    last_dynamically_generated_output = output_from_last_completed_checkpointed_step
    accumulated_validation_results: Dict[str, Dict[str, int]] = {}

    # Iterate through all steps defined in pipeline.yaml to maintain order and context
    for current_step_idx_in_yaml, step_config_from_all_steps_obj in enumerate(pipeline_config.steps):
        # step_config_from_all_steps_obj is a StepConfig instance
        step_name = step_config_from_all_steps_obj.name

        if step_config_from_all_steps_obj not in steps_user_intends_to_run_configs:
            logger.debug(f"Step '{step_name}' is not in the execution list for this run. Skipping actual execution.")
            continue

        # --- From here, we are processing a step that IS in steps_user_intends_to_run_configs ---
        # Access attributes directly from the StepConfig object
        script = step_config_from_all_steps_obj.script
        inputs_conf = step_config_from_all_steps_obj.inputs # StepInputConfig object
        outputs_conf = step_config_from_all_steps_obj.outputs # StepOutputConfig object
        additional_args = step_config_from_all_steps_obj.args
        description = step_config_from_all_steps_obj.description

        logger.info(f"--- Evaluating Step for Execution: {step_name} ---")
        if description:
            logger.info(f"Description: {description}")

        # Script presence is validated by Pydantic if StepConfig.script is not Optional
        
        step_input_source_tag = inputs_conf.main
        actual_input_file = ""

        if step_input_source_tag == "{base}":
            actual_input_file = pipeline_config.default_base_input
            # --- Add trace_id to the base dataset before the first actual processing step ---
            # This should only happen ONCE, when the input is truly the base input.
            # And only if this is the first step in the *overall intended sequence* for this run.
            # A simple check: if this is the first step in steps_user_intends_to_run_configs
            # AND its input is {base}.
            if step_config_from_all_steps_obj == steps_user_intends_to_run_configs[0]:
                logger.info(f"Preparing base input: {actual_input_file} by adding trace_id.")
                try:
                    base_dataset = load_jsonl_dataset(actual_input_file)
                    if len(base_dataset) > 0: # Only map if not empty
                        def add_trace_id_and_version(example: Dict[str, Any]) -> Dict[str, Any]:
                            example_copy = example.copy() # Avoid modifying original dict in dataset
                            example_copy['trace_id'] = str(uuid.uuid4())
                            example_copy['schema_version'] = "1.0" # Add schema version
                            return example_copy
                        
                        base_dataset_with_ids = base_dataset.map(add_trace_id_and_version)
                        
                        # Overwrite the original base input file with the version containing trace_ids
                        # Or, save to a temporary file and use that as actual_input_file.
                        # For simplicity and to ensure subsequent direct uses of {base} get IDs, let's overwrite.
                        # This assumes default_base_input is in a writable location (e.g., data/).
                        # A safer approach might be to save to a new temp file.
                        # For now, let's assume it's fine to modify the "base" if it's the true start.
                        # However, this could be problematic if the original file is precious.
                        # Let's save to a new intermediate file for the first step.
                        
                        # Create a new path for the input with trace_ids
                        base_input_dir = os.path.dirname(actual_input_file)
                        base_input_filename = os.path.basename(actual_input_file)
                        name, ext = os.path.splitext(base_input_filename)
                        input_with_ids_path = os.path.join(base_input_dir, f"{name}_with_trace_ids{ext}")
                        
                        save_jsonl_dataset(base_dataset_with_ids, input_with_ids_path)
                        logger.info(f"Saved base input with trace_ids to: {input_with_ids_path}")
                        actual_input_file = input_with_ids_path # Use this new file as input for the first step
                    else:
                        logger.info(f"Base dataset {actual_input_file} is empty. No trace_ids added.")
                except Exception as e_traceid:
                    logger.error(f"Failed to add trace_id to base input {actual_input_file}: {e_traceid}", exc_info=True)
                    sys.exit(1)
            # --- End trace_id addition ---
        elif step_input_source_tag == "{prev_output}":
            if last_dynamically_generated_output:
                actual_input_file = last_dynamically_generated_output
            elif current_step_idx_in_yaml > 0:
                preceding_step_in_yaml_config_obj = pipeline_config.steps[current_step_idx_in_yaml - 1]
                preceding_step_name_in_yaml = preceding_step_in_yaml_config_obj.name
                
                conceptual_prev_output_file = defined_outputs_map.get(preceding_step_name_in_yaml)
                # Index fallback should ideally not be needed if names are unique and map is built correctly
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
            actual_input_file = step_input_source_tag # Direct file path
        
        if not actual_input_file: # Should be caught by Pydantic if StepInputConfig.main is not Optional
             logger.error(f"Input file for step '{step_name}' could not be determined. Skipping.")
             logger.info(f"--- Finished Step: {step_name} (Error) ---\n")
             continue

        if not os.path.exists(actual_input_file):
            logger.error(f"Input file '{actual_input_file}' for step '{step_name}' does not exist. Halting pipeline.")
            logger.info(f"--- Finished Step: {step_name} (Error - Missing Input) ---\n")
            sys.exit(1)
            
        actual_output_file = outputs_conf.main # outputs_conf is StepOutputConfig

        command = ["python", script]
        command.extend(["--input_file", actual_input_file])
        command.extend(["--output_file", actual_output_file])
        
        safe_step_name_for_filename = step_name.lower().replace(' ', '_').replace('.', '').replace('/', '_')
        step_log_file_name = f"step_{safe_step_name_for_filename}.log"
        
        # Check if --log_file_name is already in additional_args
        # Note: Pydantic model for StepConfig has args as List[Any].
        # For robust checking, convert to string if necessary or ensure args are structured.
        # Assuming args are simple strings or numbers for now.
        has_log_file_arg = False
        for i in range(len(additional_args)):
            if str(additional_args[i]) == "--log_file_name":
                has_log_file_arg = True
                break
        
        if not has_log_file_arg:
            command.extend(["--log_file_name", step_log_file_name])
        
        command.extend([str(arg) for arg in additional_args]) # Ensure all args are strings for subprocess

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
                    # Determine the correct Pydantic model for this step's output
                    # Dynamically determine project root (assuming this script is in src/)
                    PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

                    script_lookup_key = script # script path from pipeline.yaml
                    if os.path.isabs(script_lookup_key):
                        # If absolute path is within our project, make it relative to project root
                        if script_lookup_key.startswith(PROJECT_ROOT_PATH + os.sep):
                            script_lookup_key = os.path.relpath(script_lookup_key, PROJECT_ROOT_PATH)
                        # Else, it's an absolute path outside the project, use as-is for lookup.
                        # (The map would need to contain this absolute path if such a script is used)
                    
                    output_model_for_step = STEP_SCRIPT_TO_OUTPUT_MODEL.get(script_lookup_key)
                    step_specific_features = STEP_SCRIPT_TO_FEATURES.get(script_lookup_key)

                    if step_specific_features:
                        logger.info(f"Using explicit features for loading output of step '{step_name}'.")
                    else:
                        logger.info(f"No explicit features defined for step '{step_name}', will use schema inference for loading.")
                    
                    # Load the dataset produced by the step, using explicit features if available
                    produced_dataset = load_jsonl_dataset(actual_output_file, features=step_specific_features)
                    
                    if not output_model_for_step:
                        logger.error(f"No output Pydantic model defined for script '{script}' (lookup key: '{script_lookup_key}') in STEP_SCRIPT_TO_OUTPUT_MODEL map. Cannot validate step '{step_name}'.")
                        sys.exit(1) # Critical configuration error

                    logger.info(f"Validating output of step '{step_name}' using model {output_model_for_step.__name__}")
                    valid_examples, invalid_examples_with_errors = validate_dataset(
                        produced_dataset,
                        output_model_for_step, # Use the specific model for this step
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
                    sys.exit(1)
            else:
                logger.warning(f"Output file {actual_output_file} for step '{step_name}' not found after execution. Cannot validate.")
                logger.error(f"Pipeline execution halted because output file for step '{step_name}' was not created.")
                sys.exit(1)

        except subprocess.CalledProcessError as e:
            logger.error(f"Step '{step_name}' failed with return code {e.returncode}.")
            if e.stdout: logger.error(f"STDOUT:\n{e.stdout}")
            if e.stderr: logger.error(f"STDERR:\n{e.stderr}")
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to step failure.")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Script '{script}' not found for step '{step_name}'. Make sure it's in the correct path or Python's PATH.")
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to missing script.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while running step '{step_name}': {e}", exc_info=True)
            logger.info(f"--- Finished Step: {step_name} (Failed) ---\n")
            logger.error("Pipeline execution halted due to unexpected error.")
            sys.exit(1)

        logger.info(f"--- Finished Step: {step_name} --- \n")

    logger.info(f"Pipeline '{pipeline_config.pipeline_name}' finished all targeted steps successfully.")
    # Optionally clear checkpoint on full successful run of *all originally intended steps*
    # For now, let the last successful step's checkpoint persist.
    # If all steps_user_intends_to_run_configs completed, and this list wasn't shortened by resume,
    # it implies a full run of what was asked.
    # A more robust "all complete" might involve checking if the last executed step
    # is also the last step in pipeline_config.steps (among enabled ones).
    # For now, this is sufficient.

    # --- Generate Final Report ---
    if accumulated_validation_results:
        logger.info("--- Pipeline Validation Summary ---")
        total_validated_across_all_steps = 0
        total_invalid_across_all_steps = 0
        report_lines = ["# Pipeline Execution Report", f"## Pipeline: {pipeline_config.pipeline_name}", "## Validation Summary"]

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
