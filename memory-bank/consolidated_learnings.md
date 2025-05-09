# Cline Consolidated Learnings

This document contains curated, summarized, and actionable insights derived from raw reflection logs. Its purpose is to serve as a refined knowledge base for long-term use, improving efficiency and reliability.

## General Python & SQLite

**Pattern: SQLite Read/Write Cursor Management**
- When reading from and writing to an SQLite database within a loop in Python, use separate database cursors for read (`SELECT`/`fetchone`) and write (`INSERT`/`UPDATE`/`DELETE`) operations.
- *Rationale:* Prevents interference where write operations on a cursor can prematurely terminate iteration over a result set being fetched by the same cursor.

**Pattern: Python Type Hint Imports**
- When using type hints from the `typing` module (e.g., `List`, `Tuple`, `Dict`, `Optional`, `Any`), ensure they are explicitly imported: `from typing import List, Tuple, Dict, Optional, Any`.
- *Rationale:* Failure to import these types will result in a `NameError` at runtime (or potentially static analysis time) when the type hint is evaluated. This is a common oversight when adding or modifying type annotations.

## Project `probir` Specifics

**Project Structure Overview:**
- `start_proxy.sh`: Deploys `probir.py` and runs `mitmdump` (typically as `mitmproxyuser`).
- `probir.py`: A `mitmproxy` addon that captures HTTP/S traffic (filtered by `TARGET_DOMAINS`) to an SQLite database (`filtered_traffic_log.db`).
- `extract_valid_traces.py`: Processes `filtered_traffic_log.db` to extract LLM traces (e.g., from `generativelanguage.googleapis.com`) into the `valid_llm_traces` table within the same database. Handles JSON and Server-Sent Events (SSE).
- `inspect_db.py`: Utility script to inspect schemas and contents of `http_traffic` and `valid_llm_traces` tables in `filtered_traffic_log.db`.
- Data Pipeline Scripts (`stepX_*.py`): A series of scripts for processing the dataset, including anonymization, scoring, session identification, and analysis. These scripts generally read from an input JSONL file and write to an output JSONL file.
- `common_utils.py`: A shared utility module for common operations like argument parsing, logging, dataset I/O, and text chunking, used by the pipeline scripts.

**Python Project Setup:**
- Standardize on using `pyproject.toml` for defining project metadata and dependencies.
- Use `uv pip install .` for installing and managing these dependencies.

## SFT Data Preparation & Hugging Face

**SFT Data Formatting (Conversational):**
- For flexibility with Hugging Face chat templates and the `trl` library, structure conversational data with:
    1.  A `"messages"` field: A list of message objects, where each object has `role` (e.g., "system", "user", "assistant") and `content` (the text of the turn). This list represents the input context.
    2.  A `"completion"` field: A separate string containing the final assistant response that the model should learn to generate.
- Each assistant turn in a conversation should be treated as a target completion, with the entire preceding conversation history (including all prior user and assistant turns, and any system prompt) serving as the prompt/input.

**Handling Google Generative AI API JSON for SFT:**
- When extracting conversational turns from Google Generative AI API responses (often found in a `request_full_json` field):
    - Look for the `contents` array. Each element typically represents a turn with `role` and `parts` (where `parts[0]['text']` contains the message content).
    - Map the API's 'model' role to 'assistant' for consistency with SFT standards.

**Hugging Face Hub Workflow:**
- **Dependency:** Add `huggingface-hub` to `pyproject.toml`.
- **Authentication:** Use `huggingface-cli login`. Ensure the token has "Write access to contents/settings of all repos under your personal namespace" permission if you need to create or push to private repositories.
- **Loading Local JSONL:** Use `datasets.load_dataset("json", data_files="<filename>.jsonl", split="train")`.
- **Pushing to Hub:** Use `dataset.push_to_hub("username/dataset_name", private=True_or_False)`. Set `private=True` for initial uploads or sensitive data.

**Dataset Processing Script Structure (General):**
- Use `argparse` (preferably via shared utilities like `create_default_arg_parser` from `common_utils.py`) for command-line arguments (e.g., input/output files, processing limits).
- Employ `datasets.load_dataset` (or a utility wrapper like `load_jsonl_dataset`) for loading data.
- Utilize `dataset.map()` for efficient per-example processing where applicable.
- Use `dataset.to_json()` (or a utility wrapper like `save_jsonl_dataset`) for saving processed datasets.
- Clearly define processing functions for single examples.
- Include summary statistics or logs about the processing performed.

## LLM Interaction (Ollama)

**LLM Processing with Chunking:**
- For analyzing long texts with an LLM:
    1.  Split the text into manageable, overlapping chunks (e.g., using `common_utils.chunk_text`).
    2.  Process these chunks in parallel using `concurrent.futures.ThreadPoolExecutor`.
    3.  Aggregate results from all chunks.
    4.  Deduplicate findings, especially if overlapping chunks might detect the same PII or entities.
- Make chunk size, overlap, and the number of parallel workers configurable (e.g., via command-line arguments).

**Ollama Resource Management:**
- Be mindful of the Ollama server's memory capacity when setting the number of parallel workers (`max_workers`) for chunk processing. Too many concurrent requests can lead to out-of-memory errors, especially with larger models.
- Start with a conservative number of workers (e.g., 4-10) and monitor performance and stability before increasing. The optimal number depends on the model, server hardware, and content complexity.

**Debugging Ollama (`format="json"`):**
- The structure of messages sent to the Ollama API (e.g., a single system message vs. a system message followed by a user message) can significantly affect the reliability of JSON-formatted output, especially when using `format="json"`.
- If encountering cryptic errors or invalid JSON responses, try adjusting the message structure (e.g., ensuring there's a user message in addition to the system prompt).
- Implement robust error logging around `ollama.Client.chat()` calls to distinguish API interaction errors from JSON parsing or Pydantic validation errors.

## Debugging & Workflow Techniques

**Debugging Loops:**
- If loop counters, progress bars (like `tqdm`), or script output seem inconsistent with the expected number of iterations, suspect premature loop exit.
- Common causes include unhandled exceptions within the loop, `break` or `return` statements, or indentation errors placing logic outside the intended loop scope.
- To isolate the issue, simplify the loop body to its bare minimum (e.g., just reading and printing/counting) and gradually reintroduce logic.

**Strategy for Large Log Files:**
- If `read_file` fails or is impractical for very large log files, use `search_files` with specific regex patterns to check for expected (or unexpected) log entries as a verification or diagnostic step.

**Iterative Testing & Development:**
- When introducing parallelism, resource-intensive operations, or complex logic, test with small data subsets first.
- Gradually increase the load, number of workers, or data complexity while monitoring for errors, performance, and resource usage.

**Task Interpretation & Adaptation:**
- Be prepared to adjust the interpretation of a task's sub-goals or initial assumptions if intermediate results suggest a different understanding or approach is needed. For example, a task initially framed as "fix failing logic" might evolve into "verify corrected logic" if an early fix appears successful.

**Task Management for Multi-Step Processes:**
- Using a dedicated file (e.g., `task_progress.md`) to document a multi-phase plan, methodology, and track progress is effective, especially for complex or evolving tasks. This file can be updated as phases are completed or plans change.

## Refactoring & Code Structure

**Principle: Shared Utilities for Pipelines**
- For multi-script data processing pipelines, identify common operations (e.g., data I/O, argument parsing, external service interaction, logging, text processing utilities like chunking).
- Centralize these common operations into a shared utility module (e.g., `common_utils.py`).
- *Rationale:* Improves maintainability, reduces code redundancy, ensures consistency, and makes individual pipeline scripts cleaner and more focused on their specific logic.

**Refactoring Strategy (Example with `common_utils.py`):**
1.  Create the shared utility file (e.g., `common_utils.py`).
2.  Implement utility functions for identified common operations:
    *   Standard argument parsing (base and specialized, e.g., for LLM tasks).
    *   Dataset loading and saving (e.g., for JSONL files using the `datasets` library).
    *   Standardized logging setup.
    *   Wrappers for external service clients (e.g., Ollama) and robust interaction patterns (e.g., chat calls with Pydantic validation).
    *   General text processing functions (e.g., `chunk_text`).
3.  Update individual pipeline scripts to import and use these utilities, removing the duplicated local implementations.

**Parameter Propagation During Refactoring:**
- When refactoring global constants or hardcoded values into configurable parameters (e.g., via `argparse`), ensure these new parameters are correctly passed down through the function call stack to all locations where they are needed.

**Iterative Refactoring:**
- For substantial refactoring tasks, an iterative approach is generally safer and more manageable than attempting to change everything at once.
- Process: Read the file, apply a small, targeted set of changes (e.g., using `replace_in_file` with concise blocks), verify the outcome, and then repeat for the next set of changes.

**Pattern: Refactoring File/Directory Structures**
- When reorganizing files or directories within a project:
    1.  **Plan:** Identify files/directories to move and their new locations.
    2.  **Create Structure:** Create new directories if they don't exist (e.g., `mkdir -p new_project_dir/data`).
    3.  **Move:** Relocate the files/directories (e.g., `mv old_path/file.txt new_project_dir/data/file.txt`).
    4.  **Update References:** Systematically find and update all references to the old paths in code, configuration files (e.g., YAML, JSON), documentation, and build scripts.
        *   Use tools like `grep` or IDE search functions to find references.
        *   Use `replace_in_file` for targeted changes in text-based configuration files.
    5.  **Test:** Thoroughly test the application or pipeline to ensure all parts function correctly with the new paths. Check for file not found errors or incorrect behavior.
    6.  **Version Control:** If using Git, use `git mv` for tracked files to preserve history, or `git add`/`git rm` after a standard `mv`. Commit the changes with a clear message.
- *Rationale:* Ensures a clean project structure, separates concerns (e.g., code from data), and maintains project functionality after reorganization.

**Pattern: Using a `src/` Layout for Python Projects**
- **Structure:** Place all main Python source code (modules and packages) into a top-level directory named `src/`. Other files like `pyproject.toml`, `README.md`, `data/`, `tests/`, `docs/` remain in the project root.
- **Execution:** Scripts within `src/` are typically run from the project root (e.g., `python src/my_app/main.py`). Python's import system usually handles imports within `src/` correctly in this scenario (e.g., `from . import utils` or `from my_app import utils` if `src/my_app/` is a package).
- **Build Configuration (`pyproject.toml` for setuptools):** To make the project installable and ensure build tools find the package(s) inside `src/`:
    ```toml
    [tool.setuptools.packages.find]
    where = ["src"]  # Tells setuptools to look for packages in src/
    # namespaces = false # Often true by default, can be set explicitly
    ```
    Or, for older `setup.py` or `setup.cfg` based projects, `package_dir = {'': 'src'}` would be used.
- *Rationale:*
    - Cleaner project root.
    - Prevents accidental imports of the package from the current working directory if it's not installed, which can hide packaging issues.
    - Common convention in many Python projects.

## Tool Usage (`cline`)

**`replace_in_file` Strategy:**
- For complex file modifications or when making multiple changes, it's more reliable to break down `replace_in_file` operations into several smaller, targeted SEARCH/REPLACE blocks rather than using a single, very large block.
- *Rationale:* Smaller blocks are less prone to matching errors caused by subtle discrepancies or incorrect ordering of changes within the SEARCH block. This makes the refactoring process more robust.

**Log File Handling with `write_to_file`:**
- When intending to append to a log file (like `raw_reflection_log.md`), remember that `write_to_file` overwrites the target file.
- Correct procedure for appending:
    1. Use `read_file` to get the existing content of the log file.
    2. Construct the new log entry.
    3. Combine the existing content and the new entry.
    4. Use `write_to_file` with the *full combined content* to update the log file.

## Testing & Pytest

**Pattern: Pytest Setup for `src/` Layout**
- To ensure `pytest` can correctly discover and import modules from a `src/` project layout:
    1.  Ensure `pytest` is a project dependency (e.g., in `pyproject.toml`).
    2.  Install the project in editable mode: `uv pip install -e .` (or `pip install -e .`). This makes the `src` package available in the environment.
    3.  Add `pythonpath = ["src"]` under `[tool.pytest.ini_options]` in `pyproject.toml`. This explicitly tells `pytest` to add the `src` directory to its Python path.
    4.  Run `pytest` using `python -m pytest`. This ensures `pytest` runs within the context of the project's Python environment, where all dependencies are correctly installed and paths are set up.
- *Rationale:* Addresses common `ModuleNotFoundError` issues when testing projects with a `src/` layout by ensuring `pytest` can find the source modules and their dependencies.

**Pattern: Testing `logging.FileHandler` Mocking**
- When mocking `logging.FileHandler` (or its methods like `setLevel`) for tests:
    - The `logging` framework internally checks `record.levelno >= handler.level`. The `handler.level` attribute must be an integer.
    - If `FileHandler.setLevel` is mocked, ensure the mock handler instance's `level` attribute is also set to an integer value. This can be done using a `side_effect` on the `setLevel` mock that updates `mock_instance.level`.
    - *Example `side_effect` for a mock `FileHandler` instance:*
      ```python
      def set_level_side_effect(level_value):
          mock_handler_instance.level = level_value # mock_handler_instance is the mocked FileHandler
          # Call original setLevel if needed, or just set the attribute
      mock_handler_instance.setLevel.side_effect = set_level_side_effect
      ```
- *Rationale:* Prevents `TypeError` during logging operations in tests due to incompatible types for handler levels.

**Pattern: Robust Log Assertions**
- When asserting log messages, especially those containing complex data structures (e.g., Pydantic error dicts, sets, or long strings):
    - Exact string matching can be brittle due to minor formatting differences or non-deterministic order of elements (e.g., in sets or dictionary string representations).
    - Prefer checking for the presence of key substrings within the logged message.
    - If logged data is structured (e.g., JSON within the log), consider deserializing it and asserting against the structure or specific values.
    - Use `caplog.text` for a single string of all logs, or iterate `caplog.records` (or `mock_logger.method.call_args_list`) for more granular checks.
- *Rationale:* Makes tests less prone to failures from trivial log message variations, focusing on the essential content.

**Pattern: Testing File I/O Across Processes (Synchronization)**
- When a test involves a parent process checking a file created/modified by a subprocess (e.g., in `pytest` using `tmp_path`):
    - Be aware of potential file system synchronization or visibility delays. `os.path.exists()` in the parent might return `False` immediately after the subprocess writes the file.
    - If direct file access from the parent is unreliable, consider:
        - Introducing short, careful delays (use with caution, can make tests flaky).
        - Having the subprocess log a confirmation of file write completion.
        - Designing the test to verify the outcome through other means (e.g., subprocess exit code, other reported metrics, or having the subprocess perform a read-back verification itself if possible).
- *Rationale:* Addresses flakiness in tests that depend on immediate cross-process file visibility.

**Pattern: Testing Orchestrators with Dummy Steps**
- For testing pipeline orchestrators that execute multiple steps (often as subprocesses):
    1.  Create a `pytest` fixture to set up a temporary test environment (e.g., using `tmp_path`). This includes directories for inputs, outputs, logs, and checkpoints.
    2.  Develop simple "dummy" step scripts. These scripts should be configurable (e.g., via CLI args or environment variables) to simulate various outcomes: success, failure (non-zero exit code), specific data transformations, producing valid/invalid data, and specific logging.
    3.  Dynamically generate test-specific pipeline configuration files (e.g., `pipeline.yaml`) within the fixture. These configs should point to the temporary data paths and the dummy step scripts. Use `Path(path_str).as_posix()` for platform-independent paths in generated files.
    4.  The orchestrator tests then invoke the orchestrator script, targeting these temporary configurations and dummy steps.
- *Rationale:* Allows isolated, repeatable, and comprehensive testing of orchestrator logic (sequencing, error handling, checkpointing, reporting) without the overhead or flakiness of running real, complex steps.

## Git Workflow

**Pattern: `git mv` to an Ignored Directory**
- If you use `git mv old_path new_path` and `new_path` (or its containing directory) is covered by `.gitignore`:
    - The file at `new_path` will become untracked and ignored by Git.
    - The commit will reflect the *deletion* of `old_path`. The file at `new_path` will not be part of the commit's tracked tree.
    - `git status` will show `old_path` as deleted and `new_path` as untracked (if it exists and is ignored).
- *Rationale:* Understanding this behavior is crucial for correctly managing files moved into ignored locations (e.g., build artifacts, local logs, temporary data).

**Pattern: Correcting Commits - Removing Files**
- To remove a file that was accidentally committed or to stop tracking a file that should now be ignored:
    1.  `git rm --cached <file_to_remove_from_tracking>`: This unstages the file and removes it from Git's index, but leaves the physical file in your working directory.
    2.  Ensure the file is listed in `.gitignore` if it should be ignored going forward.
    3.  `git commit --amend --no-edit`: This updates the previous commit, removing the file from its history, without changing the commit message. If you need to change the message, omit `--no-edit`.
- *Rationale:* Provides a clean way to correct the history of the last commit.

## Project `probir` Specifics - Testing & Data Structures

**`step1b_anonymize_llm.py` - Output Structure & Placeholders:**
- The script `src/step1b_anonymize_llm.py` produces an `llm_anonymization_details` dictionary with the following structure:
  ```python
  {
      "llm_sensitive_categories_found": sorted(list(overall_llm_pii_categories)),
      "llm_detected_pii_items_messages": list_of_pii_item_dicts_for_messages,
      "llm_detected_pii_items_completion": list_of_pii_item_dicts_for_completion
  }
  ```
  - Each `pii_item_dict` in the lists is `{"category": "...", "value": "..."}`.
- Tests asserting this structure must use these exact keys (e.g., `llm_detected_pii_items_messages`) and access PII item details using `pii_item["category"]`.
- The anonymization placeholder format used by this script is `[CATEGORY_REDACTED]`. Test data and assertions for anonymized strings must match this format.
- *Rationale:* Ensures tests accurately reflect the script's output, preventing false negatives/positives.
