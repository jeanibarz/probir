# Task Progress: Dataset Analysis and Refinement

**IMPORTANT: This document must be updated regularly to track the current task progress, especially after each major step is accomplished or if the plan changes.**

## 1. Overall Goal

To analyze and refine the `sft_dataset.jsonl` dataset to improve its quality, extract insights, and prepare it for potential Supervised Fine-tuning (SFT) or other uses. This involves multiple processing phases.

## 2. Methodology

The dataset processing will occur in multiple, independent phases. Each phase will be implemented as a Python script that:
1.  Loads an input dataset (e.g., the original `sft_dataset.jsonl` or the output of a previous phase).
2.  Performs a specific analysis or transformation.
3.  Saves the processed dataset to a new output file.

The Hugging Face `datasets` library will be used for loading, manipulating, and saving datasets. For analysis steps requiring advanced understanding (e.g., complexity scoring, feedback detection), local Large Language Models (LLMs) will be utilized via the `ollama` Python library.

## 3. Planned Phases/Steps

The following phases are planned. Each will likely correspond to a separate script:

### Phase 1: Anonymization
*   **Script:** `step1_anonymize_data.py`
*   **Status:** Implemented and executed on `sft_dataset.jsonl`.
    *   Output generated: `sft_dataset_anonymized.jsonl`.
    *   Result: No sensitive patterns (based on the initial regex set) were detected in the dataset.
*   **Objective:** Detect and anonymize sensitive information within the dataset.
*   **Initial Approach:** Utilize regular expressions (regex) to identify common patterns of sensitive data.
*   **Anonymization Strategy:** Replace identified sensitive strings with clear placeholders.
*   **Future Enhancements:** Review current regex patterns if detection was expected. Consider adding more patterns or an LLM-based approach for more nuanced PII detection if needed.
*   **Output:** A new dataset file (`sft_dataset_anonymized.jsonl`) with new columns: `anonymized_messages`, `anonymized_completion`, `sensitive_patterns_found`.

### Phase 1b: LLM-based PII Anonymization
*   **Script:** `step1b_anonymize_llm.py`
*   **Status:** Implemented and executed on `sft_dataset_anonymized.jsonl`.
    *   Input: `sft_dataset_anonymized.jsonl`
    *   Output generated: `sft_dataset_anonymized_llm.jsonl`.
    *   Result: Successfully processed 132 examples using `myaniu/qwen2.5-1m:7b` model with chunking (5000 char chunks, 1000 char overlap, 10 workers).
    *   Detected PII categories include: API_KEY_SECRET (injected test token), FILE_NAME, LOCATION_ADDRESS, LOCATOR, ORGANIZATION_NAME, OTHER_SENSITIVE_CONTEXTUAL, PHONE_NUMBER, PROJECT_CODENAME_INTERNAL, USER_ID_USERNAME.
*   **Objective:** Enhance PII detection using an LLM to identify more nuanced or contextual PII not caught by regex.
*   **Approach:**
    *   Utilize Ollama with the `myaniu/qwen2.5-1m:7b` model.
    *   Implement Pydantic models for structured JSON output from the LLM.
    *   System prompt engineered to request JSON output conforming to a specific schema.
    *   Implemented text chunking (configurable size and overlap) with parallel processing of chunks to handle long texts and improve performance.
*   **Anonymization Strategy:** Replace LLM-identified sensitive strings with placeholders (e.g., `[CATEGORY_REDACTED]`).
*   **Output:** A new dataset file (`sft_dataset_anonymized_llm.jsonl`) with new columns: `final_anonymized_messages`, `final_anonymized_completion`, `llm_sensitive_categories_found`.

### Phase 2: Complexity Scoring
*   **Script:** `step2_score_complexity.py`
*   **Status:** Heuristic scoring implemented and executed.
    *   Input: `sft_dataset_anonymized_llm.jsonl`
    *   Output generated: `sft_dataset_complexity_scored.jsonl`.
    *   Result: Successfully processed 132 examples. Dataset now includes `heuristic_complexity_score` and its components. LLM-based scoring is a placeholder.
*   **Objective:** Evaluate the complexity of each conversational trace in the dataset.
*   **Scoring:** Assign a numerical score (e.g., 0-100) representing complexity.
*   **Approach:**
    *   **Heuristic-based:** Scoring based on message length, number of turns, and presence of keywords. (Implemented)
    *   **LLM-based:** Use a local LLM (via `ollama`) to provide a more nuanced complexity assessment. (Placeholder - Future Enhancement)
*   **Output:** Dataset augmented with `heuristic_complexity_score`, `heuristic_score_num_turns`, `heuristic_score_message_length`, `heuristic_score_completion_length`, `heuristic_score_keywords`, and placeholder `llm_complexity_score` columns.

### Phase 2b: Session Identification
*   **Script:** `step2b_identify_sessions.py`
*   **Status:** Implemented and executed.
    *   Input: `sft_dataset_complexity_scored.jsonl`
    *   Output generated: `sft_dataset_sessionized.jsonl`.
    *   Result: Successfully processed 132 examples and identified 3 sessions. The session identification logic appears to be working correctly.
*   **Objective:** Add `session_id` (e.g., `session_001`) and `turn_in_session_id` (integer, 1-indexed) to each trace to group consecutive turns of a conversation.
*   **Approach:**
    *   Iterate through traces sequentially.
    *   For each trace, compare its `messages[:-1]` (context leading to the current completion) with the `previous_trace.messages + previous_trace.completion` (the state after the previous turn).
    *   If they match, it's a continuation of the current session. Increment `turn_in_session_id`.
    *   If they don't match (or it's the first trace), start a new session. Increment `session_id` and reset `turn_in_session_id` to 1.
*   **Output:** Dataset augmented with `session_id` and `turn_in_session_id` columns.

### Phase 3: Feedback/Correction Pattern Analysis
*   **Script:** `step3_analyze_correction_patterns.py` (tentative name)
*   **Objective:** Identify instances where:
    *   A user's turn provides feedback on a previous incorrect AI answer.
    *   The AI assistant corrects itself, possibly after a tool use error or other mistake.
*   **Approach:** Primarily LLM-based, using a local LLM (via `ollama`) to analyze conversational context and detect these patterns.
*   **Output:** Dataset augmented with boolean flags or descriptive tags (e.g., `is_user_feedback_on_error: true`, `assistant_self_correction: true`, `incorrect_tool_use_by_assistant: true`).

### Phase 4+ (Future Ideas)
*   Topic/Intent Analysis.
*   Instruction Quality Assessment.
*   Other analyses as identified.

## 4. Current Focus

*   **Phase 1 (Regex and LLM Anonymization) is complete.**
*   **Phase 2 (Heuristic Complexity Scoring) is complete.** LLM-based complexity scoring is deferred for now.
*   **Phase 2b (Session Identification) is complete.**
*   **Phase 3 (Feedback/Correction Pattern Analysis) is complete.** (Assuming this was completed based on previous context, if not, adjust status)

## 5. High-Level Remaining Work
*   Review the dataset after correction pattern analysis (`data/sft_dataset_S5_correction_analyzed.jsonl`).
*   Define and implement the next analysis phase or improvement.
*   (Eventually) Revisit the plan to push the final, processed, and anonymized dataset to the Hugging Face Hub.

---
*Last Updated: 2025-05-09*

## 6. Ideas for Future Improvement (Project Infrastructure & Process)
*(Status of improvements logged on 2025-05-09)*

1.  **Consolidated Logging:**
    *   **Status: COMPLETED (2025-05-09)**
    *   **Description:** All Python scripts in `src/` now use `common_utils.py` for logging to the `logs/` directory. `start_proxy.sh` also logs to this directory.
    *   **Rationale:** Centralizes logs for easier debugging, monitoring, and cleanup.

2.  **Enhanced `pyproject.toml` for `src`-Layout & Packaging:**
    *   **Status: COMPLETED (2025-05-09)**
    *   **Description:** `pyproject.toml` updated with `[tool.setuptools.packages.find] where = ["src"]`.
    *   **Rationale:** Makes the `probir` project robustly installable and aligns with modern Python packaging standards.

3.  **Formal Testing Framework (`pytest`):**
    *   **Status: INITIAL SETUP COMPLETED (2025-05-09)**
    *   **Description:** `pytest` added, `tests/` directory created, `tests/test_common_utils.py` with initial tests for `chunk_text` implemented and passing. Configuration for `src` layout and `python -m pytest` usage established.
    *   **Next Steps:** Expand test coverage for `common_utils.py` and other critical components.
    *   **Rationale:** Increases code reliability, makes refactoring safer.

4.  **Pipeline Checkpointing & Resumption:**
    *   **Status: COMPLETED (2025-05-09)**
    *   **Description:** `src/run_pipeline.py` enhanced with `--resume` and `--force-rerun` options. Checkpoints are saved to `logs/pipeline_checkpoint.json`. Individual scripts updated to exit with non-zero codes on critical errors to ensure correct checkpointing. Ollama host `http://10.0.2.2:11434` and model `myaniu/qwen2.5-1m:7b` configured in `pipeline.yaml` for LLM steps.
    *   **Rationale:** Saves time and resources by allowing pipeline resumption.

5.  **Enhanced Data Validation & Pipeline Reporting:**
    *   **Status: PENDING - CURRENT FOCUS**
    *   **Description:** Implement more rigorous data validation checks after each pipeline step. This could involve using Pydantic models to define expected data schemas. Additionally, generate a consolidated HTML or Markdown report at the end of each pipeline run, summarizing steps executed, records processed, errors encountered, and key validation outcomes.
    *   **Rationale:** Improves data integrity throughout the pipeline, makes issues easier to diagnose, and provides a clear, auditable summary of each processing run.
