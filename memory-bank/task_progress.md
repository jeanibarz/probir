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

## 3. Planned Phases/Steps (Dataset Processing)

### Phase 1: Anonymization (Regex-based)
*   **Status:** COMPLETED (Output: `data/sft_dataset_anonymized.jsonl`)

### Phase 1b: LLM-based PII Anonymization
*   **Status:** COMPLETED (Output: `data/sft_dataset_anonymized_llm.jsonl`)

### Phase 2: Complexity Scoring (Heuristic)
*   **Status:** COMPLETED (Output: `data/sft_dataset_complexity_scored.jsonl`)

### Phase 2b: Session Identification
*   **Status:** COMPLETED (Output: `data/sft_dataset_sessionized.jsonl`, then used as input for S2, S3, S4, S5 naming convention, final pre-analysis was `data/sft_dataset_S1_sessionized.jsonl` which then became input for step 2 in `pipeline.yaml`)
    *   *Note: The pipeline.yaml now reflects a different order: Session ID (S1) -> Regex Anonymize (S2) -> LLM Anonymize (S3) -> Complexity (S4) -> Correction Analysis (S5). The final output of this sequence before new work is `data/sft_dataset_S5_correction_analyzed.jsonl`.*

### Phase 3: Feedback/Correction Pattern Analysis
*   **Status:** COMPLETED (Output: `data/sft_dataset_S5_correction_analyzed.jsonl`)

### Phase 4+ (Future Ideas for Dataset Processing)
*   Topic/Intent Analysis.
*   Instruction Quality Assessment.
*   Other analyses as identified.

## 4. Current Focus & High-Level Remaining Work

*   **Review `data/sft_dataset_S5_correction_analyzed.jsonl`:** Analyze the output of the last completed pipeline run.
*   **Enhanced Data Validation & Pipeline Reporting (from Section 6):** This remains a key area for improvement.
*   **Define and implement the next analysis phase or improvement** based on review and the new ideas in Section 7.
*   (Eventually) Revisit the plan to push the final, processed, and anonymized dataset to the Hugging Face Hub.

---
*Last Updated: 2025-05-09*

## 6. Ideas for Future Improvement (Project Infrastructure & Process - Original List)

1.  **Consolidated Logging:**
    *   **Status: COMPLETED (2025-05-09)**

2.  **Enhanced `pyproject.toml` for `src`-Layout & Packaging:**
    *   **Status: COMPLETED (2025-05-09)**

3.  **Formal Testing Framework (`pytest`):**
    *   **Status: INITIAL SETUP COMPLETED (2025-05-09)**
    *   **Description:** `pytest` added, `tests/` directory created, `tests/test_common_utils.py` with initial tests for `chunk_text` implemented and passing. Configuration for `src` layout and `python -m pytest` usage established.
    *   **Next Steps:** Expand test coverage for `common_utils.py` and other critical components.
    *   **Rationale:** Increases code reliability, makes refactoring safer.

4.  **Pipeline Checkpointing & Resumption:**
    *   **Status: COMPLETED (2025-05-09)**

5.  **Enhanced Data Validation & Pipeline Reporting:**
    *   **Status: PENDING - CURRENT FOCUS**
    *   **Description:** Implement more rigorous data validation checks after each pipeline step. This could involve using Pydantic models to define expected data schemas. Additionally, generate a consolidated HTML or Markdown report at the end of each pipeline run, summarizing steps executed, records processed, errors encountered, and key validation outcomes.
    *   **Rationale:** Improves data integrity throughout the pipeline, makes issues easier to diagnose, and provides a clear, auditable summary of each processing run.

---
## 7. New Proposed Repository Improvements (as of 2025-05-09)

This section lists newly proposed ideas for enhancing the `probir` repository and its functionalities.

1.  **Enhanced Testing Strategy & Coverage:**
    *   **Status:** TODO
    *   **Idea:** Implement comprehensive integration tests for each pipeline step (`step*.py`) and end-to-end tests for the entire pipeline orchestrator (`run_pipeline.py`). Introduce test coverage measurement (e.g., using `pytest-cov`).
    *   **Benefit:** Greatly improves reliability and safety of future refactoring.

2.  **Advanced Configuration Management & Secrets Handling:**
    *   **Status:** TODO
    *   **Idea:** Use Pydantic models for structured configuration of the pipeline and its steps (e.g., loaded from `pipeline.yaml` and CLI overrides). Explore more robust secrets management strategies beyond the current `.env` file for sensitive data like API keys (e.g., by abstracting how secrets are loaded, preparing for integration with systems like Vault or cloud-specific secret managers).
    *   **Benefit:** Enhances maintainability, security, and configuration flexibility.

3.  **Interactive Data Exploration & Filtering Tool:**
    *   **Status:** POSTPONED
    *   **Idea:** Develop a simple web-based tool (e.g., using Streamlit or Gradio) or an enhanced CLI to load, view, filter, and export subsets of the datasets generated by the pipeline (JSONL files in `data/`).
    *   **Benefit:** Makes it much easier to understand, debug, and curate the data at various stages.
    *   **Note:** To be addressed after other improvements.

4.  **Pipeline Output Schema Evolution & Versioning:**
    *   **Status:** TODO
    *   **Idea:** Define more explicit Pydantic input/output schemas for each pipeline step, building upon the existing `BaseTrace`. Consider implementing a simple schema versioning system for data files. Explore using Apache Parquet format for intermediate and final datasets for efficiency with larger datasets.
    *   **Benefit:** Improves data integrity, makes pipeline evolution more robust, and can enhance performance.

5.  **Enhanced Pipeline Monitoring & Metrics Dashboard:**
    *   **Status:** TODO
    *   **Idea:** Implement structured logging (e.g., JSON format) for easier machine parsing. Systematically collect more detailed operational metrics (records processed per step, time per step, error rates, PII detection counts, average complexity scores). Enhance the pipeline's Markdown report with more visualizations or explore integration with dashboarding tools for advanced monitoring.
    *   **Benefit:** Provides better visibility into pipeline health, performance, and data quality.
