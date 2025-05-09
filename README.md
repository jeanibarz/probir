# PROxy-Based Interaction Recorder & Dataset Curator (PROBIR)

PROBIR is a comprehensive suite of tools designed to capture network interactions via a proxy and then process this raw data through a configurable pipeline to generate curated datasets suitable for various downstream tasks, including training SFT models.

Initially focused on intercepting traffic with `mitmproxy`, the project has expanded to include a multi-step data processing pipeline for anonymization, session identification, complexity scoring, pattern analysis, and dataset formatting for platforms like Hugging Face.

## Core Components

1.  **Traffic Interception (Proxy):**
    *   Uses `mitmproxy` with a custom Python script (`src/probir.py`) to selectively log HTTP/S requests and responses from configured applications to an SQLite database.
2.  **Data Curation Pipeline:**
    *   A series of Python scripts (primarily in `src/`) orchestrated by `src/run_pipeline.py` to transform the raw captured data into a structured and refined dataset.
    *   Supports checkpointing and resumption.

## Key Features

### Proxy Features
*   **Selective Interception:** Captures HTTP/S traffic from applications configured to use the proxy, targeting specific domains.
*   **Full Data Capture:** Saves complete request/response details (headers, bodies) to an SQLite database.
*   **Configurable:** Target domains and database paths are set via environment variables.

### Data Curation Pipeline Features
*   **Modular Pipeline:** Process data through a series of configurable steps defined in `pipeline.yaml`.
*   **Data Extraction:** Extracts relevant traces from the raw proxy logs.
*   **Session Identification:** Groups related interactions into sessions.
*   **Anonymization:** Includes regex-based and LLM-based PII removal.
*   **Complexity Scoring:** Assigns scores to data samples based on defined metrics.
*   **Correction Pattern Analysis:** Identifies patterns in user corrections or feedback.
*   **Dataset Generation:** Prepares datasets in formats like JSONL, suitable for Hugging Face.
*   **Hugging Face Hub Integration:** Scripts to push datasets directly to the Hugging Face Hub.
*   **Checkpointing & Resumption:** The pipeline can be resumed from the last successful step.
*   **Data Validation & Reporting:** Includes mechanisms for validating data at each step and generating execution reports.
*   **Shared Utilities:** A common utility module (`src/common_utils.py`) for logging, argument parsing, data I/O, etc.
*   **Testing:** Basic test suite using `pytest` for core utilities.

## Project Structure

The project follows a `src/` layout:

*   `src/`
    *   `probir.py`: The core mitmproxy addon script for traffic interception.
    *   `run_pipeline.py`: Orchestrator for the data curation pipeline.
    *   `step*.py` (e.g., `step1_anonymize_data.py`): Individual processing scripts for the pipeline.
    *   `common_utils.py`: Shared utilities for all Python scripts.
    *   Other utility scripts (e.g., `push_hf_dataset.py`, `inspect_db.py`).
*   `tests/`: Contains `pytest` tests.
*   `pipeline.yaml`: Configuration file defining the stages of the data curation pipeline.
*   `pyproject.toml`: Defines project dependencies and packaging information.
*   `.env.example`: Template for environment variables (copy to `.env`).
*   `start_proxy.sh`: Script to start the mitmproxy interceptor.
*   `README.md`: This file.
*   `UNLICENSE`: Project license.
*   `logs/`: Directory for log files (gitignored).
*   `data/`: Directory for datasets (gitignored).
*   `memory-bank/`: Cline's internal memory files (gitignored).

## Setup and Installation

### Prerequisites
*   Python 3.8+
*   `mitmproxy` (can be installed via pip or system package manager)
*   `uv` (recommended for fast environment setup) or `pip`

### Installation Steps

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/jeanibarz/probir.git
    cd probir
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    Using `uv` (recommended):
    ```bash
    uv pip install -e .
    ```
    Or using `pip`:
    ```bash
    pip install -e .
    ```
    This installs the `probir` package in editable mode along with all its dependencies defined in `pyproject.toml`.

4.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and customize it:
    ```bash
    cp .env.example .env
    nano .env
    ```
    Key variables for the proxy:
    *   `DATABASE_FILE`: Path to the SQLite database for raw traffic.
    *   `TARGET_DOMAINS`: Comma-separated domains to intercept.
    Key variables for the pipeline (if using LLM-based steps):
    *   `OLLAMA_HOST`: URL for the Ollama server.
    *   Other API keys or model names as required by specific pipeline steps.

5.  **(For Proxy) Mitmproxy User Setup (If running proxy as a different user):**
    The original README contained detailed steps for setting up a `mitmproxyuser`. If you intend to run `mitmproxy` under a dedicated unprivileged user, refer to `mitmproxy` documentation and ensure that user has permissions to write to `DATABASE_FILE` and the directory specified in `start_proxy.sh` for script copying (`/opt/mitmproxy_scripts/`). The `start_proxy.sh` script handles running `mitmdump` as `mitmproxyuser`.

## Usage

### 1. Running the Traffic Interception Proxy

1.  **Load Environment Variables:**
    Ensure your `.env` file is configured. In your terminal:
    ```bash
    set -a; source .env; set +a
    ```
2.  **Make `start_proxy.sh` executable (first time):**
    ```bash
    chmod +x ./start_proxy.sh
    ```
3.  **Start the Proxy:**
    ```bash
    bash ./start_proxy.sh
    ```
    The proxy will listen on `http://localhost:8080` by default.

4.  **Configure Applications to Use the Proxy:**
    In a new terminal (where you'll launch the application), load the `.env` file again (as it sets `HTTP_PROXY` and `HTTPS_PROXY`):
    ```bash
    set -a; source .env; set +a
    ```
    Then launch your application (e.g., `code .`, `curl https://example.com`). Traffic to `TARGET_DOMAINS` will be logged.

### 2. Running the Data Curation Pipeline

The pipeline is orchestrated by `src/run_pipeline.py`.

1.  **Configure `pipeline.yaml`:**
    Define the sequence of steps, their parameters, and input/output files in `pipeline.yaml`.

2.  **Ensure Input Data:**
    Make sure the initial input file for the pipeline (often the output of an extraction step from `filtered_traffic_log.db`, or the database itself) exists.

3.  **Run the Pipeline:**
    From the project root, with your virtual environment activated:
    ```bash
    python src/run_pipeline.py --config pipeline.yaml
    ```
    Common options:
    *   `--config <path_to_yaml>`: Specify the pipeline configuration file.
    *   `--resume`: Resume from the last checkpoint if available.
    *   `--force-rerun`: Ignore any existing checkpoint and run all steps.
    *   `--log-level DEBUG/INFO/WARNING/ERROR`: Set the logging level.

    The pipeline will process data according to `pipeline.yaml`, creating intermediate and final output files in the `data/` directory (or as specified in the config). Checkpoint information is stored in `logs/pipeline_checkpoint.json`. A summary report is generated in `logs/pipeline_execution_report.md`.

### 3. Other Scripts

*   **`src/inspect_db.py`:** Utility to inspect the schema and contents of the SQLite database.
    ```bash
    python src/inspect_db.py --db-path path/to/your/filtered_traffic_log.db
    ```
*   **`src/push_hf_dataset.py`:** Pushes a local dataset file to the Hugging Face Hub.
    ```bash
    python src/push_hf_dataset.py --file-path data/your_dataset.jsonl --repo-id your_username/your_dataset_name --private
    ```
    Requires `huggingface-cli login` beforehand.

## Configuration

*   **`.env` file:** For environment-specific settings like API keys, database paths, target domains for the proxy, and Ollama host.
*   **`pipeline.yaml`:** Defines the stages, scripts, parameters, and input/output flow for the data curation pipeline.
*   **`pyproject.toml`:** Manages Python dependencies and project metadata.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is released into the public domain. See the [UNLICENSE](UNLICENSE) file for details.
