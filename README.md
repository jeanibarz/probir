# PROxy-Based Interaction Recorder (PROBIR)

An intercepting proxy to capture network interactions via applications (like VS Code or others configured to use a proxy) and generate a dataset from pairs of requests/responses.

This project implements a 'man-in-the-middle' for specified domains, routing traffic through a proxy-based interaction recorder to automatically log full request and response data to an SQLite database. The gathered interactions can then be filtered for relevance and quality.

The core of this project uses `mitmproxy` with a custom Python script (`probir.py`) for selective logging. Applications are configured to use this proxy via standard HTTP/HTTPS proxy environment variables.

## Project Structure

Your project root directory should typically contain:
* `probir.py` (The mitmproxy addon script)
* `start_proxy.sh` (Script to prepare and start the proxy)
* `README.md` (This file)
* `UNLICENSE`
* `.gitignore`
* `.env.example` (and optionally `.env` for local environment variables)

## Features

* **Proxy-based Interception:** Captures HTTP traffic from applications configured to use the proxy via the `HTTP_PROXY` environment variable.
* **Selective Logging:** Logs only requests/responses to/from a predefined list of target domains (configurable via the `TARGET_DOMAINS` environment variable).
* **Full Data Capture:** Saves complete request and response details, including methods, URLs, headers, and full content bodies.
* **Database Storage:** Stores captured traffic in an SQLite database (path configured via the `DATABASE_FILE` environment variable).

## How it Works

1.  **Environment Variable Configuration:** Applications are configured by setting the `HTTP_PROXY` environment variable to point to the address and port where `mitmproxy` is listening (e.g., `http://localhost:8080`).
2.  **`mitmproxy` Interception:** `mitmproxy` runs in regular proxy mode, listening on the configured port (default 8080). Applications send their HTTP traffic directly to `mitmproxy`.
3.  **Custom Python Addon:** The `mitmproxy` Python addon script (`probir.py`) inspects each flow:
    * It checks if the request's destination host matches (ends with) any of the domains listed in the `TARGET_DOMAINS` environment variable.
    * If it's a target domain, the script logs the request and corresponding response details to the SQLite database specified by `DATABASE_FILE`.

## Prerequisites

* **Operating System:** Debian or a Debian-based distribution (e.g., Ubuntu).
* **Python:** Python 3.6+ and `pip`.
* **`mitmproxy`:** The `mitmproxy` tool.
* **`curl`:** For testing.
* **`sudo` access:** Required for installing packages, managing users/groups, and running scripts that modify system state or run processes as other users.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/jeanibarz/probir.git
cd probir
```

### 2\. Install Prerequisites

```bash
sudo apt update
sudo apt install -y python3 python3-pip curl mitmproxy
# Note: On Debian-based systems, installing mitmproxy via apt is often recommended.
```

### 3\. Create a Dedicated User for `mitmproxy`

This user (`mitmproxyuser`) will run the `mitmproxy` (mitmdump) process. Ensure this user has write permissions to the directory where the database will be stored (e.g., `/mnt/wsl_data/`).

```bash
sudo adduser mitmproxyuser
# Follow the prompts. You can skip optional info.

# Grant mitmproxyuser write permissions to the database directory.
# Example: If /mnt/wsl_data/ is owned by group 'vboxsf' (common in VirtualBox/WSL):
# sudo usermod -aG vboxsf mitmproxyuser
# Replace 'vboxsf' with the actual group owning '/mnt/wsl_data/' (check with 'ls -ld /mnt/wsl_data/').
# Ensure the group has write permissions, or adjust ownership/permissions of the target database directory.
```

### 4\. Prepare `mitmproxy` Addon Environment

The `probir.py` script (located in the project root) contains the logging logic.

1.  **Review Addon Configuration:**
    Open `probir.py` in the project root. You can modify `TARGET_DOMAINS` and `DATABASE_FILE` directly in this script if needed, before starting the proxy.

2.  **Create Directory for Operational Addon Script:**
    This directory will store the copy of `probir.py` that `mitmdump` uses. The `start_proxy.sh` script (run from the project root) will copy `probir.py` from the root into this directory.

    ```bash
    sudo mkdir -p /opt/mitmproxy_scripts
    # Set ownership to mitmproxyuser so it can operate within this dir if mitmproxy needs to write other files.
    sudo chown mitmproxyuser:mitmproxyuser /opt/mitmproxy_scripts
    sudo chmod 750 /opt/mitmproxy_scripts # rwx for user, rx for group
    ```

### 5\. Running the Proxy

1.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and fill in the appropriate values:
    ```bash
    cp .env.example .env
    ```
    Edit `.env` to suit your needs, setting `DATABASE_FILE` and `TARGET_DOMAINS` as required.

2.  **Load the Environment Variables:**
    Before running the proxy, load the `.env` file:
    ```bash
    set -a; source .env; set +a
    ```
    This ensures that the environment variables are available to the script.

3.  **Make `start_proxy.sh` executable (first time only):**
    (Ensure `start_proxy.sh` is in your project root directory)
    ```bash
    chmod +x ./start_proxy.sh
    ```

4.  **Start the proxy script:**
    Run the `start_proxy.sh` script **from the project root directory**. It copies `probir.py` (from the root) to `/opt/mitmproxy_scripts/probir.py` and then starts `mitmdump` as `mitmproxyuser`.
    ```bash
    bash ./start_proxy.sh
    ```

    Keep this terminal open. `mitmdump` will run with options `--showhost` (to display the host in `mitmdump`'s output) and `-q` (quiet, reducing `mitmdump`'s own status messages). Logging from `probir.py` will still appear. The proxy listens on `http://localhost:8080`.

    *(Note: Your `start_proxy.sh` script contains commented-out lines related to `iptables` setup. This guide focuses on the explicit proxy configuration method, which is the current active behavior of the script.)*

### 6\. Launching Applications with Proxy Configuration

To make applications use the proxy, load the `.env` file in the terminal session where you launch the application.

1.  **Open a new terminal** (while the proxy from Step 5 is running).
2.  **Load the `.env` file:**
    ```bash
    source .env
    ```
3.  **Launch your application from the same terminal:**
    For example, to launch VS Code opening the current directory:
    ```bash
    code .
    ```
    To launch `curl`:
    ```bash
    curl -v https://example.com
    ```

**Note:** Remember to load the `.env` file in every new terminal session from which you intend to launch applications through the proxy.

### 7\. Testing the Proxy

1.  Ensure `mitmproxy` (via `start_proxy.sh`) is running.
2.  Launch your application from a terminal where the `.env` file is loaded (see Step 6).
3.  Perform a network action targeting a domain listed in `TARGET_DOMAINS`.
4.  Check the `mitmproxy` terminal for logs and verify data in the database.

Test with `curl` (from a terminal with `.env` loaded):
```bash
curl -v https://example.com
```

### 8\. Stopping the Proxy

In the terminal running `bash ./start_proxy.sh`, press `Ctrl+C`.

### 9\. Database

Captured traffic is stored in an SQLite database (default: `filtered_traffic_log.db` at `/mnt/wsl_data/`, path from `DATABASE_FILE` in `probir.py`). The schema defined in your `probir.py` is:

  * `id`: INTEGER PRIMARY KEY AUTOINCREMENT
  * `timestamp`: DATETIME DEFAULT CURRENT\_TIMESTAMP (Time of record insertion)
  * `request_method`: TEXT
  * `request_url`: TEXT (Full URL as `pretty_url` from mitmproxy)
  * `request_host`: TEXT
  * `request_http_version`: TEXT
  * `request_headers`: TEXT (JSON encoded)
  * `request_content`: BLOB
  * `response_status_code`: INTEGER
  * `response_reason`: TEXT
  * `response_http_version`: TEXT
  * `response_headers`: TEXT (JSON encoded)
  * `response_content`: BLOB
  * `client_ip`: TEXT
  * `server_ip`: TEXT
  * `duration`: REAL (Request-response cycle time in seconds)

Inspect with `sqlite3`:

```bash
sqlite3 /mnt/wsl_data/filtered_traffic_log.db
sqlite> .headers on
sqlite> .schema http_traffic
sqlite> SELECT id, timestamp, request_host, request_url FROM http_traffic ORDER BY id DESC LIMIT 5;
sqlite> .quit
```

Note: The `timestamp` column defaults to the time the record is inserted into the database. `request_url` does not have a `UNIQUE` constraint, so multiple entries for the same URL are possible.

### 10\. Environment Variables (`.env` and `.env.example`)

The project includes `.env.example`, which you should copy to `.env` for local configuration. The `.env` file allows you to customize settings without modifying the codebase.

1. **Copy `.env.example` to `.env`:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` to suit your needs:**
   - `DATABASE_FILE`: Path to the SQLite database file (e.g., `/path/to/your/database.db`).
   - `TARGET_DOMAINS`: Comma-separated list of domains to log (e.g., `example.com,yourtargetdomain.com`).

These variables control the behavior of `probir.py`. After modifying `.env`, ensure that your environment is updated by sourcing it:
   ```bash
   source .env
   ```
   or by running your application from the same terminal session.

**Note:** `probir.py` reads these environment variables. Make sure to set them before running the proxy.

## Troubleshooting

  * **No Traffic Reaching Proxy:**
      * Verify `mitmproxy` (via `start_proxy.sh`) is running and listening (`sudo ss -tulnp | grep 8080`).
      * Check `HTTP_PROXY`/`HTTPS_PROXY` (`env | grep _PROXY`) in the application's terminal.
      * Ensure the application was launched from the terminal where these variables were set.
  * **Certificate Errors for HTTPS (Expected for Target Domains):**
      * SSL/TLS errors are **expected** for targeted HTTPS domains due to mitmproxy's dynamic certificates not being system-trusted.
      * **Solution:** Client-side bypass: Set `NODE_TLS_REJECT_UNAUTHORIZED=0` for Node.js apps, use `curl -k`, or equivalent for other clients, as described in Step 6.
  * **Database Issues (`unable to open database file`, etc.):**
      * Ensure `mitmproxyuser` has write permissions to the directory of `DATABASE_FILE` (e.g., `/mnt/wsl_data/`).
      * Check group ownership (`ls -ld /mnt/wsl_data/`) and `mitmproxyuser` group membership (`groups mitmproxyuser`). Adjust with `sudo usermod -aG <groupname> mitmproxyuser` if needed. Restart proxy or re-login for group changes.
  * **Script Errors in `probir.py`:**
      * Check the terminal output from `bash ./start_proxy.sh`. `mitmdump` runs with `-q` (quiet), but Python logging from `probir.py` should still be visible via mitmproxy's logging mechanisms. Errors in `probir.py` should appear there.
  * **Quick Test:**
      * To quickly verify the proxy is capturing traffic, you can use a command like `curl -v http://example.com` from a terminal where `HTTP_PROXY` is set. If the proxy is working, you should see the request logged in the `mitmproxy` terminal and the response captured in the database.

This setup provides a flexible way to capture specific network interactions for analysis and dataset creation, with a simplified approach to HTTPS handling.

## License

This project is released into the public domain. See the [UNLICENSE](UNLICENSE) file for details.
