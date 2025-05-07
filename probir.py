import sqlite3
import json
import logging
from mitmproxy import http, ctx # type: ignore

# --- Configuration ---
DATABASE_FILE = "/mnt/wsl_data/filtered_traffic_log.db"
# Define the domains you want to log.
# The script will log traffic if the request host *ends with* any of these domains.
# For example, "google.com" will match "www.google.com", "api.google.com", etc.
TARGET_DOMAINS = [
    "google.com",
    "example.com", # Add other domains you want to target
    "generativelanguage.googleapis.com"
    # "api.youraimodel.com", # Example for an AI model API
]
# --- End Configuration ---

# Set up basic logging for the script itself
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

class SelectiveTrafficLogger:
    def __init__(self):
        """
        Initializes the TrafficLogger addon.
        Sets up database connection and creates the necessary table.
        """
        self.conn = None
        self.cursor = None
        self._connect_db()
        self._create_table()
        logging.info(f"SelectiveTrafficLogger initialized. Targeting domains: {TARGET_DOMAINS}")

    def _connect_db(self):
        """Establishes connection to the SQLite database."""
        try:
            # isolation_level=None enables autocommit, simplifying transactions for each log entry.
            # check_same_thread=False is necessary because mitmproxy may run hooks in different threads.
            self.conn = sqlite3.connect(DATABASE_FILE, isolation_level=None, check_same_thread=False)
            self.cursor = self.conn.cursor()
            logging.info(f"Successfully connected to database: {DATABASE_FILE}")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database {DATABASE_FILE}: {e}")
            self.conn = None
            self.cursor = None

    def _create_table(self):
        """Creates the 'http_traffic' table if it doesn't already exist."""
        if not self.cursor:
            logging.error("Database connection not available, cannot create table.")
            return
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS http_traffic (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    request_method TEXT,
                    request_url TEXT,
                    request_host TEXT,
                    request_http_version TEXT,
                    request_headers TEXT,
                    request_content BLOB,
                    response_status_code INTEGER,
                    response_reason TEXT,
                    response_http_version TEXT,
                    response_headers TEXT,
                    response_content BLOB,
                    client_ip TEXT,
                    server_ip TEXT,
                    duration REAL
                )
            ''')
            logging.info("Table 'http_traffic' checked/created successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error creating table 'http_traffic': {e}")

    def _is_target_domain(self, request_host: str) -> bool:
        """Checks if the request_host matches any of the TARGET_DOMAINS."""
        if not request_host:
            return False
        for domain in TARGET_DOMAINS:
            if request_host.endswith(domain):
                return True
        return False

    def _save_to_db(self, flow: http.HTTPFlow):
        """Saves the flow (request and response) details to the database."""
        if not self.cursor:
            logging.warning("No DB cursor available. Skipping save.")
            return
        if not flow.request or not flow.response:
            logging.warning("Flow is missing request or response. Skipping save.")
            return

        req = flow.request
        resp = flow.response

        # Filter by domain
        if not self._is_target_domain(req.host):
            # logging.debug(f"Skipping non-target domain: {req.host}") # Optional: for verbose logging
            return

        logging.info(f"Target domain matched: {req.host}. Logging flow.")

        try:
            # Serialize headers to JSON strings for storage
            request_headers_json = json.dumps(dict(req.headers.items()))
            response_headers_json = json.dumps(dict(resp.headers.items()))

            # Get client and server IP addresses if available
            client_ip = flow.client_conn.peername[0] if flow.client_conn and flow.client_conn.peername else None
            server_ip = flow.server_conn.address[0] if flow.server_conn and flow.server_conn.address else None # Use .address for server

            # Calculate duration of the request-response cycle
            duration = None
            if resp.timestamp_end and req.timestamp_start:
                duration = resp.timestamp_end - req.timestamp_start

            # Get raw content (bytes) for request and response bodies
            request_content_blob = req.raw_content
            response_content_blob = resp.raw_content

            # Insert data into the table
            self.cursor.execute('''
                INSERT INTO http_traffic (
                    request_method, request_url, request_host, request_http_version, request_headers, request_content,
                    response_status_code, response_reason, response_http_version, response_headers, response_content,
                    client_ip, server_ip, duration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                req.method, req.pretty_url, req.host, req.http_version, request_headers_json, request_content_blob,
                resp.status_code, resp.reason, resp.http_version, response_headers_json, response_content_blob,
                client_ip, server_ip, duration
            ))
            # No explicit commit needed due to isolation_level=None (autocommit)

            logging.info(f"Logged to DB: {req.method} {req.pretty_url} -> {resp.status_code}")

        except sqlite3.Error as e:
            logging.error(f"Database error during insert: {e} for URL {req.pretty_url}")
        except Exception as e:
            logging.error(f"Unexpected error during logging: {e} for URL {req.pretty_url}")

    # --- Mitmproxy Event Hooks ---

    def load(self, loader):
        """
        Called when the addon is loaded by mitmproxy.
        `loader` is an instance of `mitmproxy.addonmanager.Loader`.
        """
        logging.info("SelectiveTrafficLogger addon loaded.")
        # Re-check DB connection on load, just in case
        if not self.conn or not self.cursor:
            logging.warning("DB connection was not ready on load, attempting to reconnect.")
            self._connect_db()


    def response(self, flow: http.HTTPFlow):
        """
        Called when a server response has been received.
        This is the primary hook where we decide to log the traffic.
        """
        # Ensure DB connection is alive before attempting to save
        if not self.conn or not self.cursor:
            logging.warning("Re-attempting DB connection before saving response.")
            self._connect_db() # Try to reconnect if needed

        # Log all responses to the console
        logging.info(f"Received response for: {flow.request.pretty_url} Status: {flow.response.status_code}")

        if self.conn and self.cursor: # Proceed only if connection is successful
            self._save_to_db(flow)
        else:
            logging.error("Cannot save flow, database connection is not available.")


    def done(self):
        """
        Called when mitmproxy is shutting down.
        Used here to close the database connection gracefully.
        """
        if self.conn:
            try:
                self.conn.close()
                logging.info("Database connection closed successfully.")
            except sqlite3.Error as e:
                logging.error(f"Error closing database connection: {e}")

# This is the entry point mitmproxy looks for to load addons.
# It must be a list named "addons".
addons = [
    SelectiveTrafficLogger()
]
