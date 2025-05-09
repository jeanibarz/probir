import sqlite3
import json
import os
import argparse
import logging
from common_utils import setup_logging

# VALID_TRACES_TABLE_NAME can remain global or become a parameter if needed
VALID_TRACES_TABLE_NAME = 'valid_llm_traces'

# Logger will be configured by setup_logging in main
logger = logging.getLogger(__name__)

def get_blob_preview(blob_data, length=100):
    if not blob_data:
        return "None" 
    if isinstance(blob_data, str): # If it's already a string
        return blob_data[:length]
    try:
        return blob_data[:length].decode('utf-8', errors='replace')
    except Exception:
        return blob_data[:length//2].hex()

def print_table_content(cursor, table_name, limit=None):
    logger.info(f"\n--- Content of table '{table_name}' ---")
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        logger.info(f"Total rows: {count}")
        if count == 0:
            logger.info("Table is empty.")
            return

        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        query += ";"
        
        cursor.execute(query)
        column_names = [description[0] for description in cursor.description]
        logger.info(f"Columns: {column_names}")
        
        rows = cursor.fetchall()
        for i, row_bytes in enumerate(rows):
            row_decoded = []
            for item in row_bytes:
                if isinstance(item, bytes):
                    try:
                        row_decoded.append(item.decode('utf-8'))
                    except UnicodeDecodeError:
                        row_decoded.append(f"<bytes_preview:{get_blob_preview(item, 50)}>") # Show preview for non-utf8 bytes
                else:
                    row_decoded.append(item)
            logger.info(f"Row {i+1}: {tuple(row_decoded)}")

    except sqlite3.Error as e:
        logger.error(f"SQLite error when querying {table_name}: {e}")


def inspect_database_logic(db_path: str):
    """Main logic for inspecting the database."""
    if not os.path.exists(db_path):
        logger.error(f"Error: Database file not found at {db_path}")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes 
        cursor = conn.cursor()

        logger.info(f"Inspecting database: {db_path}\n")

        # 1. Get schema of http_traffic
        logger.info("--- Schema for http_traffic ---")
        cursor.execute("PRAGMA table_info(http_traffic);")
        schema_rows = cursor.fetchall()
        if schema_rows:
            for row_bytes in schema_rows:
                row = tuple(item.decode('utf-8') if isinstance(item, bytes) else item for item in row_bytes)
                logger.info(row)
        else:
            logger.info("Table 'http_traffic' not found or empty schema.")
        
        print_table_content(cursor, "http_traffic", limit=5)


        # --- Inspect valid_llm_traces table ---
        logger.info(f"\n\n--- Inspecting table: {VALID_TRACES_TABLE_NAME} ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (VALID_TRACES_TABLE_NAME,))
        if cursor.fetchone():
            logger.info(f"Schema for {VALID_TRACES_TABLE_NAME}:")
            cursor.execute(f"PRAGMA table_info({VALID_TRACES_TABLE_NAME});")
            valid_schema_rows = cursor.fetchall()
            for row_bytes in valid_schema_rows:
                row = tuple(item.decode('utf-8') if isinstance(item, bytes) else item for item in row_bytes)
                logger.info(row)
            
            print_table_content(cursor, VALID_TRACES_TABLE_NAME) # Print all rows
        else:
            logger.info(f"Table '{VALID_TRACES_TABLE_NAME}' does not exist.")
        

        # 5. Try to decode and print a full request/response pair for a successful LLM call from http_traffic
        logger.info("\n\n--- Example of a decoded LLM request/response from http_traffic (if available) ---")
        cursor.execute("""
            SELECT id, request_url, request_content, response_content, response_headers
            FROM http_traffic
            WHERE request_host LIKE '%generativelanguage.googleapis.com%' AND response_status_code = 200
                  AND LENGTH(request_content) > 0 AND LENGTH(response_content) > 0
            LIMIT 1;
        """)
        llm_content_row_bytes = cursor.fetchone()
        if llm_content_row_bytes:
            row_id, req_url_bytes, req_blob, resp_blob, resp_headers_bytes = llm_content_row_bytes
            req_url = req_url_bytes.decode('utf-8')
            resp_headers_json = resp_headers_bytes.decode('utf-8')
            resp_headers = json.loads(resp_headers_json)
            
            logger.info(f"Displaying full content for original http_traffic.id = {row_id}")
            logger.info("Request Content:")
            try:
                req_text = req_blob.decode('utf-8')
                req_json = json.loads(req_text)
                logger.info(json.dumps(req_json, indent=2))
            except Exception as e:
                logger.error(f"Error decoding request: {e}")

            logger.info("\n---\n")
            logger.info("Response Content:")
            content_type = resp_headers.get("content-type", "").lower()
            is_sse = "text/event-stream" in content_type or "alt=sse" in req_url.lower() or ":streamgeneratecontent" in req_url.lower()

            if is_sse:
                logger.info("Detected SSE format. Displaying first few JSON objects from stream:")
                resp_text = resp_blob.decode('utf-8', errors='replace')
                sse_objects_found = 0
                for line in resp_text.splitlines():
                    if line.startswith("data:"):
                        json_str = line[len("data:"):].strip()
                        if json_str:
                            try:
                                resp_json_obj = json.loads(json_str)
                                logger.info(f"--- SSE Part {sse_objects_found + 1} ---")
                                logger.info(json.dumps(resp_json_obj, indent=2))
                                sse_objects_found += 1
                                if sse_objects_found >= 3: 
                                    logger.info("--- (more SSE parts may exist) ---")
                                    break 
                            except json.JSONDecodeError:
                                logger.warning(f"--- SSE Part (JSON Parse Error for: {json_str}) ---")
                    elif line.strip() == "" and sse_objects_found > 0: 
                        pass 
                if sse_objects_found == 0:
                    logger.warning("SSE detected, but no 'data:' lines with parsable JSON found.")
                    logger.info(f"Full SSE Response Preview (first 500 chars, errors replaced): {resp_text[:500]}")
            else: 
                logger.info("Attempting to parse as single JSON object:")
                try:
                    resp_text = resp_blob.decode('utf-8')
                    resp_json = json.loads(resp_text)
                    logger.info(json.dumps(resp_json, indent=2))
                except Exception as e:
                    logger.error(f"Error decoding response: {e}")
        else:
            logger.info("No suitable LLM request/response pair found for full decoding example in http_traffic.")


    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

def main():
    parser = argparse.ArgumentParser(description="Inspect tables in the SQLite database.")
    parser.add_argument(
        "--db_path",
        type=str,
        default='/home/jean/git/probir/filtered_traffic_log.db',
        help="Path to the SQLite database file (default: /home/jean/git/probir/filtered_traffic_log.db)."
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
        help="Optional: Name of the log file to be saved in the 'logs' directory (e.g., 'inspect_db.log'). If not provided, logs only to console."
    )
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)

    global logger # Re-initialize logger for this module
    logger = logging.getLogger(__name__)
    
    if args.log_file_name:
        logger.info(f"Inspection output will also be saved to logs/{args.log_file_name}")

    inspect_database_logic(args.db_path)

if __name__ == '__main__':
    main()
