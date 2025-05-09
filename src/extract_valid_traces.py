import sqlite3
import json
import os
import logging
import traceback
import argparse
from common_utils import setup_logging

# Configuration for table name and target hosts can remain global or become params
TARGET_TABLE_NAME = 'valid_llm_traces'
TARGET_LLM_HOSTS = ['generativelanguage.googleapis.com'] 

# Logger will be configured by setup_logging in main
logger = logging.getLogger(__name__)

def get_text_from_parts(parts_list):
    """Extracts and concatenates text from a list of 'parts' objects."""
    texts = []
    if isinstance(parts_list, list):
        for part in parts_list:
            if isinstance(part, dict) and 'text' in part:
                texts.append(part['text'])
    return "".join(texts)

def extract_model_from_request_json(req_json):
    """Attempts to extract model name from request JSON."""
    if not isinstance(req_json, dict):
        return None
    if 'model' in req_json: 
        return req_json['model']
    if 'contents' in req_json and isinstance(req_json['contents'], list) and len(req_json['contents']) > 0:
        first_content = req_json['contents'][0]
        # Model might be specified in URL, not request body for generateContent/streamGenerateContent
    return None # Rely on URL extraction primarily for generate/stream calls

def extract_model_from_url(req_url):
    if "models/" in req_url:
        try:
            model_part = req_url.split("models/")[1]
            model_name = model_part.split(":")[0].split("/")[0]
            return model_name
        except IndexError:
            return None
    return None


def create_target_table(cursor):
    """Creates the target table, dropping if it exists for a clean run."""
    try:
        logging.info(f"Dropping table '{TARGET_TABLE_NAME}' if it exists...")
        cursor.execute(f"DROP TABLE IF EXISTS {TARGET_TABLE_NAME}")
        logging.info(f"Creating table '{TARGET_TABLE_NAME}'...")
        cursor.execute(f'''
            CREATE TABLE {TARGET_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_http_traffic_id INTEGER UNIQUE, -- Ensure uniqueness
                timestamp DATETIME,
                request_url TEXT,
                model_name TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                llm_response_text TEXT,
                request_full_json TEXT,
                response_parts_json_array TEXT,
                duration REAL,
                prompt_token_count INTEGER,
                candidates_token_count INTEGER,
                total_token_count INTEGER,
                FOREIGN KEY (original_http_traffic_id) REFERENCES http_traffic (id)
            )
        ''')
        logging.info(f"Table '{TARGET_TABLE_NAME}' created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error recreating table '{TARGET_TABLE_NAME}': {e}")
        raise

def process_database_logic(db_path: str):
    """Main logic for processing the database, extracted to be callable from main()."""
    logger.info(f"Starting database processing for: {db_path}")
    if not os.path.exists(db_path):
        logger.error(f"Source database file not found: {db_path}")
        return

    source_conn = None
    rows_processed = 0
    traces_inserted = 0
    
    try:
        source_conn = sqlite3.connect(db_path)
        source_conn.text_factory = bytes 
        source_cursor = source_conn.cursor()
        insert_cursor = source_conn.cursor() # Create a separate cursor for inserts

        create_target_table(source_cursor) # Recreate table for clean run (using source_cursor is fine here)

        logger.info("Starting to fetch rows from http_traffic...")
        source_cursor.execute("""
            SELECT id, timestamp, request_url, request_host, request_content, 
                   response_content, response_headers, duration, response_status_code
            FROM http_traffic
            ORDER BY id -- Process in order
        """)

        while True: # Loop using fetchone
            row_bytes = source_cursor.fetchone()
            if row_bytes is None:
                logging.info("Fetched all rows from source table.")
                break # Exit loop if no more rows

            rows_processed += 1
            original_id = row_bytes[0] # Get ID early for logging
            logging.debug(f"Processing row ID {original_id}...")

            try:
                # Decode necessary fields from bytes
                ts_bytes, req_url_bytes, req_host_bytes, \
                req_content_blob, resp_content_blob, resp_headers_bytes, duration, response_status_code = row_bytes[1:]

                # --- Apply Filtering Criteria ---
                request_host = req_host_bytes.decode('utf-8', errors='replace')
                if not any(request_host.endswith(target_host) for target_host in TARGET_LLM_HOSTS):
                    logging.debug(f"Skipping row ID {original_id}: Host '{request_host}' not in target list.")
                    continue
                
                if response_status_code != 200:
                    logging.debug(f"Skipping row ID {original_id}: Status code is {response_status_code}, not 200.")
                    continue

                if not req_content_blob or not resp_content_blob:
                    logging.warning(f"Skipping row ID {original_id}: Empty request or response content (Status: {response_status_code}).")
                    continue
                
                timestamp = ts_bytes.decode('utf-8', errors='replace')
                request_url = req_url_bytes.decode('utf-8', errors='replace')
                response_headers_str = resp_headers_bytes.decode('utf-8', errors='replace')
                response_headers = json.loads(response_headers_str) # Assume headers are valid JSON

                # --- Parse Request ---
                try:
                    req_text = req_content_blob.decode('utf-8')
                    req_json = json.loads(req_text)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    logging.warning(f"Skipping row ID {original_id}: Request content parsing error - {e}")
                    continue
                
                request_full_json_str = json.dumps(req_json) 

                system_prompt_parts = req_json.get('systemInstruction', {}).get('parts', [])
                system_prompt = get_text_from_parts(system_prompt_parts) if system_prompt_parts else None

                user_prompt_text_parts = []
                for content_item in req_json.get('contents', []):
                     # Handle cases where role might be missing or different casing
                    role = content_item.get('role', '').lower() if isinstance(content_item, dict) else ''
                    if role == 'user':
                        user_prompt_text_parts.append(get_text_from_parts(content_item.get('parts', [])))
                user_prompt = "".join(user_prompt_text_parts)
                
                model_name = extract_model_from_request_json(req_json)
                if not model_name:
                    model_name = extract_model_from_url(request_url)

                # --- Parse Response (Handle SSE vs. Single JSON) ---
                llm_response_text_parts = []
                response_json_objects_list = []
                prompt_token_count, candidates_token_count, total_token_count = None, None, None

                content_type = response_headers.get("content-type", "").lower()
                is_sse = "text/event-stream" in content_type or "alt=sse" in request_url.lower() or ":streamgeneratecontent" in request_url.lower()

                try:
                    resp_text = resp_content_blob.decode('utf-8')
                except UnicodeDecodeError as e:
                    # Check for gzip magic number
                    if resp_content_blob.startswith(b'\x1f\x8b'):
                         logging.warning(f"Skipping row ID {original_id}: Response content appears gzipped but wasn't decompressed. Error: {e}")
                    else:
                         logging.warning(f"Skipping row ID {original_id}: Response content UTF-8 decoding error - {e}")
                    continue # Skip this row if response isn't decodable

                if is_sse:
                    sse_parse_error = False
                    for line in resp_text.splitlines():
                        if line.startswith("data:"):
                            json_str = line[len("data:"):].strip()
                            if json_str:
                                try:
                                    sse_json_obj = json.loads(json_str)
                                    response_json_objects_list.append(sse_json_obj)
                                    
                                    candidates = sse_json_obj.get('candidates', [])
                                    if candidates and isinstance(candidates, list):
                                        for candidate in candidates:
                                            if isinstance(candidate, dict):
                                                llm_response_text_parts.append(get_text_from_parts(candidate.get('content', {}).get('parts', [])))
                                    
                                    usage = sse_json_obj.get('usageMetadata', {})
                                    if usage: 
                                        prompt_token_count = usage.get('promptTokenCount')
                                        candidates_token_count = usage.get('candidatesTokenCount')
                                        total_token_count = usage.get('totalTokenCount')
                                    if not model_name and 'modelVersion' in sse_json_obj: 
                                        model_name = sse_json_obj['modelVersion']

                                except json.JSONDecodeError:
                                    logging.warning(f"Row ID {original_id}: SSE data part JSON parse error for: {json_str[:100]}...") # Log only preview
                                    sse_parse_error = True # Mark potential issue but try to continue
                    if not response_json_objects_list and not sse_parse_error:
                         logging.warning(f"Row ID {original_id}: SSE detected but no valid 'data:' lines found.")
                         # Optionally skip if no data parts found, or allow insert if text was somehow extracted
                         # continue 

                else: # Not SSE, try as single JSON
                    try:
                        single_resp_json = json.loads(resp_text)
                        response_json_objects_list.append(single_resp_json)
                        
                        candidates = single_resp_json.get('candidates', [])
                        if candidates and isinstance(candidates, list):
                            for candidate in candidates:
                                if isinstance(candidate, dict):
                                     llm_response_text_parts.append(get_text_from_parts(candidate.get('content', {}).get('parts', [])))

                        usage = single_resp_json.get('usageMetadata', {})
                        if usage:
                            prompt_token_count = usage.get('promptTokenCount')
                            candidates_token_count = usage.get('candidatesTokenCount')
                            total_token_count = usage.get('totalTokenCount')
                        if not model_name and 'modelVersion' in single_resp_json:
                             model_name = single_resp_json['modelVersion']

                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping row ID {original_id}: Single response JSON parsing error - {e}")
                        continue
                
                # Only insert if we successfully parsed response parts or a single response
                if not response_json_objects_list: 
                    logging.warning(f"Skipping row ID {original_id}: No valid response JSON objects could be parsed.")
                    continue

                llm_response_text_final = "".join(llm_response_text_parts)
                response_parts_json_array_str = json.dumps(response_json_objects_list)

                # --- Insert into target table ---
                logging.debug(f"Attempting insert for original ID {original_id}...")
                insert_cursor.execute(f'''
                    INSERT INTO {TARGET_TABLE_NAME} (
                        original_http_traffic_id, timestamp, request_url, model_name,
                        system_prompt, user_prompt, llm_response_text,
                        request_full_json, response_parts_json_array, duration,
                        prompt_token_count, candidates_token_count, total_token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    original_id, timestamp, request_url, model_name,
                    system_prompt, user_prompt, llm_response_text_final,
                    request_full_json_str, response_parts_json_array_str, duration,
                    prompt_token_count, candidates_token_count, total_token_count
                ))
                traces_inserted += 1
                logging.debug(f"Successfully inserted trace for original ID {original_id}.")

            except Exception as e:
                logger.error(f"Critical error processing row ID {original_id}: {e}\n{traceback.format_exc()}")
                # Decide whether to continue or stop on critical errors
                # continue 

        # --- End of loop ---
        source_conn.commit() 
        logger.info(f"Processing loop finished. Processed {rows_processed} rows. Inserted {traces_inserted} valid traces into '{TARGET_TABLE_NAME}'.")

    except Exception as e:
        logger.error(f"Major error during database processing: {e}\n{traceback.format_exc()}")
    finally:
        if source_conn:
            source_conn.close()
            logger.info("Database connection closed.")

def main():
    parser = argparse.ArgumentParser(description="Extract valid LLM traces from http_traffic table into a new table.")
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
        help="Optional: Name of the log file to be saved in the 'logs' directory (e.g., 'extract_traces.log'). If not provided, logs only to console."
    )
    args = parser.parse_args()

    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level_int, args.log_file_name)
    
    # Re-initialize logger for this module after setup_logging has run
    global logger
    logger = logging.getLogger(__name__)

    if args.log_file_name:
        logger.info(f"General logs for this script will also be saved to logs/{args.log_file_name}")

    process_database_logic(args.db_path)

if __name__ == '__main__':
    main()
