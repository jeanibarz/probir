#!/bin/bash

# Script to start the mitmproxy proxy-based interaction recorder

# Copy the addon script to the mitmproxy scripts directory
sudo cp src/probir.py /opt/mitmproxy_scripts/probir.py

# Find the absolute path of mitmdump for the current user
MITMDUMP_PATH=$(which mitmdump)

# Check if mitmdump was found
if [ -z "$MITMDUMP_PATH" ]; then
    echo "Error: mitmdump not found in the current user's PATH. Please ensure it is installed and in your PATH."
    exit 1
fi

# Create logs directory if it doesn't exist
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/probir_mitmproxy.log"

if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    sudo mkdir -p "$LOG_DIR"
    # Ensure mitmproxyuser can write to it. 
    # This assumes mitmproxyuser exists. If running as current user, this chown might not be needed or different.
    # For simplicity, if sudo is used for mitmdump, sudo for chown is consistent.
    sudo chown mitmproxyuser "$LOG_DIR" 
fi
# Ensure the log file can be written to by mitmproxyuser, even if it exists from a previous root run
sudo touch "$LOG_FILE"
sudo chown mitmproxyuser "$LOG_FILE"

# Run mitmdump as mitmproxyuser using the absolute path in their home directory
echo "Starting proxy-based interaction recorder. Traffic logged to DB. Mitmproxy & addon logs to: $LOG_FILE"
sudo -u mitmproxyuser bash -c "cd ~ && \"$MITMDUMP_PATH\" --showhost -s /opt/mitmproxy_scripts/probir.py -q >> \"$LOG_FILE\" 2>&1"
