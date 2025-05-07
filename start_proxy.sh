#!/bin/bash

# Script to start the mitmproxy proxy-based interaction recorder

# Copy the addon script to the mitmproxy scripts directory
sudo cp probir.py /opt/mitmproxy_scripts/probir.py

# Find the absolute path of mitmdump for the current user
MITMDUMP_PATH=$(which mitmdump)

# Check if mitmdump was found
if [ -z "$MITMDUMP_PATH" ]; then
    echo "Error: mitmdump not found in the current user's PATH. Please ensure it is installed and in your PATH."
    exit 1
fi

# Run mitmdump as mitmproxyuser using the absolute path in their home directory
echo "Starting proxy-based interaction recorder and logging traffic to /mnt/wsl_data/filtered_traffic_log.db..."
sudo -u mitmproxyuser bash -c "cd ~ && \"$MITMDUMP_PATH\" --showhost -s /opt/mitmproxy_scripts/probir.py -q"
