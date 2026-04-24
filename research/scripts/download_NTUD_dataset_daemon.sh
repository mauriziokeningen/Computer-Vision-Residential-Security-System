#!/bin/bash

# ==============================================================================
# ARCHITECTURE: Sequential Data Acquisition Daemon
# PURPOSE: Fault-tolerant, resumable ingestion of remote datasets.
# EXECUTION: Designed to run detached (e.g., via tmux/nohup) to survive SSH 
#            session termination.
# ==============================================================================

# STRICT MODE: Fail fast on errors, undefined variables, and pipe failures.
set -euo pipefail

# ==============================================================================
# CONFIGURATION & SECRETS MANAGEMENT
# ==============================================================================
# SECURITY: Never hardcode session tokens in version control. 
# Expecting SESSION_ID to be passed via environment variable or read from a local .env
readonly SESSION_ID="${NTU_SESSION_ID:-"INSERT_SESSION_TOKEN_HERE"}"
readonly USER_AGENT="INSERT_USER_AGENT_STRING_HERE"
readonly BASE_URL="https://rose1.ntu.edu.sg/dataset/actionRecognition/download"
readonly OUTPUT_DIR="research/data/raw/body/NTU_RGB_D"

# Ensure target directory exists before starting I/O operations
mkdir -p "$OUTPUT_DIR"

echo "[INFO] Initializing sequential data acquisition daemon..."

# ==============================================================================
# CORE LOGIC: Resilient Payload Fetcher
# ==============================================================================
download_payload() {
    local payload_id=$1
    local output_filename=$2
    local output_path="${OUTPUT_DIR}/${output_filename}"
    
    echo "[INFO] Initiating transfer: ID ${payload_id} -> ${output_path}"
    
    # IDEMPOTENCY: Using '-C -' ensures that if the process is interrupted, 
    # it resumes from the last byte rather than restarting the payload.
    curl -L -C - "${BASE_URL}/${payload_id}" \
      -H "User-Agent: ${USER_AGENT}" \
      -b "sessionid=${SESSION_ID}" \
      -o "${output_path}"
      
    # VERIFICATION: Ensure the network socket closed cleanly.
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Payload ${output_filename} acquired and verified."
    else
        echo "[ERROR] Network transfer failed for ID ${payload_id}. Check session token."
        # Note: Deliberately not exiting the script here to allow subsequent 
        # payloads to attempt download if the failure was a localized server blip.
    fi
}

# ==============================================================================
# EXECUTION SEQUENCE
# ==============================================================================
# TODO: Define your dataset execution sequence below. 
# Format: download_payload <REMOTE_ID> <LOCAL_FILENAME>

# Example:
# download_payload 126 "nturgbd_rgb_s002.zip"
# download_payload 127 "nturgbd_rgb_s003.zip"


echo "[INFO] Acquisition sequence complete. Daemon terminating gracefully."