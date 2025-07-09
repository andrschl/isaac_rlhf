#!/bin/bash
# api_key_importer.sh

# Get the directory where the script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY_FILE="${SCRIPT_DIR}/.env.api_keys"

if [ -f "$KEY_FILE" ]; then
    echo "Loading environment variables from ${KEY_FILE}"
    while IFS= read -r line || [ -n "$line" ]; do
        trimmed=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ -z "$trimmed" || "$trimmed" == \#* ]]; then
            continue
        fi
        key=$(echo "$trimmed" | cut -d '=' -f1)
        value=$(echo "$trimmed" | cut -d '=' -f2-)
        export "$key=$value"
    done < "$KEY_FILE"
else
    echo "WARNING: ${KEY_FILE} not found. No API keys are loaded."
fi
