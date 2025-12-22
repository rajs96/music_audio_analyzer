#!/bin/bash

# Script to copy audio files to a remote server
# Usage: ./copy_audio_files.sh <hostname_or_ip>
# Example: ./copy_audio_files.sh 192.222.55.13

if [ $# -eq 0 ]; then
    echo "Error: Hostname or IP address required"
    echo "Usage: $0 <hostname_or_ip>"
    echo "Example: $0 192.222.55.13"
    exit 1
fi

HOSTNAME=$1
SSH_KEY="$HOME/.ssh/test_qwen"
BASE_DIR="/Users/rajsingh/Desktop/code/music_audio_analyzer"
AUDIO_FILES_DIR="${BASE_DIR}/audio_files"
SRC_DIR="${BASE_DIR}/src"
NOTEBOOKS_DIR="${BASE_DIR}/notebooks"
DEST_DIR="/home/ubuntu/"

echo "Copying audio files to ubuntu@${HOSTNAME}..."
scp -i "$SSH_KEY" -r "$AUDIO_FILES_DIR" "ubuntu@${HOSTNAME}:${DEST_DIR}"

if [ $? -eq 0 ]; then
    echo "Successfully copied audio files to ${HOSTNAME}"
else
    echo "Error: Failed to copy audio files to ${HOSTNAME}"
    exit 1
fi

echo "Copying src folder to ubuntu@${HOSTNAME}..."
scp -i "$SSH_KEY" -r "$SRC_DIR" "ubuntu@${HOSTNAME}:${DEST_DIR}"

if [ $? -eq 0 ]; then
    echo "Successfully copied src folder to ${HOSTNAME}"
else
    echo "Error: Failed to copy src folder to ${HOSTNAME}"
    exit 1
fi

echo "Copying notebooks folder to ubuntu@${HOSTNAME}..."
scp -i "$SSH_KEY" -r "$NOTEBOOKS_DIR" "ubuntu@${HOSTNAME}:${DEST_DIR}"

if [ $? -eq 0 ]; then
    echo "Successfully copied notebooks folder to ${HOSTNAME}"
else
    echo "Error: Failed to copy notebooks folder to ${HOSTNAME}"
    exit 1
fi

echo "All files copied successfully to ${HOSTNAME}"

