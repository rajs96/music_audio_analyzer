#!/bin/bash
# Pull latest code on RunPod instance
# Usage: ./pull_latest.sh [path_to_repo]

REPO_PATH="${1:-/workspace/music_audio_analyzer}"

cd "$REPO_PATH" || { echo "ERROR: Directory $REPO_PATH not found"; exit 1; }

echo "Pulling latest code in $REPO_PATH..."
git fetch origin
git reset --hard origin/main

echo "Done! Latest code pulled."
