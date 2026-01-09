#!/bin/bash
# Pull latest code on RunPod instance
# Replaces /app/src with latest from GitHub

cd /app && rm -rf src && git clone --depth 1 https://github.com/rajs96/music_audio_analyzer.git /tmp/repo && cp -r /tmp/repo/src . && rm -rf /tmp/repo

echo "Done! Latest src/ pulled from GitHub."
