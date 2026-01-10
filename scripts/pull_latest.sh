#!/bin/bash
# Pull latest code on RunPod instance
# Replaces /app with latest from GitHub

git clone --depth 1 https://github.com/rajs96/music_audio_analyzer.git /tmp/repo && cp -r /tmp/repo/* /app/ && rm -rf /tmp/repo

echo "Done! Latest repo pulled from GitHub."
