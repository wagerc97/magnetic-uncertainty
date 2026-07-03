#!/usr/bin/bash 
set -e

# setup.sh 
# Setup git repository for magnetic uncertainty study 
# 1. Load git submodules 
HERE="$(cd "$(dirname "$0")" && pwd)"  
echo "Project root directory: $HERE"

# 1. Load git submodules
git submodule update --remote --recursive
echo "" 
echo "Git submodules loaded successfully."

echo ""
echo "Setup complete. You can now proceed with your magnetic uncertainty study."