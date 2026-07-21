#!/usr/bin/env bash

set -e

PROJECT_DIR="$HOME/Jay_AI_Companion_Scaffold"
VENV_DIR="$PROJECT_DIR/.ai"

cd "$PROJECT_DIR"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment not found:"
    echo "$VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

exec python3 main.py --mode voice