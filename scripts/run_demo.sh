#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .ai/bin/activate
python3 -m demos.conversation_demo
