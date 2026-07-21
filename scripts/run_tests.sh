#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
source .ai/bin/activate
pytest
