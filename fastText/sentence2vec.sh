#!/usr/bin/env bash
#
# Train a skipgram model and generate sentence vectors.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "${SCRIPT_DIR}/fasttext" ]; then
    echo "Error: fasttext binary not found. Run 'make' first." >&2
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/big.txt" ]; then
    echo "Error: big.txt corpus not found in ${SCRIPT_DIR}." >&2
    exit 1
fi

# Train skipgram model
"${SCRIPT_DIR}/fasttext" skipgram -input "${SCRIPT_DIR}/big.txt" -output "${SCRIPT_DIR}/model"

# Generate sentence vectors
echo "there was no secret marriage" | "${SCRIPT_DIR}/fasttext" print-sentence-vectors "${SCRIPT_DIR}/model.bin"
