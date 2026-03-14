#!/bin/bash

# Train a skipgram model on the input corpus
./fasttext skipgram -input big.txt -output model

# Generate sentence vectors from the trained model
echo "there was no secret marriage" | ./fasttext print-sentence-vectors model.bin
