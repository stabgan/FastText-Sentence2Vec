# FastText Sentence2Vec

Generate sentence-level vector representations using Facebook's [fastText](https://github.com/facebookresearch/fastText) skip-gram model, with an additional Bahdanau-style attention decoder notebook for sequence memorisation experiments.

## What It Does

1. **Sentence vectors** — trains a skip-gram word-embedding model on a text corpus, then averages (L2-normalised) word vectors to produce a single vector per sentence via `print-sentence-vectors`.
2. **Attention decoder** — the included Colab notebook (`Attention_Networks.ipynb`) compares a vanilla encoder-decoder LSTM against one augmented with Bahdanau attention on a toy sequence-memorisation task.

## Repository Structure

```
.
├── Attention_Networks.ipynb   # Colab notebook (attention vs baseline LSTM)
└── fastText/
    ├── src/                   # C++ fastText source (skipgram, cbow, etc.)
    ├── eval.py                # Word-vector evaluation (Spearman correlation)
    ├── sentence2vec.sh        # Quick-start: train + generate sentence vectors
    ├── Makefile / CMakeLists.txt
    ├── setup.py               # Python bindings (pybind11)
    └── ...
```

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| 🧠 Embeddings | fastText (C++) |
| 🐍 Evaluation | Python 3, NumPy, SciPy |
| 🔥 Attention notebook | TensorFlow / Keras |
| 🔧 Build | Make, CMake ≥ 3.1, C++11 |

## Getting Started

### Build fastText

```bash
cd fastText
make          # or: mkdir build && cd build && cmake .. && make
```

### Train & Query Sentence Vectors

```bash
cd fastText
./fasttext skipgram -input big.txt -output model
echo "there was no secret marriage" | ./fasttext print-sentence-vectors model.bin
```

### Evaluate Word Vectors

```bash
python fastText/eval.py -m vectors.txt -d rw.txt
```

### Run the Attention Notebook

Open `Attention_Networks.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter and run all cells. Requires `tensorflow >= 2.x`.

## Dependencies

- **C++ compiler** with C++11 support (GCC ≥ 4.8 / Clang ≥ 3.3)
- **Python ≥ 3.7**
- `numpy`, `scipy` (for `eval.py`)
- `tensorflow >= 2.x` (for the notebook)

## ⚠️ Known Issues

- The `fastText/` subtree is a modified fork of Facebook's fastText; it may not compile on all platforms without adjustments to the Makefile flags (e.g. `-march=native`).
- The Attention notebook's vanilla-LSTM regressor cell is commented out as a template — fill in your own data before running.
- Pre-compiled `.o` object files and the `fasttext` binary are checked into the repo; they are platform-specific and should be rebuilt locally.

## License

See [LICENSE](fastText/LICENSE) (BSD).
