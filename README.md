# FastText Sentence2Vec

Generate sentence-level vector representations using a FastText skipgram model. This project trains word embeddings on a text corpus with Facebook's [fastText](https://github.com/facebookresearch/fastText) and extends them to produce sentence vectors (sentence2vec) by averaging normalized word vectors. Includes an attention-based encoder-decoder notebook for sequence modeling experiments.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| 🧠 Embeddings | [FastText](https://fasttext.cc/) (skipgram) |
| 💻 Core Engine | C++ 11 (pthreads, POSIX) |
| 🐍 Evaluation | Python 3 (NumPy, SciPy) |
| 📓 Notebook | Jupyter / Google Colab |
| 🤖 Deep Learning | Keras + TensorFlow (attention decoder) |

---

## 📦 Dependencies

**C++ (fastText core):**
- A C++11-compatible compiler (`g++` or `clang++`)
- `make`
- pthreads (included on Linux/macOS)

**Python (evaluation & notebook):**
- Python 3.6+
- `numpy`
- `scipy`
- `keras` (for the attention notebook)
- `tensorflow` (Keras backend)
- `pandas`, `matplotlib` (notebook utilities)

```bash
pip install numpy scipy keras tensorflow pandas matplotlib
```

---

## 🚀 How to Run

### 1. Build fastText

```bash
cd fastText
make
```

### 2. Train a skipgram model

Place your training corpus as `big.txt` inside the `fastText/` directory, then:

```bash
cd fastText
./fasttext skipgram -input big.txt -output model
```

This produces `model.bin` (binary model) and `model.vec` (word vectors).

### 3. Generate sentence vectors

```bash
echo "there was no secret marriage" | ./fasttext print-sentence-vectors model.bin
```

Or use the helper script:

```bash
bash sentence2vec.sh
```

### 4. Evaluate word vectors

```bash
python eval.py -m model.vec -d <path-to-similarity-dataset>
```

The evaluation computes Spearman correlation against a gold-standard word similarity dataset.

### 5. Attention Networks notebook

Open `Attention_Networks.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) to experiment with the attention-based encoder-decoder model.

---

## ⚠️ Known Issues

- The Attention Networks notebook uses legacy Keras APIs (`keras.layers.recurrent.Recurrent`, `keras.engine.InputSpec`, `_time_distributed_dense`) that are removed in Keras 2.x+. It runs on Keras ≤ 1.x / early 2.0 or requires migration to `tf.keras`.
- The `model.vec` and compiled `.o` / binary files were previously committed to the repo. A `.gitignore` has been added to prevent this going forward.
- Training on large corpora requires significant memory and CPU time. Adjust `-thread`, `-dim`, and `-epoch` flags as needed.
- The `eval.py` script expects a tab/space-separated similarity dataset with format: `word1 word2 score`.

---

## 📄 License

The fastText library is BSD-licensed. See [fastText/LICENSE](fastText/LICENSE) and [fastText/PATENTS](fastText/PATENTS) for details.
