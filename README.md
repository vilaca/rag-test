# RAG Test CLI

A lightweight **Retrieval-Augmented Generation (RAG)** command-line app for asking questions over one or more local text files.

It:
- loads input `.txt` files,
- splits them into overlapping chunks,
- generates embeddings with `sentence-transformers`,
- builds a FAISS similarity index,
- and answers questions interactively in a REPL.

---

## Requirements

- Python **3.9+**
- Internet connection on first run (to download Hugging Face models)

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start

Run the CLI with one or more text files:

```bash
python3 main.py ./notes.txt
```

or

```bash
python3 main.py ./doc1.txt ./doc2.txt ./doc3.txt
```

Then ask questions in the prompt:

```text
>>> what is the main topic?
>>> summarize the key points
>>> exit
```

Type `exit` (or press `Ctrl+C`) to quit.

---

## Command-Line Options

```bash
python3 main.py <file1.txt> [file2.txt ...] [options]
```

### Positional arguments
- `subtitles` (required): one or more input text file paths.

### Optional arguments
- `--model <model_id>`
  - Text generation model (default: `distilgpt2`).
- `--embedding-model <model_id>`
  - Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`).
- `--mmap-index`
  - Memory-map the FAISS index from disk to reduce RAM usage.
- `--index-file <path>`
  - FAISS index path when using `--mmap-index` (default: `index.faiss`).
- `--debug-retrieval`
  - Print retrieved chunks and FAISS scores per query.

### Example with options

```bash
python3 main.py ./dataset.txt \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --model distilgpt2 \
  --debug-retrieval
```

---

## Notes

- Empty files are skipped with a warning.
- If all provided files are empty, initialization fails.
- UTF-8 is used by default, with a latin-1 fallback for reading files.

---

## Dependencies

Defined in `requirements.txt`:
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
