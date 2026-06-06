# RAG Test

This project is a simple **RAG (Retrieval-Augmented Generation)** CLI that:

1. Loads a subtitles/text file
2. Splits it into chunks
3. Builds embeddings + a FAISS index
4. Lets you ask questions in a terminal REPL

## Requirements

- Python 3.9+
- Internet connection (first run downloads models from Hugging Face)

## Setup

Create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Note: `main.py` also tries to install missing runtime packages automatically (`faiss-cpu`, `sentence-transformers`, `transformers`, `torch`) if they are not already installed.

## Run

Use `main.py` and pass a text/subtitles file path:

```bash
python main.py muffin.txt
```

You will enter an interactive prompt:

```text
RAG System REPL. Type 'exit' to quit.
>>> what is this text about?
```

Type `exit` to quit.

### Command-Line Arguments

- **`subtitles`** (required): Path to the input text/subtitles file.
  Example: `python main.py /path/to/your/subtitles.txt`

- **`--model`** (optional): Text generation model to use.
  Default: `gpt2`
  Example: `python main.py muffin.txt --model distilgpt2`

- **`--embedding-model`** (optional): Embedding model for vector search.
  Default: `sentence-transformers/all-MiniLM-L6-v2`
  Example: `python main.py muffin.txt --embedding-model sentence-transformers/all-mpnet-base-v2`

## Using your own file

Run with any `.txt` file:

```bash
python main.py /path/to/your/subtitles.txt
```

## Notes

- The app attempts to load `mistralai/Mistral-7B-Instruct-v0.1` for generation.
- If that fails, it falls back to `distilgpt2`.
- First run can take a while due to model downloads.
