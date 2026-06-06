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

Use `main.py` and pass one or more text/subtitles file paths.

Full command example using all files in `dataset/`:

```bash
python3 main.py dataset/muffin.txt dataset/muffin-2.txt dataset/muffin-3.txt dataset/muffin-4.txt dataset/muffin-5.txt
```

You will enter an interactive prompt:

```text
RAG System REPL. Type 'exit' to quit.
>>> what is this text about?
```

Type `exit` to quit.

### Command-Line Arguments

- **`subtitles`** (required): One or more paths to input text/subtitles files.
  Example: `python3 main.py /path/to/file1.txt /path/to/file2.txt`

- **`--model`** (optional): Text generation model to use.
  Default: `gpt2`
  Example: `python3 main.py dataset/muffin.txt --model distilgpt2`

- **`--embedding-model`** (optional): Embedding model for vector search.
  Default: `sentence-transformers/all-MiniLM-L6-v2`
  Example: `python3 main.py dataset/muffin.txt --embedding-model sentence-transformers/all-mpnet-base-v2`

## Using your own file(s)

Run with any `.txt` file(s):

```bash
python3 main.py /path/to/your/subtitles1.txt /path/to/your/subtitles2.txt
```

## Dataset

The dataset used in this repository is taken from the YouTube channel **Lab Muffin Beauty Science**:
https://www.youtube.com/@LabMuffinBeautyScience/videos

## Notes

- The app attempts to load `mistralai/Mistral-7B-Instruct-v0.1` for generation.
- If that fails, it falls back to `distilgpt2`.
- First run can take a while due to model downloads.
