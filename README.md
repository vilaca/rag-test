# RAG Test

This project is a simple **RAG (Retrieval-Augmented Generation)** CLI that:

1. Loads one or more text files
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

Use `main.py` and pass one or more text file paths.

Full command example using all files in `dataset/`:

```bash
python3 main.py dataset/muffin.txt dataset/muffin-2.txt dataset/muffin-3.txt dataset/muffin-4.txt dataset/muffin-5.txt
```

You will enter an interactive prompt:

```text
RAG System REPL. Type 'exit' to quit.
>>> what is this text about?

>>> When to use sunscreen?
Answer: Use sunscreen every day on exposed skin during daylight. Apply as the last skincare step before makeup, ideally 15 minutes before sun exposure.

>>> Is sunscreen necessary if it is cloudy?
Answer: Yes. Use sunscreen even when it's cloudy—UVA still reaches skin through clouds. Apply in the morning, then reapply about every 2 hours when outdoors, and after swimming, sweating, or towel-drying.
```

Type `exit` to quit.

### Command-Line Arguments

- **`input_files`** (required): One or more paths to input text files.
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
python3 main.py /path/to/your/text1.txt /path/to/your/text2.txt
```

## Dataset

The dataset used in this repository is taken from the YouTube channel **Lab Muffin Beauty Science**:
https://www.youtube.com/@LabMuffinBeautyScience/videos

## Notes

- The app attempts to load `mistralai/Mistral-7B-Instruct-v0.1` for generation.
- If that fails, it falls back to `distilgpt2`.
- First run can take a while due to model downloads.
