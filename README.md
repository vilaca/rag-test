# RAG Test CLI

A lightweight **Retrieval-Augmented Generation (RAG)** command-line app for asking questions over one or more local text files.

## Architecture Overview

The system uses an enhanced RAG architecture with the following components:

### 1. **Hybrid Retrieval System**
- **Dense Retrieval**: FAISS vector similarity search using embeddings
- **Sparse Retrieval**: BM25 keyword-based search
- **Combined Results**: Reciprocal Rank Fusion for optimal coverage

### 2. **Advanced Embedding Models**
- **Primary**: `BAAI/bge-large-en-v1.5` for state-of-the-art semantic understanding
- **Fallback**: `sentence-transformers/multi-qa-mpnet-base-dot-v1` for compatibility

### 3. **Intelligent Chunking**
- Semantic-aware chunking that respects sentence/paragraph boundaries
- Dynamic chunk sizing based on content structure
- Overlap preservation for context continuity

### 4. **Specialized Answer Synthesis**
- Question-type specific handling (definitions, explanations, overviews)
- Comprehensive answer generation with source attribution
- Multi-sentence synthesis for complex queries

### 5. **Relevance Filtering**
- Query-specific relevance scoring
- Semantic relatedness checking
- Minimum relevance thresholds to filter noise

## Processing Pipeline

1. **Content Ingestion**: Load and preprocess text files
2. **Semantic Chunking**: Split content into meaningful units
3. **Hybrid Embedding**: Generate both dense and sparse representations
4. **Index Construction**: Build optimized search structures
5. **Query Processing**: Hybrid retrieval with relevance filtering
6. **Answer Synthesis**: Type-specific generation with post-processing
7. **Response Delivery**: Formatted answers with source citations

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
  - Embedding model (default: `BAAI/bge-large-en-v1.5`).
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

## Architecture Improvements

### Enhanced Retrieval
- **Hybrid Approach**: Combines FAISS (dense) and BM25 (sparse) retrieval
- **Reciprocal Rank Fusion**: Intelligent combination of both retrieval methods
- **Relevance Filtering**: Query-specific scoring to eliminate off-topic results

### Advanced Answer Generation
- **Specialized Handlers**: Dedicated methods for different question types
- **Comprehensive Synthesis**: Multi-sentence answers with context combination
- **Source Attribution**: Automatic citation of source material

### Robust Processing
- **Semantic Chunking**: Preserves meaningful content units
- **Intelligent Filtering**: Minimum relevance thresholds
- **Fallback Strategies**: Graceful degradation when perfect matches aren't found

## Dependencies

Defined in `requirements.txt`:
- `torch`
- `transformers`
- `sentence-transformers`
- `rank-bm25` (for hybrid retrieval)
- `faiss-cpu`
