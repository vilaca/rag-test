"""Argument parsing for the RAG CLI."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RAG application."""
    parser = argparse.ArgumentParser(description="RAG System for YouTube Subtitles")
    parser.add_argument("subtitles", nargs="+", type=str, help="Path(s) to the subtitles file(s)")
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Text generation model to use (default: distilgpt2)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--mmap-index",
        action="store_true",
        help="Memory-map the FAISS index from disk to reduce RAM usage",
    )
    parser.add_argument(
        "--index-file",
        type=str,
        default="index.faiss",
        help="Path for FAISS index file when using --mmap-index (default: index.faiss)",
    )
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print retrieved chunks with FAISS scores for each query",
    )
    return parser.parse_args()
