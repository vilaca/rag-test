"""File ingestion and RAG initialization helpers."""

import os
from typing import Callable, List

from rag_system import RAGSystem


def _read_text_file(path: str) -> str:
    """Read text file with UTF-8 fallback to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def initialize_rag_from_files(
    subtitles_paths: List[str],
    embedding_model: str,
    generation_model: str = "distilgpt2",
    use_mmap_index: bool = False,
    index_file: str = "index.faiss",
    debug_retrieval: bool = False,
    logger: Callable[[str], None] = print,
) -> RAGSystem:
    """Validate, ingest, and initialize a RAGSystem from one or more text files."""
    for subtitles_path in subtitles_paths:
        if not os.path.exists(subtitles_path):
            raise FileNotFoundError(f"Subtitles file '{subtitles_path}' not found.")

    combined_parts = []
    for subtitles_path in subtitles_paths:
        content = _read_text_file(subtitles_path)
        if content.strip():
            combined_parts.append(content)
        else:
            logger(f"Warning: '{subtitles_path}' is empty and will be skipped.")

    if not combined_parts:
        raise ValueError("All provided files are empty.")

    combined_text = "\n\n".join(combined_parts)

    rag = RAGSystem(
        subtitles_paths[0],
        model_name=embedding_model,
        use_mmap_index=use_mmap_index,
        index_file=index_file,
        debug_retrieval=debug_retrieval,
    )
    rag.original_files = subtitles_paths
    rag.subtitles = combined_text

    if not rag.subtitles.strip():
        raise ValueError("Subtitles content is empty")

    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()

    if generation_model != "distilgpt2":
        rag.generator = rag.init_generator(model_id=generation_model)

    return rag
