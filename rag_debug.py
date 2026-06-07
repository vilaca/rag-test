"""Debug utilities for RAG retrieval output."""


def print_retrieval_debug(rag, preview_chars: int = 180):
    """Print retrieval ranking details for the last query."""
    debug_rows = getattr(rag, "last_retrieval_debug", [])
    if not debug_rows:
        print("[debug-retrieval] No retrieval results.")
        return

    print("[debug-retrieval] Top retrieved chunks:")
    for row in debug_rows:
        chunk_preview = row["chunk"].replace("\n", " ").strip()
        if len(chunk_preview) > preview_chars:
            chunk_preview = chunk_preview[:preview_chars].rstrip() + "..."
        print(f"  #{row['rank']:02d} idx={row['index']} score={row['score']:.4f} | {chunk_preview}")
