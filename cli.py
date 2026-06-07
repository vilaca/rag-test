"""CLI and REPL for the RAG system."""

import argparse
import os

from rag_system import RAGSystem, print_retrieval_debug


def repl(rag: RAGSystem):
    """Simple REPL for querying the RAG system with improved UX."""
    print("RAG System REPL. Type 'exit' to quit.")
    if hasattr(rag, "original_files") and len(rag.original_files) > 1:
        print(
            f"Loaded {len(rag.chunks)} chunks from {len(rag.original_files)} files: {', '.join(rag.original_files)}"
        )
    else:
        print(f"Loaded {len(rag.chunks)} chunks from '{rag.subtitles_path}'")

    while True:
        try:
            query = input(">>> ")
            if query.lower() == "exit":
                break

            if not query.strip():
                continue

            # Show thinking indicator
            print("Processing...", end="\r")

            answer = rag.query(query)

            # Clear thinking indicator
            print(" " * 50, end="\r")
            if getattr(rag, "debug_retrieval", False):
                print_retrieval_debug(rag)
            print(f"\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def _read_text_file(path: str) -> str:
    """Read text file with UTF-8 fallback to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def main():
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
    args = parser.parse_args()

    # Check if all files exist
    for subtitles_path in args.subtitles:
        if not os.path.exists(subtitles_path):
            print(f"Error: Subtitles file '{subtitles_path}' not found.")
            return

    try:
        print("Initializing RAG system with:")
        print(f"  - Input files: {', '.join(args.subtitles)}")
        print(f"  - Embedding model: {args.embedding_model}")
        print(f"  - Generation model: {args.model}")
        print(f"  - Memory-mapped index: {'enabled' if args.mmap_index else 'disabled'}")
        print(f"  - Debug retrieval: {'enabled' if args.debug_retrieval else 'disabled'}")
        if args.mmap_index:
            print(f"  - Index file: {args.index_file}")

        # Load and combine multiple files
        combined_parts = []
        for subtitles_path in args.subtitles:
            content = _read_text_file(subtitles_path)
            if content.strip():
                combined_parts.append(content)
            else:
                print(f"Warning: '{subtitles_path}' is empty and will be skipped.")

        if not combined_parts:
            raise ValueError("All provided files are empty.")

        combined_text = "\n\n".join(combined_parts)

        # Create RAG system with combined content
        rag = RAGSystem(
            args.subtitles[0],
            model_name=args.embedding_model,
            use_mmap_index=args.mmap_index,
            index_file=args.index_file,
            debug_retrieval=args.debug_retrieval,
        )  # Use first file path as identifier
        rag.original_files = args.subtitles
        # Use already-loaded combined content
        rag.subtitles = combined_text
        if not rag.subtitles.strip():
            raise ValueError("Subtitles content is empty")

        # Preprocess subtitles
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()

        # Override the generator model only when different from init default
        if args.model != "distilgpt2":
            rag.generator = rag.init_generator(model_id=args.model)

        repl(rag)
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
