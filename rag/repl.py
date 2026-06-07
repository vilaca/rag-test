"""Interactive REPL for querying the RAG system."""

from rag.rag_system import RAGSystem, print_retrieval_debug


def repl(rag: RAGSystem):
    """Simple REPL for querying the RAG system with improved UX."""
    print("RAG System REPL. Type 'exit' to quit.")
    if hasattr(rag, "original_files") and len(rag.original_files) > 1:
        print(
            f"Loaded {len(rag.chunks)} chunks from {len(rag.original_files)} files: {', '.join(rag.original_files)}"
        )
    else:
        print(f"Loaded {len(rag.chunks)} chunks from '{rag.content_path}'")

    while True:
        try:
            query = input("\n>>> ")
            if query.lower() == "exit":
                break

            if not query.strip():
                continue

            print("Processing...", end="\r")
            answer = rag.query(query)
            print(" " * 50, end="\r")

            if getattr(rag, "debug_retrieval", False):
                print_retrieval_debug(rag)
            print(f"\nQuestion: {query}\nAnswer: {answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
