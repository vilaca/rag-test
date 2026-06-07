"""CLI orchestration for the RAG system."""

from cli_args import parse_args
from ingestion import initialize_rag_from_files
from repl import repl


def main():
    args = parse_args()

    try:
        print("Initializing RAG system with:")
        print(f"  - Input files: {', '.join(args.subtitles)}")
        print(f"  - Embedding model: {args.embedding_model}")
        print(f"  - Generation model: {args.model}")
        print(f"  - Memory-mapped index: {'enabled' if args.mmap_index else 'disabled'}")
        print(f"  - Debug retrieval: {'enabled' if args.debug_retrieval else 'disabled'}")
        if args.mmap_index:
            print(f"  - Index file: {args.index_file}")

        rag = initialize_rag_from_files(
            subtitles_paths=args.subtitles,
            embedding_model=args.embedding_model,
            generation_model=args.model,
            use_mmap_index=args.mmap_index,
            index_file=args.index_file,
            debug_retrieval=args.debug_retrieval,
            logger=print,
        )

        repl(rag)
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
