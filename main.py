#!/usr/bin/env python3
"""Entry point for the RAG CLI."""

import logging
import sys
from rag.cli_args import parse_args
from rag.ingestion import initialize_rag_from_files
from rag.repl import repl


def configure_logging():
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def validate_args(args):
    """Validate the CLI arguments."""
    if not args.files:
        raise ValueError("No input files provided. Use --files to specify input files.")
    
    if args.mmap_index and not args.index_file:
        raise ValueError("Memory-mapped index enabled but no index file specified. Use --index-file to specify the index file.")


def main():
    """Main entry point for the RAG CLI."""
    logger = configure_logging()
    
    try:
        args = parse_args()
        validate_args(args)
        
        logger.info("Initializing RAG system with:")
        logger.info(f"  - Input files: {', '.join(args.files)}")
        logger.info(f"  - Embedding model: {args.embedding_model}")
        logger.info(f"  - Generation model: {args.model}")
        logger.info(f"  - Memory-mapped index: {'enabled' if args.mmap_index else 'disabled'}")
        logger.info(f"  - Debug retrieval: {'enabled' if args.debug_retrieval else 'disabled'}")
        if args.mmap_index:
            logger.info(f"  - Index file: {args.index_file}")

        rag = initialize_rag_from_files(
            file_paths=args.files,
            embedding_model=args.embedding_model,
            generation_model=args.model,
            use_mmap_index=args.mmap_index,
            index_file=args.index_file,
            debug_retrieval=args.debug_retrieval,
            logger=logger,
        )

        repl(rag)
        
    except ValueError as ve:
        logger.error(f"Invalid arguments: {str(ve)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        logger.debug("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
