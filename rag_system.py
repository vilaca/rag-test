"""Core RAG system type composed from smaller modules."""

from typing import List

from sentence_transformers import SentenceTransformer

from rag_answering import AnsweringMixin
from rag_debug import print_retrieval_debug
from rag_retrieval import RetrievalMixin


class RAGSystem(RetrievalMixin, AnsweringMixin):
    def __init__(
        self,
        subtitles_path: str,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        use_mmap_index: bool = False,
        index_file: str = "index.faiss",
        debug_retrieval: bool = False,
    ):
        """Initialize the RAG system with subtitles and embedding model."""
        self.subtitles_path = subtitles_path
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings = None
        self.index = None
        self.subtitles = ""
        self.original_files = [subtitles_path]
        self.use_mmap_index = use_mmap_index
        self.index_file = index_file
        self.debug_retrieval = debug_retrieval
        self.last_retrieval_debug = []

        # Initialize Mistral/Devstral for generation
        self.generator = self.init_generator()

    def load_subtitles(self):
        """Load subtitles from a file with validation."""
        try:
            with open(self.subtitles_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                raise ValueError("Input file is empty")

            if len(content) < 100:
                print("Warning: Input file is very small (< 100 characters)")

            self.subtitles = content
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(self.subtitles_path, "r", encoding="latin-1") as f:
                content = f.read()
            self.subtitles = content
        except Exception as e:
            raise RuntimeError(f"Failed to load subtitles file: {str(e)}")

        # Validate the content
        if not self.subtitles.strip():
            raise ValueError("Subtitles content is empty")


__all__ = ["RAGSystem", "print_retrieval_debug"]
