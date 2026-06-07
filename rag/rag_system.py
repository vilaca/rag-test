"""Core RAG system type composed from smaller modules."""

from typing import List

from sentence_transformers import SentenceTransformer

from rag.rag_answering import AnsweringMixin
from rag.rag_debug import print_retrieval_debug
from rag.rag_retrieval import RetrievalMixin


class RAGSystem(RetrievalMixin, AnsweringMixin):
    def __init__(
        self,
        content_path: str,
        model_name: str = "BAAI/bge-large-en-v1.5",
        use_mmap_index: bool = False,
        index_file: str = "index.faiss",
        debug_retrieval: bool = False,
    ):
        """Initialize the RAG system with content and embedding model."""
        self.content_path = content_path
        self.model_name = model_name
        # Use the specified model or default to NV-Embed-v2
        embedding_model_name = model_name if model_name else "nvidia/NV-Embed-v2"
        
        # Some models require trust_remote_code
        trust_remote_code = "gte-large" in embedding_model_name.lower()
        
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            trust_remote_code=trust_remote_code
        )
        self.chunks: List[str] = []
        self.embeddings = None
        self.index = None
        self.bm25 = None
        self.reranker = None
        self.content = ""
        self.original_files = [content_path]
        self.use_mmap_index = use_mmap_index
        self.index_file = index_file
        self.debug_retrieval = debug_retrieval
        self.last_retrieval_debug = []
        self.chunk_metadata = []  # Store hierarchical metadata
        self._init_reranker()

    def _init_reranker(self):
        """Initialize cross-encoder reranker if available."""
        try:
            from FlagEmbedding import FlagReranker
            # Use a smaller reranker model that's more practical for CLI use
            self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False)
            print("✅ Cross-encoder reranker initialized")
        except ImportError:
            print("⚠️  Cross-encoder reranker not available (install FlagEmbedding for better results)")
            self.reranker = None
        except Exception as e:
            print(f"⚠️  Failed to initialize reranker: {e}")
            self.reranker = None

        # Initialize Mistral/Devstral for generation
        self.generator = self.init_generator()

    def load_content(self):
        """Load content from a file with validation."""
        try:
            with open(self.content_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                raise ValueError("Input file is empty")

            if len(content) < 100:
                print("Warning: Input file is very small (< 100 characters)")

            self.content = content
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(self.content_path, "r", encoding="latin-1") as f:
                content = f.read()
            self.content = content
        except Exception as e:
            raise RuntimeError(f"Failed to load content file: {str(e)}")

        # Validate the content
        if not self.content.strip():
            raise ValueError("Content is empty")


__all__ = ["RAGSystem", "print_retrieval_debug"]
