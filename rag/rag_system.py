"""Core RAG system type composed from smaller modules."""

from typing import List, Dict
import time
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

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
        # Use the specified model or default to gte-large-en-v1.5
        embedding_model_name = model_name if model_name else "Alibaba-NLP/gte-large-en-v1.5"
        
        # Some models require trust_remote_code
        trust_remote_code = ("gte-large" in embedding_model_name.lower() or 
                           "nv-embed" in embedding_model_name.lower())
        
        try:
            # Try to load the model with timeout
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            if "nv-embed" in embedding_model_name.lower():
                print(f"⚠️  NV-Embed-v2 failed to load: {e}")
                print("   Falling back to BAAI/bge-large-en-v1.5...")
                self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            else:
                raise RuntimeError(f"Failed to load embedding model {embedding_model_name}: {e}")
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
        self.chunk_registry = {}  # Map: chunk_id -> (chunk_text, metadata, embedding)
        self.next_chunk_id = 0  # Counter for incremental updates
        self._init_reranker()

    def add_document(self, document_text: str, document_id: str = None):
        """Add a document incrementally without rebuilding the entire index."""
        if not document_id:
            document_id = f"doc_{len(self.original_files) + 1}"
        
        # Store original document
        self.original_files.append(document_id)
        
        # Process document into chunks
        new_chunks = self._process_document_to_chunks(document_text)
        
        # Add chunks to registry and index
        self._add_chunks_to_index(new_chunks)
        
        return len(new_chunks)
    
    def remove_document(self, document_id: str):
        """Remove a document and its chunks from the index."""
        if document_id not in self.original_files:
            return 0
        
        # Find and remove chunks from this document
        chunks_removed = self._remove_document_chunks(document_id)
        
        # Remove from original files list
        if document_id in self.original_files:
            self.original_files.remove(document_id)
        
        return chunks_removed
    
    def _process_document_to_chunks(self, text: str) -> List[Dict]:
        """Process document into chunks with metadata."""
        # Store original content
        original_content = self.content
        self.content = text
        
        # Process into chunks
        self.split_chunks(use_semantic_chunking=True, use_hierarchical=True)
        
        # Generate embeddings for new chunks
        if self.chunks:
            new_embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True
            ).astype("float32")
        else:
            new_embeddings = np.array([])
        
        # Create chunk registry entries
        chunks_with_metadata = []
        for i, (chunk, embedding) in enumerate(zip(self.chunks, new_embeddings)):
            chunk_id = f"{self.next_chunk_id + i}"
            metadata = {
                'chunk_id': chunk_id,
                'document_id': f"doc_{len(self.original_files)}",
                'chunk_text': chunk,
                'embedding': embedding,
                'timestamp': time.time()
            }
            chunks_with_metadata.append(metadata)
        
        # Restore original content
        self.content = original_content
        
        # Update counter
        self.next_chunk_id += len(chunks_with_metadata)
        
        return chunks_with_metadata
    
    def _add_chunks_to_index(self, chunks: List[Dict]):
        """Add new chunks to FAISS index incrementally."""
        if not chunks:
            return
        
        # Initialize index if it doesn't exist
        if self.index is None:
            dimension = chunks[0]['embedding'].shape[0]
            self.index = faiss.IndexFlatIP(dimension)
            self.embeddings = np.array([chunk['embedding'] for chunk in chunks])
        else:
            # Add to existing index
            new_embeddings = np.array([chunk['embedding'] for chunk in chunks])
            self.index.add(new_embeddings)
            
            # Update embeddings array
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
        
        # Add to chunk registry
        for chunk in chunks:
            self.chunk_registry[chunk['chunk_id']] = chunk
        
        # Update BM25
        if hasattr(self, 'bm25') and self.bm25:
            # Rebuild BM25 with all chunks
            self._rebuild_bm25()
    
    def _remove_document_chunks(self, document_id: str) -> int:
        """Remove chunks belonging to a document from index."""
        chunks_to_remove = [
            chunk_id for chunk_id, chunk in self.chunk_registry.items()
            if chunk.get('document_id') == document_id
        ]
        
        if not chunks_to_remove:
            return 0
        
        # Remove from FAISS index (requires rebuilding for now)
        # In production, use IndexIDMap for true incremental removal
        self._rebuild_faiss_index()
        
        # Remove from registry
        for chunk_id in chunks_to_remove:
            if chunk_id in self.chunk_registry:
                del self.chunk_registry[chunk_id]
        
        return len(chunks_to_remove)
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from chunk registry."""
        if not self.chunk_registry:
            self.index = None
            self.embeddings = None
            return
        
        # Get all embeddings
        all_embeddings = np.array([
            chunk['embedding'] for chunk in self.chunk_registry.values()
        ])
        
        # Rebuild index
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(all_embeddings)
        self.embeddings = all_embeddings
    
    def _rebuild_bm25(self):
        """Rebuild BM25 index from all chunks."""
        if not self.chunk_registry:
            self.bm25 = None
            return
        
        # Get all chunk texts
        all_chunks = [chunk['chunk_text'] for chunk in self.chunk_registry.values()]
        tokenized_chunks = [chunk.split() for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

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
