"""Embedding generation and indexing functionality."""

import time
from typing import List
import numpy as np
import faiss


class EmbeddingGenerationMixin:
    def generate_embeddings(self):
        """Generate embeddings for all chunks."""
        if not hasattr(self, 'chunks') or not self.chunks:
            print("No chunks available for embedding generation.")
            return
        
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        start_time = time.time()
        
        # Generate embeddings using the embedding model
        try:
            embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True, convert_to_numpy=True)
            
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            self.embeddings = embeddings
            print(f"Embedding generation completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def build_index(self):
        """Build FAISS index for efficient similarity search."""
        if not hasattr(self, 'embeddings') or self.embeddings is None:
            print("No embeddings available. Generating embeddings first...")
            self.generate_embeddings()
        
        if not hasattr(self, 'embeddings'):
            raise ValueError("Embeddings not available")
        
        print("Building FAISS index...")
        start_time = time.time()
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        
        # Use appropriate index based on dataset size
        if len(self.embeddings) < 1000:
            # Small dataset: use flat index
            self.index = faiss.IndexFlatIP(dimension)
        elif len(self.embeddings) < 10000:
            # Medium dataset: use IVF
            nlist = min(100, len(self.embeddings) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
        else:
            # Large dataset: use HNSW
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        print(f"FAISS index built in {time.time() - start_time:.2f} seconds")
        print(f"Index contains {self.index.ntotal} vectors")
    
    def _initialize_bm25(self):
        """Initialize BM25 retriever."""
        if not hasattr(self, 'chunks') or not self.chunks:
            raise ValueError("No chunks available for BM25 initialization")
        
        # Tokenize chunks for BM25
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
