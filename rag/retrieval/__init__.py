"""Retrieval module combining all retrieval-related functionality."""

from .text_chunking import TextChunkingMixin
from .embedding_generation import EmbeddingGenerationMixin
from .retrieval_strategies import RetrievalStrategiesMixin
from .retrieval_postprocessing import RetrievalPostprocessingMixin
from .retrieval_optimization import RetrievalOptimizationMixin


class RetrievalMixin(
    TextChunkingMixin,
    EmbeddingGenerationMixin,
    RetrievalStrategiesMixin,
    RetrievalPostprocessingMixin,
    RetrievalOptimizationMixin
):
    """Comprehensive retrieval mixin combining all retrieval-related functionality."""
    
    def split_chunks(self, chunk_size: int = 512, overlap: int = 100, use_semantic_chunking: bool = True, use_hierarchical: bool = True):
        """Split content into overlapping chunks with hierarchical metadata."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 5)

        # Keep paragraph structure first (don't collapse newlines too early)
        raw_text = self.content.replace("\r\n", "\n").replace("\r", "\n")

        # Build hierarchical structure for metadata preservation
        hierarchical_structure = self._build_hierarchical_structure(raw_text)
        
        # Store metadata for context reconstruction
        self.chunk_metadata = []
        
        if use_hierarchical and hierarchical_structure['paragraphs']:
            # Hierarchical chunking: work with paragraphs and preserve metadata
            chunks = []
            
            for para_data in hierarchical_structure['paragraphs']:
                paragraph_text = para_data['text']
                
                # Split paragraph into chunks with overlap
                para_chunks = self._chunk_paragraph(
                    paragraph_text, 
                    chunk_size, 
                    overlap,
                    para_data['section_title'],
                    para_data['section_index'],
                    para_data['paragraph_index']
                )
                chunks.extend(para_chunks)
            
            self.chunks = chunks
        else:
            # Fallback: simple chunking
            words = raw_text.split()
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)
                
                # Basic metadata
                self._add_chunk_metadata(chunk_text, "unknown", 0, 0, len(chunks) - 1, True)
            
            self.chunks = chunks
        
        print(f"Split content into {len(self.chunks)} chunks")
        return self.chunks
    
    def _query_keywords(self, query: str) -> list:
        """Extract keywords from query."""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
                     "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                     "do", "does", "did", "will", "would", "could", "should", "what", "how", "why"}
        
        words = [word.lower() for word in query.split() if len(word) > 2]
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
