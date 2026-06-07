"""Post-processing and enhancement of retrieval results."""

from typing import List
import re


class RetrievalPostprocessingMixin:
    def _add_contextual_chunks(self, query: str, ranked_chunks: list, indices, scores) -> list:
        """Add contextual chunks around top results."""
        if not ranked_chunks or len(ranked_chunks) < 3:
            return ranked_chunks
        
        # Get the top 3 indices
        top_indices = indices[:3]
        
        # For each top chunk, add nearby chunks if they provide additional context
        enhanced_chunks = list(ranked_chunks)
        
        for idx in top_indices:
            # Add previous and next chunks if they exist and are relevant
            if idx > 0 and idx - 1 not in top_indices:
                prev_chunk = self.chunks[idx - 1]
                if self._contains_related_terms(prev_chunk, self._query_keywords(query)):
                    enhanced_chunks.append(prev_chunk)
            
            if idx < len(self.chunks) - 1 and idx + 1 not in top_indices:
                next_chunk = self.chunks[idx + 1]
                if self._contains_related_terms(next_chunk, self._query_keywords(query)):
                    enhanced_chunks.append(next_chunk)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in enhanced_chunks:
            chunk_key = chunk[:50]  # Use first 50 chars as key
            if chunk_key not in seen:
                seen.add(chunk_key)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _filter_relevant_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """Filter chunks to keep only the most relevant ones."""
        if not chunks:
            return []
        
        # Extract keywords from query
        keywords = self._query_keywords(query)
        if not keywords:
            return chunks
        
        # Filter chunks based on keyword presence and relevance
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_lower = chunk.lower()
            
            # Check if chunk contains any query keywords
            keyword_present = any(keyword in chunk_lower for keyword in keywords)
            
            # Check if chunk is reasonably sized
            reasonable_size = 50 < len(chunk) < 500
            
            if keyword_present and reasonable_size:
                filtered_chunks.append(chunk)
            
            # Stop if we have enough chunks
            if len(filtered_chunks) >= 20:
                break
        
        return filtered_chunks if filtered_chunks else chunks
    
    def _reconstruct_context(self, chunk: str) -> str:
        """Reconstruct broader context for a chunk."""
        if not hasattr(self, 'chunk_metadata') or not self.chunk_metadata:
            return chunk
        
        # Find the metadata for this chunk
        chunk_key = chunk[:100]  # Use first 100 chars as key
        
        for metadata in self.chunk_metadata:
            if metadata['chunk_text'][:100] == chunk_key:
                # Get the full paragraph if this is a partial chunk
                if not metadata['is_complete_paragraph']:
                    section_chunks = [
                        m['chunk_text'] for m in self.chunk_metadata
                        if m['section_index'] == metadata['section_index'] and
                           m['paragraph_index'] == metadata['paragraph_index']
                    ]
                    
                    # Combine all chunks from the same paragraph
                    full_paragraph = ' '.join(section_chunks)
                    return full_paragraph
                
                break
        
        return chunk
    
    def _contains_related_terms(self, chunk: str, keywords: List[str]) -> bool:
        """Check if chunk contains related terms to keywords."""
        if not keywords:
            return False
        
        chunk_lower = chunk.lower()
        
        # Check for exact keyword matches
        exact_matches = any(keyword in chunk_lower for keyword in keywords)
        
        # Check for related terms (simple stemming)
        related_terms = []
        for keyword in keywords:
            if len(keyword) > 4:
                # Simple stemming: remove common endings
                stem = keyword.rstrip('ing')
                stem = stem.rstrip('ed')
                stem = stem.rstrip('es')
                stem = stem.rstrip('s')
                if len(stem) > 3:
                    related_terms.append(stem)
        
        related_matches = any(term in chunk_lower for term in related_terms)
        
        return exact_matches or related_matches
