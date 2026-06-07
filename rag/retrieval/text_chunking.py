"""Text chunking and preprocessing functionality."""

import re
from typing import List, Dict


class TextChunkingMixin:
    def _build_hierarchical_structure(self, text: str) -> Dict:
        """Build hierarchical structure of the document."""
        # Split into sections based on markdown headers
        sections = []
        current_section = {"title": "Introduction", "content": "", "paragraphs": []}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Check for markdown headers
            if line.startswith('#') or line.startswith('##') or line.startswith('###'):
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                header_level = len(line.split()[0]) if line.split() else 1
                header_text = line[header_level:].strip()
                current_section = {"title": header_text, "content": "", "paragraphs": []}
            else:
                # Add to current section content
                if line:
                    current_section["content"] += line + " "
        
        # Add the last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # Split sections into paragraphs
        structured_data = {"sections": [], "paragraphs": []}
        
        for section_idx, section in enumerate(sections):
            # Split section content into paragraphs
            paragraphs = re.split(r'\n\s*\n', section["content"])
            
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if paragraph:
                    structured_data["paragraphs"].append({
                        "text": paragraph,
                        "section_title": section["title"],
                        "section_index": section_idx,
                        "paragraph_index": para_idx
                    })
            
            structured_data["sections"].append({
                "title": section["title"],
                "paragraph_count": len(paragraphs)
            })
        
        return structured_data
    
    def _chunk_paragraph(self, paragraph_text: str, chunk_size: int, overlap: int, 
                         section_title: str, section_index: int, paragraph_index: int) -> List[str]:
        """Split a paragraph into chunks with overlap."""
        chunks = []
        text_length = len(paragraph_text)
        
        if text_length <= chunk_size:
            # Single chunk for short paragraphs
            chunk_text = paragraph_text
            chunks.append(chunk_text)
            
            # Add metadata
            self.chunk_metadata.append({
                "chunk_text": chunk_text,
                "section_title": section_title,
                "section_index": section_index,
                "paragraph_index": paragraph_index,
                "chunk_index": 0,
                "is_complete_paragraph": True
            })
        else:
            # Multiple chunks for long paragraphs
            start = 0
            chunk_idx = 0
            
            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunk_text = paragraph_text[start:end]
                
                # Try to end at sentence boundary
                if end < text_length:
                    last_period = chunk_text.rfind('.')
                    if last_period > chunk_size // 2:  # Don't make chunks too small
                        chunk_text = chunk_text[:last_period + 1]
                        end = start + len(chunk_text)
                
                chunks.append(chunk_text)
                
                # Add metadata
                self.chunk_metadata.append({
                    "chunk_text": chunk_text,
                    "section_title": section_title,
                    "section_index": section_index,
                    "paragraph_index": paragraph_index,
                    "chunk_index": chunk_idx,
                    "is_complete_paragraph": False
                })
                
                # Move to next chunk with overlap
                start = end - overlap if end - overlap > start else end
                chunk_idx += 1
        
        return chunks
    
    def _add_chunk_metadata(self, chunk_text: str, section_title: str, section_index: int, 
                           paragraph_index: int, chunk_index: int, is_complete: bool):
        """Add metadata to a chunk."""
        if not hasattr(self, 'chunk_metadata'):
            self.chunk_metadata = []
        
        self.chunk_metadata.append({
            "chunk_text": chunk_text,
            "section_title": section_title,
            "section_index": section_index,
            "paragraph_index": paragraph_index,
            "chunk_index": chunk_index,
            "is_complete_paragraph": is_complete
        })
    
    def _is_structured_document(self, text: str) -> bool:
        """Check if text appears to be a structured document."""
        lines = text.split('\n')
        header_count = 0
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('#') or line.startswith('##') or line.startswith('###'):
                header_count += 1
        
        return header_count >= 2
    
    def _split_structured_document(self, text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        """Split structured documents while preserving structure."""
        if not self._is_structured_document(text):
            return []
        
        # Use hierarchical structure
        structure = self._build_hierarchical_structure(text)
        chunks = []
        
        for para_data in structure['paragraphs']:
            para_chunks = self._chunk_paragraph(
                para_data['text'],
                chunk_size,
                overlap,
                para_data['section_title'],
                para_data['section_index'],
                para_data['paragraph_index']
            )
            chunks.extend(para_chunks)
        
        return chunks
