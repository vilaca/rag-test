"""Utility functions for answer generation."""

import re
from typing import List, Set


class AnswerUtilsMixin:
    def _extract_candidate_sentences(self, context: List[str]) -> List[str]:
        """Extract candidate sentences from context chunks."""
        candidates = []
        
        for chunk in context:
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            
            for sentence in sentences:
                # Filter out very short or very long sentences
                if 15 < len(sentence.split()) < 50:
                    # Clean up the sentence
                    cleaned = sentence.strip()
                    if cleaned and cleaned not in candidates:
                        candidates.append(cleaned)
        
        return candidates
    
    def _query_keywords(self, query: str) -> set:
        """Extract keywords from query."""
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
                     "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                     "do", "does", "did", "will", "would", "could", "should", "what", "how", "why"}
        
        # Extract meaningful words
        words = [word.lower() for word in query.split() if len(word) > 2]
        keywords = [word for word in words if word not in stop_words]
        
        return set(keywords)
    
    def _looks_like_definition(self, text: str) -> bool:
        """Check if text looks like a definition."""
        text_lower = text.lower()
        
        # Check for definition patterns
        definition_patterns = [
            " is ", " are ", " refers to ", " means ", " defined as ",
            " is defined as ", " are defined as ", " can be defined as "
        ]
        
        # Check for definition structure
        if any(pattern in text_lower for pattern in definition_patterns):
            # Check if it's a complete thought
            if len(text.split()) > 8 and len(text.split()) < 40:
                return True
        
        return False
    
    def _looks_fragmented(self, text: str) -> bool:
        """Check if text looks like a fragmented sentence."""
        # Check for incomplete sentences
        if not text.endswith('.'):
            return True
        
        # Check for very short sentences
        if len(text.split()) < 5:
            return True
        
        # Check for sentences that seem cut off
        last_word = text.split()[-1].lower()
        if last_word in ["the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from"]:
            return True
        
        return False
    
    def _simple_extractive_answer(self, query: str, context: List[str]) -> str:
        """Generate simple extractive answer."""
        if not context:
            return ""
        
        # Find the chunk with the most overlap with the query
        query_lower = query.lower()
        best_chunk = ""
        best_score = 0
        
        for chunk in context:
            chunk_lower = chunk.lower()
            
            # Calculate overlap score
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)
            
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_chunk:
            # Extract a relevant sentence
            sentences = re.split(r"(?<=[.!?])\s+", best_chunk)
            for sentence in sentences:
                if len(sentence) > 20 and len(sentence) < 200:
                    return sentence
        
        return ""
    
    def _rule_based_answer(self, question: str) -> str:
        """Generate rule-based answers for common questions."""
        question_lower = question.lower()
        
        # Handle common questions with predefined answers
        if "what is the purpose" in question_lower:
            return "The purpose is to provide comprehensive information about the topic."
        
        if "who created this" in question_lower or "who wrote this" in question_lower:
            return "This document was created by experts in the field."
        
        if "when was this created" in question_lower:
            return "This document contains up-to-date information."
        
        return ""
