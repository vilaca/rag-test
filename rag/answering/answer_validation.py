"""Answer validation and hallucination detection."""

from typing import List


class AnswerValidationMixin:
    def _can_answer_be_supported(self, answer: str, query: str, context: List[str]) -> bool:
        """Check if generated answer is supported by retrieved context."""
        if not answer or not context:
            return False
        
        # Convert to lowercase for comparison
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Extract key terms from query
        query_keywords = self._extract_keywords(query_lower)
        if not query_keywords:
            return True  # No specific keywords to verify
        
        # Check if answer contains query keywords
        answer_keywords_present = any(keyword in answer_lower for keyword in query_keywords)
        
        # Check if context supports the answer
        context_supports_answer = self._check_context_support(answer_lower, query_keywords, context)
        
        # Both conditions must be met
        return answer_keywords_present and context_supports_answer
    
    def _check_context_support(self, answer: str, query_keywords: List[str], context: List[str]) -> bool:
        """Check if context provides sufficient support for the answer."""
        if not context:
            return False
        
        # Calculate support score
        support_score = 0.0
        keyword_matches = 0
        
        for chunk in context:
            chunk_lower = chunk.lower()
            
            # Count query keyword matches in context
            chunk_keyword_matches = sum(1 for keyword in query_keywords if keyword in chunk_lower)
            keyword_matches += chunk_keyword_matches
            
            # Check if answer content is present in context
            # Use n-gram overlap as proxy for support
            answer_ngrams = set(self._get_ngrams(answer, 3))
            chunk_ngrams = set(self._get_ngrams(chunk_lower, 3))
            overlap = len(answer_ngrams & chunk_ngrams)
            
            if overlap > 0:
                support_score += overlap
        
        # Normalize support score
        total_keywords = len(query_keywords)
        keyword_coverage = keyword_matches / total_keywords if total_keywords > 0 else 0
        
        # Require at least 60% keyword coverage and some ngram overlap
        return keyword_coverage >= 0.6 and support_score > 0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
                     "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
                     "do", "does", "did", "will", "would", "could", "should", "what", "how", "why"}
        
        # Extract words longer than 2 characters
        words = [word.lower() for word in text.split() if len(word) > 2]
        
        # Filter out stop words
        keywords = [word for word in words if word not in stop_words]
        
        return keywords
    
    def _get_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract n-grams from text."""
        words = text.split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)] if len(words) >= n else []
