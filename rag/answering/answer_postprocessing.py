"""Answer post-processing and formatting."""

import re
from typing import List


class AnswerPostprocessingMixin:
    def _post_process_answer(self, answer: str, query: str, context: List[str]) -> str:
        """Post-process answer for quality and reliability."""
        if not answer or not answer.strip():
            # Try to extract some relevant information from context
            if context:
                # Look for any sentence containing query keywords
                query_keywords = self._query_keywords(query)
                for chunk in context:
                    chunk_lower = chunk.lower()
                    if any(keyword in chunk_lower for keyword in query_keywords if len(keyword) > 3):
                        # Extract a relevant sentence
                        sentences = re.split(r"(?<=[.!?])\s+", chunk)
                        for sentence in sentences:
                            if len(sentence) > 20 and len(sentence) < 300:
                                sentence_lower = sentence.lower()
                                if any(keyword in sentence_lower for keyword in query_keywords if len(keyword) > 3):
                                    return self._clean_answer(sentence)
            return "I couldn't find any relevant information about that in the content."
        
        # Clean up the answer
        answer = self._clean_answer(answer)
        
        # Add source attribution if possible
        answer = self._add_source_attribution(answer, context)
        
        # Validate answer completeness
        if self._is_unsatisfactory_answer(answer, query):
            answer += " " + self._get_additional_context(query, context)
        
        return answer
    
    def _add_source_attribution(self, answer: str, context: List[str]) -> str:
        """Add source attribution to the answer."""
        # Find the most relevant context chunk
        if context:
            best_chunk = max(context, key=lambda x: len(x) if 50 < len(x) < 300 else 0)
            if len(best_chunk) > 50:
                # Extract a short citation
                citation = best_chunk[:100] + "..." if len(best_chunk) > 100 else best_chunk
                answer += f"\n\nSource: {citation}"
        return answer
    
    def _get_additional_context(self, query: str, context: List[str]) -> str:
        """Get additional context for incomplete answers."""
        query_lower = query.lower()
        additional_info = []
        
        for chunk in context[:3]:
            if len(chunk) > 50 and len(chunk) < 300:
                # Check if this chunk provides additional relevant information
                chunk_lower = chunk.lower()
                if any(term in chunk_lower for term in query_lower.split() if len(term) > 3):
                    # Extract key sentences
                    sentences = re.split(r"(?<=[.!?])\s+", chunk)
                    for sentence in sentences:
                        if len(sentence) > 20 and len(sentence) < 200:
                            sentence_lower = sentence.lower()
                            if any(term in sentence_lower for term in query_lower.split() if len(term) > 3):
                                additional_info.append(sentence)
                                if len(additional_info) >= 2:
                                    break
        
        if additional_info:
            return "Additionally, " + ". ".join(additional_info[:2]) + "."
        
        return ""
    
    def _clean_answer(self, text: str) -> str:
        """Clean up answer text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove markdown noise
        text = self._strip_markdown_noise(text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if text else text
        
        # Ensure proper punctuation
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        return text
    
    def _strip_markdown_noise(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove bold/italic markers
        text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^\*]+)\*", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        
        # Remove links
        text = re.sub(r"\[([^\[\]]+)\]\([^\)]+\)", r"\1", text)
        
        # Remove headers
        text = re.sub(r"^#+\s+", "", text)
        
        # Remove excessive newlines
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"\n", " ", text)
        
        return text
    
    def _is_unsatisfactory_answer(self, answer: str, question: str) -> bool:
        """Check if answer is unsatisfactory or incomplete."""
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Check for vague answers
        vague_phrases = [
            "may be", "might be", "could be", "possibly", "perhaps", "sometimes",
            "in some cases", "can be", "often", "typically", "generally"
        ]
        
        if any(phrase in answer_lower for phrase in vague_phrases):
            return True
        
        # Check if answer is too short
        if len(answer.split()) < 8:
            return True
        
        # Check if answer doesn't address the question
        question_keywords = [word for word in question_lower.split() if len(word) > 3]
        answer_keywords = [word for word in answer_lower.split() if len(word) > 3]
        
        keyword_overlap = len(set(question_keywords) & set(answer_keywords))
        if keyword_overlap < 2:
            return True
        
        return False
