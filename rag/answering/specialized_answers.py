"""Specialized answer generation for specific question types."""

from typing import List


class SpecializedAnswersMixin:
    def _generate_thematic_answer(self, query: str, context: List[str]) -> str:
        """Generate answer for thematic questions."""
        if not context:
            return ""
        
        # Extract the theme/topic from the query
        query_lower = query.lower()
        
        # Find the most relevant chunks that discuss the theme
        thematic_chunks = []
        for chunk in context:
            chunk_lower = chunk.lower()
            if "laws about " in query_lower or "principles about " in query_lower:
                if "laws" in chunk_lower or "principles" in chunk_lower:
                    thematic_chunks.append(chunk)
            elif "related to " in query_lower:
                # Extract the topic after "related to"
                topic = query_lower.split("related to ")[-1].strip()
                if topic in chunk_lower:
                    thematic_chunks.append(chunk)
        
        if thematic_chunks:
            # Combine the most relevant thematic information
            combined = " ".join(thematic_chunks[:3])
            return combined if len(combined) < 500 else combined[:500] + "..."
        
        return ""
    
    def _generate_overview_answer(self, query: str, context: List[str]) -> str:
        """Generate overview answer for summary questions."""
        if not context:
            return ""
        
        # Extract the topic from the query
        query_lower = query.lower()
        
        # Find chunks that provide overview information
        overview_chunks = []
        for chunk in context:
            chunk_lower = chunk.lower()
            
            # Look for summary indicators
            if any(indicator in chunk_lower for indicator in [
                "summary", "overview", "main points", "key aspects",
                "main principles", "main laws", "main topics", "main themes"
            ]):
                overview_chunks.append(chunk)
            
            # Also look for introductory paragraphs
            if len(chunk) > 100 and len(chunk) < 300:
                # Check if it contains multiple key points
                sentences = chunk.split('.')
                if len(sentences) > 2:
                    overview_chunks.append(chunk)
        
        if overview_chunks:
            # Combine and summarize
            combined = " ".join(overview_chunks[:2])
            return combined if len(combined) < 400 else combined[:400] + "..."
        
        return ""
    
    def _generate_meta_answer(self, query: str) -> str:
        """Generate meta answers about the document itself."""
        query_lower = query.lower()
        
        # Handle different types of meta questions
        if "how many laws" in query_lower:
            law_count = self._count_mentioned_laws(self.context_chunks)
            return f"This document mentions approximately {law_count} laws or legal principles."
        
        elif "how many principles" in query_lower:
            principle_count = self._count_mentioned_laws(self.context_chunks)
            return f"This document mentions approximately {principle_count} principles."
        
        elif "what is this document about" in query_lower or "what topics does this cover" in query_lower:
            categories = self._extract_categories(self.context_chunks)
            if categories:
                return f"This document covers topics such as: {', '.join(categories[:5])}."
            else:
                return "This document covers various topics related to laws and principles."
        
        elif "who would benefit from reading this" in query_lower:
            return "This document would benefit anyone interested in understanding laws and principles, including students, professionals, and researchers."
        
        elif "what is the purpose of this document" in query_lower:
            return "The purpose of this document is to provide comprehensive information about laws and principles."
        
        return ""
    
    def _enhance_definition_answer(self, original_answer: str, question: str, context: List[str]) -> str:
        """Enhance definition answers with additional context."""
        if not original_answer:
            return ""
        
        # Extract additional relevant information from context
        question_lower = question.lower()
        additional_info = []
        
        for chunk in context:
            chunk_lower = chunk.lower()
            
            # Check if chunk provides additional relevant details
            if len(chunk) > 50 and len(chunk) < 250:
                # Look for elaboration patterns
                if " in addition " in chunk_lower or " furthermore " in chunk_lower or " moreover " in chunk_lower or " also " in chunk_lower:
                    sentences = chunk.split('.')
                    for sentence in sentences:
                        if len(sentence) > 20 and len(sentence) < 150:
                            sentence_lower = sentence.lower()
                            # Check if sentence contains relevant keywords from the question
                            question_keywords = [word for word in question_lower.split() if len(word) > 3]
                            if any(keyword in sentence_lower for keyword in question_keywords):
                                additional_info.append(sentence)
                                if len(additional_info) >= 2:
                                    break
        
        if additional_info:
            additional_context = ' '.join(additional_info[:2])
            return f"{original_answer} {additional_context}"
        
        return original_answer
    
    def _count_mentioned_laws(self, context: List[str]) -> int:
        """Count how many laws/principles are mentioned in context."""
        law_keywords = ["law", "laws", "principle", "principles", "rule", "rules", "regulation", "regulations"]
        count = 0
        
        for chunk in context:
            chunk_lower = chunk.lower()
            # Count occurrences of law-related keywords
            count += sum(1 for keyword in law_keywords if keyword in chunk_lower)
        
        # Return a reasonable estimate
        return max(5, min(50, count // 3))
    
    def _extract_categories(self, context: List[str]) -> List[str]:
        """Extract main categories/topics from context."""
        categories = set()
        
        # Common category indicators
        category_indicators = [
            "laws about ", "principles of ", "topics include ",
            "categories: ", "types of ", "kinds of ", "main areas:"
        ]
        
        for chunk in context:
            chunk_lower = chunk.lower()
            
            # Look for category indicators
            for indicator in category_indicators:
                if indicator in chunk_lower:
                    # Extract text after the indicator
                    start_idx = chunk_lower.find(indicator) + len(indicator)
                    remaining_text = chunk[start_idx:]
                    
                    # Extract potential categories (noun phrases)
                    words = remaining_text.split()
                    for i in range(min(10, len(words))):
                        if len(words[i]) > 3 and words[i][0].isupper():
                            categories.add(words[i])
        
        return list(categories)[:5]
