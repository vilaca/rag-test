"""Core answer generation logic."""

import re
from typing import List


class AnswerGenerationMixin:
    def _synthesize_answer(self, query: str, context: List[str], question_type: str) -> str:
        """Synthesize answer from retrieved context."""
        if not context:
            return ""
        
        # Extract candidate sentences from context
        candidates = self._extract_candidate_sentences(context)
        if not candidates:
            return ""
        
        # Get query keywords for matching
        keywords = self._query_keywords(query)
        if not keywords:
            return ""
        
        # Route to appropriate handler based on question type
        if question_type == "definition":
            return self._handle_definition_question(query, candidates, keywords)
        elif question_type == "explanation":
            return self._handle_explanation_question(query, candidates, keywords)
        elif question_type == "list":
            return self._handle_list_question(query, candidates, keywords)
        elif question_type == "comparison":
            return self._handle_comparison_question(query, candidates, keywords)
        elif question_type == "yesno":
            return self._handle_yesno_question(query, candidates, keywords)
        else:
            # General question handling
            return self._handle_general_question(query, candidates, keywords)
    
    def _handle_definition_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle definition questions."""
        query_lower = query.lower()
        
        # Look for sentences that contain definition patterns
        best_candidate = ""
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = 0
            
            # Check for definition patterns
            if " is " in candidate_lower or " are " in candidate_lower or " refers to " in candidate_lower or " means " in candidate_lower or " defined as " in candidate_lower:
                score += 2
            
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in candidate_lower)
            score += keyword_matches
            
            # Check if candidate looks like a definition
            if self._looks_like_definition(candidate):
                score += 3
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_candidate else ""
    
    def _handle_explanation_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle explanation questions."""
        query_lower = query.lower()
        
        # Look for sentences that explain processes or reasons
        best_candidate = ""
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = 0
            
            # Check for explanation patterns
            if " because " in candidate_lower or " since " in candidate_lower or " due to " in candidate_lower or " as a result " in candidate_lower or " therefore " in candidate_lower or " thus " in candidate_lower:
                score += 3
            
            # Check for process descriptions
            if " process " in candidate_lower or " procedure " in candidate_lower or " steps " in candidate_lower or " method " in candidate_lower:
                score += 2
            
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword in candidate_lower)
            score += keyword_matches
            
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate if best_candidate else ""
    
    def _handle_general_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle general questions."""
        # Find the candidate with the most keyword matches
        best_candidate = ""
        best_score = 0
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in candidate_lower)
            
            if keyword_matches > best_score:
                best_score = keyword_matches
                best_candidate = candidate
        
        return best_candidate if best_candidate else ""
    
    def _handle_list_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle list/enumeration questions."""
        # Look for candidates that contain lists or multiple items
        list_candidates = []
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Check for list indicators
            if ", " in candidate or "; " in candidate or " and " in candidate or "•" in candidate or "-" in candidate[:10]:
                # Check for keyword matches
                keyword_matches = sum(1 for keyword in keywords if keyword in candidate_lower)
                if keyword_matches > 0:
                    list_candidates.append((candidate, keyword_matches))
        
        if list_candidates:
            # Return the candidate with the most keyword matches
            list_candidates.sort(key=lambda x: x[1], reverse=True)
            return list_candidates[0][0]
        
        return ""
    
    def _handle_comparison_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle comparison questions."""
        query_lower = query.lower()
        
        # Look for candidates that compare items
        comparison_candidates = []
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Check for comparison patterns
            if " compared to " in candidate_lower or " whereas " in candidate_lower or " while " in candidate_lower or " unlike " in candidate_lower or " similar to " in candidate_lower:
                keyword_matches = sum(1 for keyword in keywords if keyword in candidate_lower)
                if keyword_matches > 0:
                    comparison_candidates.append((candidate, keyword_matches))
        
        if comparison_candidates:
            comparison_candidates.sort(key=lambda x: x[1], reverse=True)
            return comparison_candidates[0][0]
        
        return ""
    
    def _handle_yesno_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Handle yes/no questions."""
        query_lower = query.lower()
        
        # Look for candidates that answer yes/no questions
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Check for affirmative/negative patterns
            if " can " in query_lower or " could " in query_lower or " should " in query_lower:
                if " can " in candidate_lower or " could " in candidate_lower or " should " in candidate_lower:
                    return candidate
            
            if " is " in query_lower or " are " in query_lower or " was " in query_lower or " were " in query_lower:
                if " is " in candidate_lower or " are " in candidate_lower or " was " in candidate_lower or " were " in candidate_lower:
                    return candidate
        
        return ""
