"""Question analysis and classification for answer generation."""

from typing import List


class QuestionAnalysisMixin:
    def _analyze_question(self, query: str) -> str:
        """Analyze question type to determine best answering strategy."""
        query_lower = query.lower().strip()
        
        # Remove question marks and normalize
        query_clean = query_lower.rstrip('?')
        
        # Check for meta/document-level questions first
        if any(phrase in query_clean for phrase in [
            "how many laws are in this document",
            "how many principles are in this document",
            "what is this document about",
            "who would benefit from reading this",
            "what topics does this cover",
            "what is the purpose of this document", "what topics does this cover", "what topics does this cover",
            "who is this document for"
        ]):
            return "meta"
        
        # Check for meta/document-level questions first
        if any(phrase in query_clean for phrase in [
            "how many laws are in this document",
            "how many principles are in this document",
            "what is this document about",
            "who would benefit from reading this",
            "what topics does this cover",
            "what is the purpose of this document", "what topics does this cover", "what topics does this cover",
            "who is this document for"
        ]):
            return "meta"
        
        # Check for thematic/topic questions first
        if any(phrase in query_clean for phrase in [
            "laws about ",
            "principles about ",
            "laws related to ",
            "principles related to ",
            "what law talks about ",
            "which principle deals with ",
            "find laws related to ",
            "laws concerning ",
            "principles concerning ",
            "what laws cover ",
            "what principles cover "
        ]):
            return "thematic"
        
        # Check for overview/summary questions
        if any(phrase in query_clean for phrase in [
            "summarize ",
            "give me an overview of ",
            "what are the main points of ",
            "what are the key aspects of ",
            "what are the main principles of ",
            "what are the main laws of ",
            "what are the main topics of ",
            "what are the main themes of ",
            "what are the main ideas of "
        ]):
            return "overview"
        
        # Check for definition questions
        if any(phrase in query_clean for phrase in [
            "what is ",
            "what are ",
            "define ",
            "explain ",
            "what does ",
            "what do ",
            "what was ",
            "what were "
        ]):
            return "definition"
        
        # Check for explanation questions
        if any(phrase in query_clean for phrase in [
            "how does ",
            "how do ",
            "how did ",
            "how can ",
            "how could ",
            "how would ",
            "how will ",
            "why does ",
            "why do ",
            "why did ",
            "why is ",
            "why are ",
            "why was ",
            "why were "
        ]):
            return "explanation"
        
        # Check for list/enumeration questions
        if any(phrase in query_clean for phrase in [
            "list the ",
            "what are the ",
            "name the ",
            "identify the ",
            "enumerate the ",
            "describe the "
        ]):
            return "list"
        
        # Check for comparison questions
        if any(phrase in query_clean for phrase in [
            "compare ",
            "contrast ",
            "what is the difference between ",
            "how does compare to ",
            "how does differ from "
        ]):
            return "comparison"
        
        # Check for yes/no questions
        if any(phrase in query_clean for phrase in [
            "does ",
            "do ",
            "did ",
            "is ",
            "are ",
            "was ",
            "were ",
            "can ",
            "could ",
            "should ",
            "would ",
            "will "
        ]) and (query_clean.endswith('?') or any(word in query_clean for word in ['whether', 'if'])):
            return "yesno"
        
        # Default to general question type
        return "general"
