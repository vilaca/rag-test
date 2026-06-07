#!/usr/bin/env python3
"""Debug script to understand answer generation failure."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_answer_generation():
    """Debug the answer generation process."""
    print("Debugging answer generation...")
    
    # Initialize system with the software engineering laws document
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    
    try:
        rag = RAGSystem(doc_path)
        rag.load_content()
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()
        
        print(f"Loaded {len(rag.chunks)} chunks")
        
        # Test the DRY question
        question = "explain DRY"
        print(f"\nQuestion: {question}")
        
        # Get context manually (we know this works)
        context = [
            "### 31. DRY (Don't Repeat Yourself)\n\n*See also: [Rule of Three](#32-rule-of-three), the discipline that keeps DRY from being applied too eagerly.",
            "### 31. DRY (Don't Repeat Yourself)\n\n**Caveats.** Aggressive DRY is one of the leading sources of bad abstractions.",
            "### Contents\n\nDRY](#31-dry-dont-repeat-yourself) - One authoritative representation per piece of knowledge — but knowledge, not syntax"
        ]
        
        print(f"Context: {len(context)} chunks")
        for i, chunk in enumerate(context):
            print(f"  {i+1}. {chunk[:100]}...")
        
        # Test keyword extraction
        print(f"\n=== Keyword Extraction ===")
        query_keywords = rag._extract_keywords(question.lower())
        print(f"Extracted keywords: {query_keywords}")
        
        # Generate answer
        print(f"\n=== Answer Generation ===")
        answer = rag._synthesize_answer(question, context, "explanatory")
        print(f"Generated answer: {answer}")
        
        # Test hallucination guard
        print(f"\n=== Hallucination Guard ===")
        
        # Check if answer contains query keywords
        answer_lower = answer.lower()
        answer_keywords_present = any(keyword in answer_lower for keyword in query_keywords)
        print(f"Answer contains query keywords: {answer_keywords_present}")
        
        # Check context support
        context_supports_answer = rag._check_context_support(answer_lower, query_keywords, context)
        print(f"Context supports answer: {context_supports_answer}")
        
        # Overall hallucination check
        can_be_supported = rag._can_answer_be_supported(answer, question, context)
        print(f"Answer can be supported: {can_be_supported}")
        
        if can_be_supported:
            print("✅ Answer passes hallucination guard")
        else:
            print("❌ Answer fails hallucination guard")
            
        # Test the full generate_answer method
        print(f"\n=== Full generate_answer method ===")
        full_answer = rag.generate_answer(question, context)
        print(f"Full answer: {full_answer}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_answer_generation()
