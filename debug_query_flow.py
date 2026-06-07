#!/usr/bin/env python3
"""Debug script to trace the actual query flow."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_query_flow():
    """Debug the actual query flow step by step."""
    print("Debugging actual query flow...")
    
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
        
        # Step 1: Rule-based answer
        print(f"\n=== Step 1: Rule-based answer ===")
        direct = rag._rule_based_answer(question)
        print(f"Rule-based answer: {direct}")
        
        # Step 2: Retrieval
        print(f"\n=== Step 2: Retrieval ===")
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks:")
        
        for i, chunk in enumerate(context):
            has_dry = "DRY" in chunk or "don't repeat yourself" in chunk.lower()
            print(f"  {i+1}. Has DRY: {has_dry}")
            if has_dry:
                print(f"     🎯 {chunk[:150]}...")
            else:
                print(f"     ❌ {chunk[:150]}...")
        
        # Step 3: Answer generation
        if context:
            print(f"\n=== Step 3: Answer generation ===")
            answer = rag.generate_answer(question, context)
            print(f"Generated answer: {answer}")
            
            # Check if answer is unsatisfactory
            print(f"\n=== Step 4: Answer quality check ===")
            is_unsatisfactory = rag._is_unsatisfactory_answer(answer, question)
            print(f"Is unsatisfactory: {is_unsatisfactory}")
            
            if is_unsatisfactory:
                print(f"Attempting to enhance answer...")
                enhanced = rag._enhance_answer(answer, question, context)
                print(f"Enhanced answer: {enhanced}")
            
            # Check if answer looks fragmented
            print(f"\n=== Step 5: Fragmentation check ===")
            looks_fragmented = rag._looks_fragmented(answer)
            print(f"Looks fragmented: {looks_fragmented}")
            
            if looks_fragmented:
                print(f"Falling back to rule-based answer...")
                fallback = rag._rule_based_answer(question)
                print(f"Fallback answer: {fallback}")
        else:
            print(f"\n=== Step 3: No context found ===")
            suggestions = rag._suggest_related_topics(question)
            print(f"Suggestions: {suggestions}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_query_flow()
