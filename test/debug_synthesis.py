#!/usr/bin/env python3
"""Debug the answer synthesis process step by step."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def debug_synthesis():
    """Debug synthesis step by step."""
    print("🔧 Debugging Answer Synthesis Step by Step")
    print("=" * 50)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    question = "define Brooks's Law"
    print(f"Question: {question}")
    print()
    
    # Get context
    context = rag.retrieve(question)
    print(f"Retrieved {len(context)} chunks")
    
    # Check question type
    question_type = rag._analyze_question(question)
    print(f"Question type: {question_type}")
    
    # Extract candidates
    candidates = rag._extract_candidate_sentences(context)
    print(f"\nExtracted {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates[:5]):
        print(f"  {i+1}. {candidate[:80]}...")
    
    # Try synthesis
    try:
        answer = rag._synthesize_answer(question, context, question_type)
        print(f"\nSynthesis result: {answer[:100]}...")
    except Exception as e:
        print(f"\nSynthesis failed: {e}")
    
    # Check if definition handling is triggered
    if question_type == "factual":
        print(f"\nDefinition handling should be triggered for factual question")
        definition_result = rag._handle_definition_question(question, candidates, rag._query_keywords(question))
        print(f"Definition result: {definition_result[:100]}...")
    
    # Final answer
    final_answer = rag.query(question)
    print(f"\nFinal answer: {final_answer}")

if __name__ == "__main__":
    debug_synthesis()