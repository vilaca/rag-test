#!/usr/bin/env python3
"""Debug script to understand candidate scoring."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_scoring():
    """Debug the candidate scoring."""
    print("Debugging candidate scoring...")
    
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
        
        # Get context
        context = rag.retrieve(question)
        print(f"Retrieved {len(context)} chunks")
        
        # Extract candidates
        candidates = rag._extract_candidate_sentences(context)
        print(f"Extracted {len(candidates)} candidates")
        
        # Score candidates manually
        keywords = rag._query_keywords(question)
        question_type = rag._analyze_question(question)
        
        print(f"Keywords: {keywords}")
        print(f"Question type: {question_type}")
        
        scored = []
        for sent in candidates[:10]:  # Score first 10 candidates
            s_lower = sent.lower()
            score = 0.0
            
            # Keyword overlap
            overlap = sum(1 for k in keywords if f" {k} " in f" {s_lower} ")
            score += overlap * 2.0
            
            # Question type specific scoring
            if question_type == "advice" and any(
                x in s_lower for x in ["should", "recommend", "important", "best", "must", "need"]
            ):
                score += 1.0
            elif question_type == "explanatory" and any(
                x in s_lower for x in ["states that", "means that", "refers to", "is the principle that"]
            ):
                score += 1.5
            
            # Length preference
            if 40 <= len(sent) <= 180:
                score += 0.5
            
            scored.append((score, sent))
            has_dry = "dry" in s_lower
            print(f"Score {score:.1f}: {has_dry} - {sent[:80]}...")
            
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            max_score = scored[0][0]
            print(f"\nMax score: {max_score:.1f}")
            print(f"Threshold would be: {max(0.7, max_score * 0.4):.1f}")
            
            top = [s for sc, s in scored if sc >= max(0.7, max_score * 0.4)]
            print(f"Top candidates passing threshold: {len(top)}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_scoring()
