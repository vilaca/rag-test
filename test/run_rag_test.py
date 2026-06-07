#!/usr/bin/env python3
"""
Comprehensive RAG System Test Runner
=====================================

This script tests the RAG system with a variety of question types and provides
detailed output showing the system's performance.

Usage:
    python run_rag_test.py          # Run full test suite
    python run_rag_test.py quick    # Run quick test with key questions
    python run_rag_test.py debug    # Run with debug output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def run_full_test():
    """Run the complete test suite with all question categories."""
    print("🧪 Running Full RAG System Test Suite")
    print("=" * 60)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test categories with questions
    test_categories = {
        "Definition Questions": [
            "what is YAGNI?",
            "what is Conway's Law?",
            "define Brooks's Law",
            "what does the CAP Theorem state?",
            "explain Price's Law"
        ],
        "Overview Questions": [
            "what software engineering laws do you know?",
            "list the software engineering laws",
            "tell me about the principles in this document",
            "describe the laws in this collection",
            "what kinds of software engineering laws are there?"
        ],
        "Category/Organization Questions": [
            "how are the laws organized?",
            "what categories do the laws fall into?",
            "tell me about the different types of laws",
            "how are the software engineering laws categorized?"
        ],
        "Specific Law Details": [
            "explain the implications of Conway's Law",
            "what are the caveats of YAGNI?",
            "how does Brooks's Law apply in practice?",
            "what are some examples of the CAP Theorem?"
        ],
        "Thematic Questions": [
            "what law talks about team communication?",
            "find laws related to system design",
            "which principle deals with software evolution?",
            "what are the laws about constraints?"
        ],
        "Comparison Questions": [
            "how does Conway's Law differ from Brooks's Law?",
            "compare YAGNI and KISS principles",
            "what's the difference between mathematical constraints and practical rules?"
        ],
        "Edge Cases": [
            "what's the deal with these software laws?",
            "tell me something interesting about software engineering principles",
            "give me an overview of the most important laws",
            "what should I know about these software engineering concepts?"
        ],
        "Retrieval Challenges": [
            "what law talks about team communication?",
            "which principle deals with software evolution?",
            "find laws related to system design",
            "what are the laws about constraints?"
        ],
        "Meta Questions": [
            "how many laws are in this document?",
            "what is this document about?",
            "who would benefit from reading this?",
            "what topics does this cover?"
        ]
    }
    
    total_questions = 0
    good_answers = 0
    
    for category, questions in test_categories.items():
        print(f"\n📋 {category}")
        print("-" * len(category))
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                answer = rag.query(question)
                print(f"A: {answer}")
                
                # Sophisticated evaluation based on content quality
                is_good = evaluate_answer_quality(question, answer)
                
                if is_good:
                    print("✅ Good answer")
                    good_answers += 1
                else:
                    print("❌ Needs improvement")
                
                total_questions += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                total_questions += 1
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    print(f"Total questions tested: {total_questions}")
    print(f"Good answers: {good_answers}")
    print(f"Success rate: {good_answers/total_questions*100:.1f}%")

def run_quick_test():
    """Run a quick test with key questions."""
    print("🚀 Running Quick RAG Test")
    print("=" * 40)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Key test questions
    key_questions = [
        "what is YAGNI?",
        "explain Conway's Law",
        "what software engineering laws do you know?",
        "find laws related to system design",
        "how does Brooks's Law apply in practice?"
    ]
    
    for i, question in enumerate(key_questions, 1):
        print(f"\n{i}. Question: {question}")
        answer = rag.query(question)
        print(f"   Answer: {answer}")
        print("   " + "-" * 50)

def run_debug_test():
    """Run test with debug output to see retrieval details."""
    print("🔍 Running Debug Test with Retrieval Details")
    print("=" * 50)
    
    # Initialize system with debug enabled
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path, debug_retrieval=True)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test a few questions with debug output
    debug_questions = [
        "what is Conway's Law?",
        "what software engineering laws do you know?",
        "find laws related to system design"
    ]
    
    for question in debug_questions:
        print(f"\nQuestion: {question}")
        print("Debug retrieval information:")
        answer = rag.query(question)
        print(f"\nFinal Answer: {answer}")
        print("=" * 60)

def evaluate_answer_quality(question, answer):
    """Evaluate answer quality based on multiple criteria."""
    if not answer or len(answer) < 30:
        return False
    
    # Check for common failure patterns
    failure_patterns = [
        "i couldn't find a clear definition",
        "the content doesn't provide a detailed explanation",
        "try asking about specific aspects",
        "no direct answer",
        "not found in this content"
    ]
    
    answer_lower = answer.lower()
    for pattern in failure_patterns:
        if pattern in answer_lower:
            return False
    
    # Check for reasonable length
    if len(answer) < 40 or len(answer) > 600:
        return False
    
    # Check for completeness (should be complete sentences)
    if not any(p in answer for p in ".!?"):
        return False
    
    # Check that first character is capitalized (indicates proper sentence)
    if answer[0].islower():
        return False
    
    # Question-type specific checks
    question_lower = question.lower()
    
    # Definition questions should contain definitions
    if question_lower.startswith("what is") or question_lower.startswith("define "):
        if not any(phrase in answer_lower for phrase in [
            " is ", " are ", " means ", " refers to ", 
            " states that ", " can be described as ", ">"
        ]):
            return False
    
    # Overview questions should be comprehensive
    if "laws do you know" in question_lower or "principles do you know" in question_lower:
        if len(answer) < 150:  # Overviews should be substantial
            return False
        if "software engineering" not in answer_lower:
            return False
    
    # Thematic questions should mention the topic
    if any(phrase in question_lower for phrase in [
        "laws about ", "related to ", "deals with ", "talks about "
    ]):
        # Extract topic and check if it's mentioned
        topic = ""
        for phrase in ["laws about ", "related to ", "deals with ", "talks about "]:
            if phrase in question_lower:
                topic = question_lower.split(phrase)[1].split()[0]
                break
        if topic and topic not in answer_lower:
            return False
    
    return True

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_test()
        elif sys.argv[1] == "debug":
            run_debug_test()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python run_rag_test.py [quick|debug]")
    else:
        run_full_test()

if __name__ == "__main__":
    main()