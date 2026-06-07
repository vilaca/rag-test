#!/usr/bin/env python3
"""Comprehensive test suite for RAG system evaluation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def run_comprehensive_test():
    """Run comprehensive test suite and evaluate performance."""
    print("🧪 Comprehensive RAG System Test Suite")
    print("=" * 60)
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test categories
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
    
    # Run tests and collect results
    results = {}
    total_questions = 0
    good_answers = 0
    
    for category, questions in test_categories.items():
        print(f"\n📋 {category}")
        print("-" * len(category))
        
        category_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                answer = rag.query(question)
                print(f"A: {answer}")
                
                # Simple evaluation (could be enhanced with more sophisticated metrics)
                is_good = len(answer) > 50 and len(answer) < 500  # Reasonable length
                
                if is_good:
                    print("✅ Good answer")
                    good_answers += 1
                else:
                    print("❌ Needs improvement")
                    
                category_results.append({
                    "question": question,
                    "answer": answer,
                    "good": is_good
                })
                
                total_questions += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                category_results.append({
                    "question": question,
                    "answer": f"Error: {e}",
                    "good": False
                })
                total_questions += 1
        
        results[category] = category_results
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    print(f"Total questions tested: {total_questions}")
    print(f"Good answers: {good_answers}")
    print(f"Success rate: {good_answers/total_questions*100:.1f}%")
    
    # Detailed results by category
    print(f"\n📈 Results by Category:")
    for category, category_results in results.items():
        good_in_category = sum(1 for r in category_results if r["good"])
        total_in_category = len(category_results)
        success_rate = (good_in_category / total_in_category * 100) if total_in_category > 0 else 0
        print(f"  {category}: {good_in_category}/{total_in_category} ({success_rate:.1f}%)")
    
    # Save detailed results for analysis
    with open("test_results.txt", "w") as f:
        f.write("Comprehensive RAG System Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        for category, category_results in results.items():
            f.write(f"📋 {category}\n")
            f.write("-" * len(category) + "\n\n")
            
            for i, result in enumerate(category_results, 1):
                f.write(f"Q{i}: {result['question']}\n")
                f.write(f"A: {result['answer']}\n")
                f.write(f"Status: {'✅ Good' if result['good'] else '❌ Needs improvement'}\n")
                f.write("-" * 50 + "\n")
            
            f.write("\n")
        
        f.write(f"📊 Summary\n")
        f.write(f"Total questions: {total_questions}\n")
        f.write(f"Good answers: {good_answers}\n")
        f.write(f"Success rate: {good_answers/total_questions*100:.1f}%\n")
    
    print(f"\n💾 Detailed results saved to test_results.txt")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()