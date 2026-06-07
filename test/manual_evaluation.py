#!/usr/bin/env python3
"""
Manual RAG System Evaluation
=============================

Runs comprehensive tests and lets you manually evaluate each answer.
Saves results with your ratings for tracking improvements.

Usage:
    python manual_evaluation.py          # Run test and manually rate answers
    python manual_evaluation.py analyze  # Analyze previous evaluations
"""

import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def run_manual_evaluation(run_number=1, changes=""):
    """Run comprehensive test and allow manual evaluation."""
    
    print(f"🧪 Manual Evaluation - Run #{run_number}")
    print("=" * 50)
    if changes:
        print(f"Changes since last run: {changes}")
    print("\nEvaluating 30+ questions across 8 categories...")
    print("For each answer, you'll be asked to rate it as: good/fair/poor")
    print("Type your rating and press Enter.\n")
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Comprehensive test suite with 30+ questions
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
        "Meta Questions": [
            "how many laws are in this document?",
            "what is this document about?",
            "who would benefit from reading this?",
            "what topics does this cover?"
        ]
    }
    
    results = []
    good_answers = 0
    total_questions = 0
    
    for category, questions in test_categories.items():
        print(f"\n{'='*60}")
        print(f"📋 {category}")
        print(f"{'='*60}")
        
        for i, question in enumerate(questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                answer = rag.query(question)
                print(f"\nA: {answer}")
                print("\n" + "-"*60)
                
                # Manual evaluation
                while True:
                    rating = input("Rate this answer (good/fair/poor): ").lower().strip()
                    if rating in ['good', 'fair', 'poor']:
                        break
                    print("Please enter 'good', 'fair', or 'poor'")
                
                # Get optional comments
                comments = input("Add comments (optional): ").strip()
                
                results.append({
                    'run': run_number,
                    'category': category,
                    'question': question,
                    'answer': answer,
                    'rating': rating,
                    'comments': comments
                })
                
                if rating == 'good':
                    good_answers += 1
                total_questions += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'run': run_number,
                    'category': category,
                    'question': question,
                    'answer': f"Error: {e}",
                    'rating': 'poor',
                    'comments': 'System error occurred'
                })
                total_questions += 1
    
    # Summary
    success_rate = (good_answers / total_questions * 100) if total_questions > 0 else 0
    print(f"\n{'='*60}")
    print(f"📊 Run #{run_number} Summary")
    print(f"{'='*60}")
    print(f"Total questions: {total_questions}")
    print(f"Good answers: {good_answers}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"manual_run_{run_number}_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'run_number': run_number,
            'timestamp': timestamp,
            'changes': changes,
            'total_questions': total_questions,
            'good_answers': good_answers,
            'success_rate': success_rate,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return results_file, success_rate

def analyze_manual_results():
    """Analyze all manual evaluation results."""
    print("📊 Analyzing Manual Evaluation Results")
    print("=" * 50)
    
    # Find all manual run files
    run_files = [f for f in os.listdir('.') if f.startswith('manual_run_') and f.endswith('.json')]
    run_files.sort()
    
    if not run_files:
        print("No manual evaluation runs found.")
        return
    
    print(f"Found {len(run_files)} manual evaluation runs:\n")
    
    for run_file in run_files:
        with open(run_file, 'r') as f:
            data = json.load(f)
            print(f"Run #{data['run_number']} ({data['timestamp']}):")
            print(f"  Success rate: {data['success_rate']:.1f}%")
            print(f"  Questions: {data['total_questions']}")
            print(f"  Good answers: {data['good_answers']}")
            print(f"  Changes: {data['changes']}")
            
            # Show rating distribution
            ratings = {'good': 0, 'fair': 0, 'poor': 0}
            for result in data['results']:
                ratings[result['rating']] += 1
            
            print(f"  Ratings: {ratings['good']} good, {ratings['fair']} fair, {ratings['poor']} poor")
            print()

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_manual_results()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python manual_evaluation.py [analyze]")
    else:
        # Check if this is the first run
        run_files = [f for f in os.listdir('.') if f.startswith('manual_run_') and f.endswith('.json')]
        next_run = len(run_files) + 1
        
        changes = ""
        if next_run > 1:
            changes = input(f"What changes were made since run #{next_run-1}? (or press enter for none): ")
        
        run_manual_evaluation(next_run, changes)

if __name__ == "__main__":
    main()