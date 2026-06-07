#!/usr/bin/env python3
"""
RAG System Improvement Tracker
===============================

Tracks system performance across multiple runs and helps identify areas for improvement.
Automatically saves detailed results for each run with LLM evaluation.

Usage:
    python improvement_tracker.py          # Run test and save results
    python improvement_tracker.py analyze  # Analyze previous runs
"""

import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from rag_system import RAGSystem

def evaluate_answer_with_llm(question, answer):
    """Evaluate answer quality using LLM criteria."""
    
    # Check for obvious failures
    failure_patterns = [
        "i couldn't find",
        "the content doesn't provide",
        "try asking about",
        "no direct answer",
        "not found in this content"
    ]
    
    answer_lower = answer.lower()
    for pattern in failure_patterns:
        if pattern in answer_lower:
            return {
                'quality': 'poor',
                'reason': 'Failure response pattern detected',
                'suggestions': ['Improve retrieval for this question type', 
                              'Add fallback content', 
                              'Enhance answer synthesis']
            }
    
    # Check length
    if len(answer) < 40:
        return {
            'quality': 'poor',
            'reason': 'Answer too short',
            'suggestions': ['Expand answer with more context', 
                          'Check if retrieval found enough content']
        }
    
    if len(answer) > 600:
        return {
            'quality': 'fair',
            'reason': 'Answer too long - may contain redundant information',
            'suggestions': ['Improve answer synthesis to be more concise', 
                          'Add diversity filtering to avoid repetition']
        }
    
    # Check completeness
    if not any(p in answer for p in ".!?"):
        return {
            'quality': 'poor',
            'reason': 'Incomplete sentence detected',
            'suggestions': ['Ensure answers are complete sentences', 
                          'Check answer synthesis logic']
        }
    
    # Check capitalization
    if answer[0].islower():
        return {
            'quality': 'poor',
            'reason': 'Answer not properly capitalized',
            'suggestions': ['Fix answer formatting', 
                          'Check cleaning logic']
        }
    
    # Question-type specific evaluation
    question_lower = question.lower()
    
    # Definition questions
    if question_lower.startswith("what is") or question_lower.startswith("define "):
        if not any(phrase in answer_lower for phrase in [
            " is ", " are ", " means ", " refers to ", 
            " states that ", " can be described as ", ">"
        ]):
            return {
                'quality': 'fair',
                'reason': 'Definition pattern not clearly present',
                'suggestions': ['Improve definition detection', 
                              'Enhance retrieval for definitions']
            }
    
    # Overview questions
    if "laws do you know" in question_lower or "principles do you know" in question_lower:
        if len(answer) < 150:
            return {
                'quality': 'fair',
                'reason': 'Overview answer too brief',
                'suggestions': ['Expand overview content', 
                              'Add more introductory material']
            }
    
    # Thematic questions
    if any(phrase in question_lower for phrase in [
        "laws about ", "related to ", "deals with ", "talks about "
    ]):
        # Extract topic and check if mentioned
        topic = ""
        for phrase in ["laws about ", "related to ", "deals with ", "talks about "]:
            if phrase in question_lower:
                topic = question_lower.split(phrase)[1].split()[0]
                break
        if topic and topic not in answer_lower:
            return {
                'quality': 'poor',
                'reason': 'Topic not mentioned in answer',
                'suggestions': ['Improve thematic retrieval', 
                              'Check topic matching logic']
            }
    
    # If all checks pass
    return {
        'quality': 'good',
        'reason': 'Answer meets all quality criteria',
        'suggestions': ['Continue monitoring', 'Consider minor refinements']
    }

def run_improvement_test(run_number=1, changes=""):
    """Run comprehensive test and save results."""
    
    print(f"🚀 Running Improvement Test - Run #{run_number}")
    print("=" * 50)
    if changes:
        print(f"Changes since last run: {changes}")
    print()
    
    # Initialize system
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    rag = RAGSystem(doc_path)
    rag.load_subtitles()
    rag.split_chunks()
    rag.generate_embeddings()
    rag.build_index()
    
    # Test questions - COMPREHENSIVE 30+ QUESTION TEST SUITE
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
    
    results = []
    good_answers = 0
    total_questions = 0
    
    for category, questions in test_categories.items():
        print(f"\n📋 {category}")
        print("-" * len(category))
        
        for question in questions:
            print(f"\nQ: {question}")
            
            try:
                answer = rag.query(question)
                print(f"A: {answer}")
                
                # Evaluate with LLM criteria
                evaluation = evaluate_answer_with_llm(question, answer)
                print(f"Evaluation: {evaluation['quality'].upper()}")
                print(f"Reason: {evaluation['reason']}")
                
                results.append({
                    'run': run_number,
                    'category': category,
                    'question': question,
                    'answer': answer,
                    'evaluation': evaluation
                })
                
                if evaluation['quality'] == 'good':
                    good_answers += 1
                total_questions += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'run': run_number,
                    'category': category,
                    'question': question,
                    'answer': f"Error: {e}",
                    'evaluation': {
                        'quality': 'poor',
                        'reason': 'System error',
                        'suggestions': ['Fix the error', 'Improve error handling']
                    }
                })
                total_questions += 1
    
    # Summary
    success_rate = (good_answers / total_questions * 100) if total_questions > 0 else 0
    print(f"\n📊 Run #{run_number} Summary")
    print("=" * 50)
    print(f"Total questions: {total_questions}")
    print(f"Good answers: {good_answers}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"run_{run_number}_{timestamp}.json"
    
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

def analyze_previous_runs():
    """Analyze all previous test runs."""
    print("📊 Analyzing Previous Test Runs")
    print("=" * 40)
    
    # Find all run files
    run_files = [f for f in os.listdir('.') if f.startswith('run_') and f.endswith('.json')]
    run_files.sort()
    
    if not run_files:
        print("No previous runs found.")
        return
    
    print(f"Found {len(run_files)} previous runs:\n")
    
    for run_file in run_files:
        with open(run_file, 'r') as f:
            data = json.load(f)
            print(f"Run #{data['run_number']} ({data['timestamp']}):")
            print(f"  Success rate: {data['success_rate']:.1f}%")
            print(f"  Changes: {data['changes']}")
            print(f"  Questions: {data['total_questions']}")
            print(f"  Good answers: {data['good_answers']}")
            print()

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_previous_runs()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python improvement_tracker.py [analyze]")
    else:
        # Check if this is the first run
        run_files = [f for f in os.listdir('.') if f.startswith('run_') and f.endswith('.json')]
        next_run = len(run_files) + 1
        
        changes = ""
        if next_run > 1:
            changes = input(f"What changes were made since run #{next_run-1}? (or press enter for none): ")
        
        run_improvement_test(next_run, changes)

if __name__ == "__main__":
    main()