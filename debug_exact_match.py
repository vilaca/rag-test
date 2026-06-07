#!/usr/bin/env python3
"""Debug script to understand exact matching in re-ranking."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def debug_exact_match():
    """Debug the exact matching logic."""
    print("Debugging exact matching...")
    
    # Test DRY chunks
    dry_chunks = [
        "### 31. DRY (Don't Repeat Yourself)",
        "DRY](#31-dry-dont-repeat-yourself) - One authoritative representation",
        "The principle is often misread as 'don't repeat code.' Hunt and Thomas were stricter: don't repeat *knowledge*.",
        "**Caveats.** Aggressive DRY is one of the leading sources of bad abstractions."
    ]
    
    term = "dry"
    
    for i, chunk in enumerate(dry_chunks):
        chunk_lower = chunk.lower()
        
        # Test exact phrase match (current logic)
        exact_match = f" {term} " in f" {chunk_lower} " or chunk_lower.startswith(term + " ") or chunk_lower.endswith(" " + term)
        
        # Test partial match
        partial_match = term in chunk_lower
        
        print(f"\nChunk {i+1}: {chunk[:80]}...")
        print(f"  Exact match: {exact_match}")
        print(f"  Partial match: {partial_match}")
        print(f"  Chunk lower: {chunk_lower[:80]}...")
        
        # Show where "dry" appears
        if term in chunk_lower:
            index = chunk_lower.find(term)
            context = chunk_lower[max(0, index-10):index+len(term)+10]
            print(f"  Context around 'dry': '{context}'")

if __name__ == "__main__":
    debug_exact_match()
