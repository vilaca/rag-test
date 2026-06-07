#!/usr/bin/env python3
"""Debug script to check if DRY is in the chunks."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_system import RAGSystem

def debug_dry_chunks():
    """Check if DRY content is properly chunked."""
    print("Checking DRY content in chunks...")
    
    # Initialize system with the software engineering laws document
    doc_path = "/Users/vilaca/work/rag-test/../tw/n-software-engineering-laws/n-software-engineering-laws.md"
    
    try:
        rag = RAGSystem(doc_path)
        rag.load_content()
        rag.split_chunks()
        
        print(f"Loaded {len(rag.chunks)} chunks")
        
        # Search for DRY in chunks
        dry_chunks = []
        for i, chunk in enumerate(rag.chunks):
            if "DRY" in chunk or "Don't Repeat Yourself" in chunk or "don't repeat yourself" in chunk:
                dry_chunks.append((i, chunk))
                print(f"\nFound DRY in chunk {i}:")
                print(chunk[:300] + "...")
        
        print(f"\nFound {len(dry_chunks)} chunks containing DRY content")
        
        if dry_chunks:
            print("✅ DRY content is properly chunked")
        else:
            print("❌ DRY content not found in chunks - chunking issue!")
            
            # Let's check if the content contains DRY at all
            if "DRY" in rag.content or "Don't Repeat Yourself" in rag.content:
                print("✅ DRY is in the original content")
                
                # Find where DRY appears in the content
                lines = rag.content.split('\n')
                for line_num, line in enumerate(lines):
                    if "DRY" in line or "Don't Repeat Yourself" in line:
                        print(f"DRY found at line {line_num}: {line[:100]}")
            else:
                print("❌ DRY not found in original content")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dry_chunks()
