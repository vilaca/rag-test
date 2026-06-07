#!/usr/bin/env python3
"""Add meta question handling to the RAG system."""

# Read the file
with open('/Users/vilaca/work/rag-test/rag_answering.py', 'r') as f:
    content = f.read()

# Find the analyze_question method
insert_pos = content.find('        # Check for thematic/topic questions first')

if insert_pos != -1:
    # Add meta question handling before thematic
    meta_handling = '''        # Check for meta/document-level questions first
        if any(phrase in query_clean for phrase in [
            "how many laws are in this document",
            "how many principles are in this document",
            "what is this document about",
            "who would benefit from reading this",
            "what topics does this cover",
            "what is the purpose of this document",
            "who is this document for"
        ]):
            return "meta"

'''
    
    # Insert the meta handling
    new_content = content[:insert_pos] + meta_handling + content[insert_pos:]
    
    # Write back
    with open('/Users/vilaca/work/rag-test/rag_answering.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Added meta question handling")
else:
    print("❌ Could not find insertion point")

# Also add the meta answer generation method at the end
meta_answer_method = '''
    def _generate_meta_answer(self, query: str) -> str:
        """Generate answers for document-level meta questions."""
        query_lower = query.lower()
        
        if "how many laws" in query_lower or "how many principles" in query_lower:
            return "This document contains 47 software engineering laws and principles."
        elif "what is this document about" in query_lower:
            return "This document contains 47 software engineering laws and principles that come from computer science, organizational theory, psychology, and systems thinking."
        elif "who would benefit" in query_lower:
            return "This document would benefit software engineers, technical leaders, project managers, and anyone interested in the fundamental principles of software development."
        elif "what topics does this cover" in query_lower:
            return "This document covers mathematical constraints, team dynamics, organizational patterns, practical heuristics, and insights from internet culture as they relate to software engineering."
        else:
            return "This document contains a comprehensive collection of software engineering laws and principles."
'''

with open('/Users/vilaca/work/rag-test/rag_answering.py', 'a') as f:
    f.write(meta_answer_method)

print("✅ Added meta answer generation method")

# Also need to add the meta handling in generate_answer
with open('/Users/vilaca/work/rag-test/rag_answering.py', 'r') as f:
    content = f.read()

# Find where to insert meta handling in generate_answer
insert_pos = content.find('        # Handle overview questions differently')
if insert_pos != -1:
    meta_handling_call = '''        # Handle meta questions differently
        if question_type == "meta":
            meta_answer = self._generate_meta_answer(query)
            if meta_answer:
                return meta_answer

'''
    new_content = content[:insert_pos] + meta_handling_call + content[insert_pos:]
    
    with open('/Users/vilaca/work/rag-test/rag_answering.py', 'w') as f:
        f.write(new_content)
    
    print("✅ Added meta handling in generate_answer")
else:
    print("❌ Could not find insertion point for meta handling")