#!/usr/bin/env python3
"""
RAG System for YouTube Subtitles
- Load subtitles, split into chunks, and generate embeddings.
- Use FAISS for vector storage and Mistral/Devstral for generation.
- Provide a REPL for querying.
"""

import os
import re
from typing import List

# Install required packages if not already installed
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    print("Installing required packages...")
    import subprocess

    subprocess.run(["pip", "install", "faiss-cpu", "sentence-transformers", "transformers", "torch"], check=True)
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class RAGSystem:
    def __init__(self, subtitles_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG system with subtitles and embedding model."""
        self.subtitles_path = subtitles_path
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings = None
        self.index = None

        # Load and preprocess subtitles
        self.load_subtitles()
        self.split_chunks()
        self.generate_embeddings()
        self.build_index()

        # Initialize Mistral/Devstral for generation
        self.generator = self.init_generator()

    def load_subtitles(self):
        """Load subtitles from a file."""
        with open(self.subtitles_path, "r", encoding="utf-8") as f:
            self.subtitles = f.read()

    def split_chunks(self, chunk_size: int = 200):
        """Split subtitles into chunks of roughly `chunk_size` characters."""
        # Clean and split text into sentences/paragraphs
        text = re.sub(r"\s+", " ", self.subtitles).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += (" " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    self.chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            self.chunks.append(current_chunk)

    def generate_embeddings(self):
        """Generate embeddings for each chunk."""
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def build_index(self):
        """Build FAISS index for fast similarity search."""
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def init_generator(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        """Initialize the Mistral/Devstral generator."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception:
            print(f"Failed to load {model_id}. Falling back to a smaller model.")
            fallback = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModelForCausalLM.from_pretrained(fallback)
            return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top-k chunks relevant to the query."""
        if not self.chunks:
            return []

        k = min(k, len(self.chunks))
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0] if i != -1]

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate an answer using the retrieved context."""
        context_str = "\n".join(context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        result = self.generator(prompt, max_new_tokens=256, num_return_sequences=1)
        return result[0]["generated_text"]

    def query(self, question: str) -> str:
        """Answer a question using the RAG system."""
        context = self.retrieve(question)
        return self.generate_answer(question, context)


def repl(rag: RAGSystem):
    """Simple REPL for querying the RAG system."""
    print("RAG System REPL. Type 'exit' to quit.")
    while True:
        try:
            query = input(">>> ")
            if query.lower() == "exit":
                break
            answer = rag.query(query)
            print(answer)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG System for YouTube Subtitles")
    parser.add_argument("subtitles", type=str, help="Path to the subtitles file")
    args = parser.parse_args()

    if not os.path.exists(args.subtitles):
        print(f"Error: Subtitles file '{args.subtitles}' not found.")
        return

    rag = RAGSystem(args.subtitles)
    repl(rag)


if __name__ == "__main__":
    main()
