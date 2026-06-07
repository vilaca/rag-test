"""Retrieval and indexing mixin for the RAG system."""

import re
from typing import List

import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class RetrievalMixin:
    def split_chunks(self, chunk_size: int = 512, overlap: int = 100):
        """Split subtitles into overlapping chunks for better context preservation."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 5)

        # Keep paragraph structure first (don't collapse newlines too early)
        raw_text = self.subtitles.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]

        if not paragraphs:
            paragraphs = [raw_text.strip()] if raw_text.strip() else []

        chunks = []
        step = max(1, chunk_size - overlap)

        for para in paragraphs:
            # Normalize whitespace inside each paragraph only
            para = re.sub(r"\s+", " ", para).strip()
            if not para:
                continue

            # Split paragraph into sentences, but handle poorly punctuated transcripts too
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            if not sentences:
                sentences = [para]

            current_chunk = ""
            for sentence in sentences:
                # Hard-split extremely long sentences that exceed chunk size
                if len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    for i in range(0, len(sentence), step):
                        piece = sentence[i : i + chunk_size].strip()
                        if piece:
                            chunks.append(piece)
                    continue

                candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if overlap > 0 and chunks:
                        # Build overlap from full trailing words, not raw characters.
                        prev_words = chunks[-1].split()
                        overlap_words = []
                        char_count = 0
                        for w in reversed(prev_words):
                            next_len = char_count + len(w) + (1 if overlap_words else 0)
                            if next_len > overlap and overlap_words:
                                break
                            overlap_words.append(w)
                            char_count = next_len
                        overlap_text = " ".join(reversed(overlap_words)).strip()

                        current_chunk = (overlap_text + " " + sentence).strip() if overlap_text else sentence
                        if len(current_chunk) > chunk_size:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk)

        self.chunks = chunks

    def generate_embeddings(self):
        """Generate embeddings for each chunk."""
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        # Normalize for cosine-similarity style retrieval
        faiss.normalize_L2(self.embeddings)

    def build_index(self):
        """Build FAISS index for fast similarity search."""
        dimension = self.embeddings.shape[1]
        # With normalized vectors, inner product ~= cosine similarity
        base_index = faiss.IndexFlatIP(dimension)
        base_index.add(self.embeddings)

        if self.use_mmap_index:
            try:
                faiss.write_index(base_index, self.index_file)
                self.index = faiss.read_index(self.index_file, faiss.IO_FLAG_MMAP)
            except Exception as e:
                print(f"Warning: failed to memory-map index ({e}). Using in-memory index.")
                self.index = base_index
        else:
            self.index = base_index

    def init_generator(self, model_id: str = "distilgpt2"):
        """Initialize the generator with DistilGPT2 as default."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Failed to load {model_id}: {str(e)}. Falling back to a smaller model.")
            fallback = "gpt2"  # Fallback to original GPT-2 if DistilGPT2 fails
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback)
                model = AutoModelForCausalLM.from_pretrained(fallback)
                return pipeline("text-generation", model=model, tokenizer=tokenizer)
            except Exception as e2:
                print(f"Failed to load fallback model {fallback}: {str(e2)}")
                raise RuntimeError("Could not initialize any text generation model.")

    def retrieve(self, query: str, k: int = 12) -> List[str]:
        """Retrieve top-k chunks relevant to the query."""
        self.last_retrieval_debug = []
        if not self.chunks or self.index is None:
            return []

        k = min(k, len(self.chunks))
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        # Keep FAISS ranking order and remove duplicates by index only.
        ranked_chunks: List[str] = []
        seen_indices = set()
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1 or idx in seen_indices or idx >= len(self.chunks):
                continue
            seen_indices.add(idx)

            chunk = self.chunks[idx]
            ranked_chunks.append(chunk)
            self.last_retrieval_debug.append(
                {
                    "rank": len(ranked_chunks),
                    "index": idx,
                    "score": float(scores[0][i]),
                    "chunk": chunk,
                }
            )
            if len(ranked_chunks) >= k:
                break

        return ranked_chunks

    def _get_diverse_indices(self, indices: List[int], max_count: int = 4) -> List[int]:
        """Get a diverse set of chunk indices to avoid repetitive answers."""
        if len(indices) <= max_count:
            return indices

        # Simple diversity: pick some from beginning, middle, and end
        step = max(1, len(indices) // max_count)
        diverse_indices = []
        for i in range(0, len(indices), step):
            if len(diverse_indices) >= max_count:
                break
            diverse_indices.append(indices[i])

        return diverse_indices
