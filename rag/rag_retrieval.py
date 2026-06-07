"""Retrieval and indexing mixin for the RAG system."""

import re
from typing import List, Dict, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class RetrievalMixin:
    def split_chunks(self, chunk_size: int = 512, overlap: int = 100, use_semantic_chunking: bool = True):
        """Split content into overlapping chunks for better context preservation."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 5)

        # Keep paragraph structure first (don't collapse newlines too early)
        raw_text = self.content.replace("\r\n", "\n").replace("\r", "\n")
        
        # Check if this looks like a structured document with headings
        if self._is_structured_document(raw_text):
            chunks = self._split_structured_document(raw_text, chunk_size, overlap)
        elif use_semantic_chunking:
            # Semantic-aware chunking based on sentences and paragraphs
            chunks = self._split_semantic_chunks(raw_text, chunk_size, overlap)
        else:
            # Original subtitle/transcript chunking logic
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
    
    def _split_semantic_chunks(self, text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        """Split text into semantic chunks based on sentences and paragraphs."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        
        for para in paragraphs:
            # Split paragraph into sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            if not sentences:
                sentences = [para]
            
            current_chunk = ""
            for sentence in sentences:
                # Try to keep complete sentences together
                candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
                
                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        # Sentence is too long, split it
                        words = sentence.split()
                        current_chunk = ""
                        for word in words:
                            test_chunk = (current_chunk + " " + word).strip() if current_chunk else word
                            if len(test_chunk) <= chunk_size:
                                current_chunk = test_chunk
                            else:
                                chunks.append(current_chunk)
                                current_chunk = word
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = ""
            
            if current_chunk:
                chunks.append(current_chunk)
        
        return chunks

    def _is_structured_document(self, text: str) -> bool:
        """Check if text appears to be a structured document with headings."""
        # Look for markdown heading patterns
        heading_pattern = r"^(#{1,6}\s+.+|##\s+.+|###\s+.+)"
        
        lines = text.split('\n')
        heading_count = 0
        
        for line in lines[:50]:  # Check first 50 lines
            if re.match(heading_pattern, line.strip()):
                heading_count += 1
                if heading_count >= 3:  # At least 3 headings suggests structured doc
                    return True
        
        return False

    def _split_structured_document(self, text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
        """Split structured documents while preserving heading-content relationships."""
        chunks = []
        
        # Split by major sections first
        sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Split section into heading and content
            lines = section.split('\n')
            if not lines:
                continue
            
            heading_line = lines[0].strip()
            content_lines = lines[1:]
            
            # Add heading as its own chunk (important for retrieval)
            if heading_line:
                chunks.append(heading_line)
            
            # Process content in logical blocks
            current_block = ""
            for line in content_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for subheadings or new structural elements
                if re.match(r"^(#{4,6}|\*\*|>)", line):
                    if current_block:
                        chunks.append(current_block.strip())
                        current_block = ""
                    # Add structural elements as separate chunks
                    chunks.append(line)
                else:
                    # Accumulate content
                    if current_block:
                        candidate = f"{current_block} {line}"
                        if len(candidate) <= chunk_size:
                            current_block = candidate
                        else:
                            chunks.append(current_block.strip())
                            current_block = line
                    else:
                        current_block = line
            
            if current_block:
                chunks.append(current_block.strip())
        
        return chunks

    def generate_embeddings(self):
        """Generate embeddings for each chunk."""
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        # Normalize for cosine-similarity style retrieval
        faiss.normalize_L2(self.embeddings)
        
        # Initialize BM25 for hybrid retrieval
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 for sparse retrieval."""
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

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
        """Retrieve top-k chunks relevant to the query using hybrid retrieval."""
        self.last_retrieval_debug = []
        if not self.chunks or self.index is None:
            return []

        k = min(k, len(self.chunks))
        
        # Dense retrieval using FAISS
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        dense_scores, dense_indices = self.index.search(query_embedding, k)
        
        # Sparse retrieval using BM25
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]  # Top k from BM25
        
        # Combine results from both retrieval methods
        combined_scores, combined_indices = self._combine_retrieval_results(
            dense_scores, dense_indices, bm25_scores, bm25_indices, k
        )

        # Apply keyword-based re-ranking to boost exact matches
        ranked_chunks, debug_info = self._re_rank_with_keywords(query, combined_scores, combined_indices)
        self.last_retrieval_debug = debug_info

        # Filter out chunks that are not relevant to the query
        ranked_chunks = self._filter_relevant_chunks(query, ranked_chunks)

        # Add contextual chunks if we found headings but not content
        ranked_chunks = self._add_contextual_chunks(query, ranked_chunks, combined_indices, combined_scores)

        return ranked_chunks
    
    def _combine_retrieval_results(self, dense_scores, dense_indices, bm25_scores, bm25_indices, k):
        """Combine results from dense and sparse retrieval using reciprocal rank fusion."""
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Add dense retrieval results
        for i, idx in enumerate(dense_indices[0]):
            if idx != -1:
                combined_scores[idx] = combined_scores.get(idx, 0.0) + 1.0 / (i + 1)
        
        # Add BM25 results
        for i, idx in enumerate(bm25_indices):
            if idx != -1:
                combined_scores[idx] = combined_scores.get(idx, 0.0) + 1.0 / (i + 1)
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
        sorted_scores = np.array([combined_scores[idx] for idx in sorted_indices])
        
        return sorted_scores.reshape(1, -1), np.array([sorted_indices])

    def _add_contextual_chunks(self, query: str, ranked_chunks: list, indices, scores) -> list:
        """Add neighboring chunks when headings are found but content is missing."""
        import re
        
        # Check if we have headings in the top results
        heading_indices = []
        for i, chunk in enumerate(ranked_chunks[:3]):  # Check top 3
            if re.match(r"^#{2,6}\s", chunk):
                # Find the index of this heading in the original chunks
                for j, orig_chunk in enumerate(self.chunks):
                    if orig_chunk == chunk:
                        heading_indices.append((j, i))  # (original_idx, rank)
                        break
        
        if not heading_indices:
            return ranked_chunks
        
        # Only add context for the highest-ranked heading (most relevant)
        # Sort by rank (lower rank = more relevant)
        heading_indices.sort(key=lambda x: x[1])
        best_heading_idx = heading_indices[0][0]
        
        # Add the next few chunks after the best heading only
        additional_chunks = []
        for i in range(1, 4):  # Next 3 chunks
            context_idx = best_heading_idx + i
            if context_idx < len(self.chunks):
                context_chunk = self.chunks[context_idx]
                # Avoid duplicates
                if context_chunk not in ranked_chunks and context_chunk not in additional_chunks:
                    # Prioritize quote blocks and content over source/bibliographic info
                    if re.match(r"^>", context_chunk) or not context_chunk.strip().startswith("**"):
                        additional_chunks.append(context_chunk)
                        if len(additional_chunks) >= 3:  # Limit additional chunks
                            break
                    elif len(additional_chunks) < 2:  # Allow one source chunk if needed
                        additional_chunks.append(context_chunk)
        
        if additional_chunks:
            # Add them to the end of the ranked chunks
            ranked_chunks.extend(additional_chunks)
            
            # Update debug info for additional chunks
            for i, chunk in enumerate(additional_chunks):
                self.last_retrieval_debug.append({
                    "rank": len(ranked_chunks) - len(additional_chunks) + i + 1,
                    "index": "contextual",
                    "original_score": 0.0,
                    "keyword_boost": 0.5,
                    "final_score": 0.5,
                    "chunk": chunk,
                })
        
        return ranked_chunks

    def _filter_relevant_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """Filter chunks to ensure they are relevant to the query."""
        if not chunks or len(chunks) <= 1:
            return chunks
        
        # Extract meaningful keywords from query (exclude stop words)
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "explain", "describe", "what"}
        query_keywords = [word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2]
        
        if not query_keywords:
            return chunks
        
        # For each chunk, calculate a relevance score
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            
            # Count how many query keywords appear in the chunk
            keyword_matches = sum(1 for keyword in query_keywords if keyword in chunk_lower)
            
            # Calculate relevance score (0-1)
            relevance_score = keyword_matches / len(query_keywords)
            
            # Also check for related terms
            related_score = 1.0 if self._contains_related_terms(chunk_lower, query_keywords) else 0.0
            
            # Combined score
            final_score = relevance_score + (related_score * 0.3)
            
            scored_chunks.append((final_score, chunk))
        
        # Filter out low-relevance chunks and sort by relevance
        min_relevance = 0.3  # Minimum relevance threshold
        filtered_chunks = [chunk for score, chunk in scored_chunks if score >= min_relevance]
        
        # Sort by relevance (highest first)
        filtered_chunks.sort(key=lambda x: next(score for score, chunk in scored_chunks if chunk == x), reverse=True)
        
        return filtered_chunks if filtered_chunks else chunks
    
    def _contains_related_terms(self, chunk: str, keywords: List[str]) -> bool:
        """Check if chunk contains terms related to the keywords."""
        # Simple synonym/related term checking
        related_terms = {
            "cap": ["consistency", "availability", "partition", "theorem"],
            "theorem": ["principle", "law", "rule", "concept"],
            "explain": ["describe", "definition", "means", "states"],
            "distributed": ["system", "network", "nodes", "cluster"],
            "system": ["architecture", "design", "model"]
        }
        
        chunk_words = set(chunk.split())
        for keyword in keywords:
            if keyword in related_terms:
                if any(term in chunk_words for term in related_terms[keyword]):
                    return True
        
        return False

    def _re_rank_with_keywords(self, query: str, scores, indices) -> tuple:
        """Re-rank results using keyword matching to boost exact matches."""
        import re
        
        # Extract key terms from query (excluding stop words)
        stop_words = {"explain", "describe", "what", "is", "are", "the", "a", "an", "of", "and", "or", "to", "in", "for"}
        query_lower = query.lower()
        
        # Extract multi-word phrases first (like "Conway's Law")
        key_phrases = []
        words = query_lower.split()
        
        # Look for 2-3 word phrases that contain meaningful terms
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if any(word not in stop_words for word in phrase.split()):
                key_phrases.append(phrase)
            
            if i < len(words) - 2:
                phrase3 = f"{words[i]} {words[i+1]} {words[i+2]}"
                if any(word not in stop_words for word in phrase3.split()):
                    key_phrases.append(phrase3)
        
        # Also add single key words
        key_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Add synonyms for common question words
        # Map "define" to the actual term being defined
        if "define" in query_lower:
            # Find what's being defined
            for i, word in enumerate(words):
                if word == "define" and i + 1 < len(words):
                    defined_term = words[i + 1]
                    if defined_term.lower() not in stop_words:
                        key_words.append(defined_term)
                        break
        
        # Combine and deduplicate
        key_terms = list(set(key_phrases + key_words))
        
        # Re-rank results
        boosted_results = []
        seen_indices = set()
        
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1 or idx in seen_indices or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            original_score = float(scores[0][i])
            
            # Calculate keyword boost
            keyword_boost = 0.0
            chunk_lower = chunk.lower()
            
            # Penalize source/bibliographic chunks
            if chunk_lower.startswith("**source.**") or chunk_lower.startswith("**source:**"):
                keyword_boost -= 1.0  # Strong penalty for source sections
            elif "named " in chunk_lower and " by " in chunk_lower:
                keyword_boost -= 0.5  # Penalty for attribution phrases
            
            for term in key_terms:
                # Exact phrase match gets higher boost
                if f" {term} " in f" {chunk_lower} " or chunk_lower.startswith(term + " ") or chunk_lower.endswith(" " + term):
                    keyword_boost += 0.5
                # Partial match gets smaller boost
                elif term in chunk_lower:
                    keyword_boost += 0.2
            
            # Strong boost for heading matches
            if re.match(r"^#{1,6}\s", chunk):
                heading_text = re.sub(r"^#{1,6}\s*", "", chunk).split("\n")[0].lower()
                for term in key_terms:
                    if term in heading_text:
                        keyword_boost += 1.0
                        break
            
            # Boost for content chunks that follow headings (likely definitions)
            if re.match(r"^>", chunk):  # Quote blocks often contain definitions
                keyword_boost += 0.8
            elif chunk.strip().startswith("**Source.**"):  # Source chunks
                keyword_boost -= 0.5  # Penalize source chunks
            elif len(key_terms) == 1 and any(term.lower() in chunk.lower() for term in key_terms):
                # If single term matches and it's not a heading, give moderate boost
                keyword_boost += 0.3
            
            # Extra boost for quote blocks (definitions)
            if re.match(r"^>", chunk):
                keyword_boost += 0.8
            
            # Final score = original semantic score + keyword boost
            final_score = original_score + keyword_boost
            
            boosted_results.append((final_score, idx, original_score, keyword_boost))
            seen_indices.add(idx)
        
        # Sort by final score
        boosted_results.sort(key=lambda x: x[0], reverse=True)
        
        # Build final result list
        ranked_chunks = []
        debug_info = []
        max_results = min(12, len(boosted_results))  # Use fixed max like original k
        
        for i, (final_score, idx, original_score, keyword_boost) in enumerate(boosted_results[:max_results]):
            chunk = self.chunks[idx]
            ranked_chunks.append(chunk)
            debug_info.append({
                "rank": i + 1,
                "index": idx,
                "original_score": original_score,
                "keyword_boost": keyword_boost,
                "final_score": final_score,
                "chunk": chunk,
            })
        
        return ranked_chunks, debug_info

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
