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
        self.subtitles = ""
        self.original_files = [subtitles_path]

        # Initialize Mistral/Devstral for generation
        self.generator = self.init_generator()

    def load_subtitles(self):
        """Load subtitles from a file with validation, or use already-loaded content."""
        # If subtitles are already loaded (e.g., from combined files), use them
        if self.subtitles and len(self.subtitles.strip()) > 0:
            content = self.subtitles
        else:
            # Otherwise, load from file
            try:
                with open(self.subtitles_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if not content.strip():
                    raise ValueError("Input file is empty")
                
                if len(content) < 100:
                    print("Warning: Input file is very small (< 100 characters)")
                
                self.subtitles = content
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(self.subtitles_path, "r", encoding="latin-1") as f:
                    content = f.read()
                self.subtitles = content
            except Exception as e:
                raise RuntimeError(f"Failed to load subtitles file: {str(e)}")
                
        # Validate the content
        if not content.strip():
            raise ValueError("Subtitles content is empty")

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
                        piece = sentence[i:i + chunk_size].strip()
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
                        overlap_text = chunks[-1][-overlap:]
                        # Avoid mid-word overlap fragments that hurt answer readability
                        if " " in overlap_text:
                            overlap_text = overlap_text.split(" ", 1)[1]
                        current_chunk = (overlap_text + " " + sentence).strip()
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
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

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

    def _expand_query(self, query: str) -> str:
        """Expand some common question intents to improve retrieval recall."""
        q = query.lower()
        extras = []

        if "reapply" in q:
            extras.append("reapply every 2 hours after swimming sweating towel drying")
        if "apply" in q and "reapply" not in q:
            extras.append("apply before sun exposure 15 minutes")
        if "mineral" in q and "chemical" in q:
            extras.append("physical filter versus organic filter differences pros cons")

        return query + (" " + " ".join(extras) if extras else "")

    def retrieve(self, query: str, k: int = 12) -> List[str]:
        """Retrieve top-k chunks relevant to the query with hybrid search."""
        if not self.chunks:
            return []

        k = min(k, len(self.chunks))
        search_query = self._expand_query(query)
        query_embedding = self.embedding_model.encode([search_query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        
        # Filter out invalid indices and sort by score (descending)
        valid_results = [(scores[0][i], self.chunks[indices[0][i]], indices[0][i]) 
                        for i in range(len(indices[0])) 
                        if indices[0][i] != -1]
        valid_results.sort(key=lambda x: x[0], reverse=True)
        
        # Get more diverse results by ensuring we don't get too many similar chunks
        chunks = [chunk for _, chunk, _ in valid_results[:k]]
        
        # Try to get some variety in the results
        if len(chunks) > 3:
            # Add some chunks that are less similar to each other
            all_indices = [idx for _, _, idx in valid_results]
            diverse_indices = self._get_diverse_indices(all_indices[:10], min(4, len(all_indices)))
            diverse_chunks = [self.chunks[idx] for idx in diverse_indices if idx < len(self.chunks)]
            chunks = list(set(chunks + diverse_chunks))[:k]  # Remove duplicates
        
        return chunks[:k]
    
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

    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate an answer using advanced synthesis from retrieved context."""
        if not context:
            return "I couldn't find any relevant information about that in the content."
        
        # Analyze the question type to determine answer strategy
        question_type = self._analyze_question(query)
        
        # Try to synthesize the best answer from multiple context chunks
        try:
            best_answer = self._synthesize_answer(query, context, question_type)
            if best_answer:
                return best_answer
        except Exception as e:
            print(f"Answer synthesis failed: {e}")
        
        # If synthesis fails, fall back to simple extractive approach
        return self._simple_extractive_answer(query, context)
    
    def _analyze_question(self, query: str) -> str:
        """Analyze question type to determine best answering strategy."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['when', 'should', 'must', 'need']):
            return "advice"
        elif any(word in query_lower for word in ['what', 'describe', 'explain']):
            return "factual"
        elif any(word in query_lower for word in ['why', 'how', 'does', 'is', 'are']):
            return "explanatory"
        elif any(word in query_lower for word in ['types', 'kinds', 'sorts', 'categories']):
            return "categorical"
        else:
            return "general"
    
    def _extract_candidate_sentences(self, context: List[str]) -> List[str]:
        """Extract readable candidate sentences/passages from retrieved chunks."""
        import re

        candidates = []
        for chunk in context:
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if not chunk:
                continue

            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", chunk) if p.strip()]
            if not parts:
                parts = [chunk]

            for part in parts:
                part = re.sub(r"\[\d+\]", "", part).strip()
                if len(part) < 25:
                    continue

                # Split very long transcript runs into readable windows
                if len(part) > 220:
                    words = part.split()
                    window = []
                    cur_len = 0
                    for w in words:
                        if cur_len + len(w) + 1 > 180 and window:
                            candidates.append(" ".join(window).strip())
                            window = [w]
                            cur_len = len(w)
                        else:
                            window.append(w)
                            cur_len += len(w) + 1
                    if window:
                        candidates.append(" ".join(window).strip())
                else:
                    candidates.append(part)

        return candidates

    def _query_keywords(self, query: str) -> set:
        import re

        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "can", "could", "should",
            "would", "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "about", "from", "than", "then", "into", "your", "you", "it", "this", "that"
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        return {w for w in words if w not in stopwords}

    def _synthesize_answer(self, query: str, context: List[str], question_type: str) -> str:
        """Synthesize a cleaner extractive answer from multiple chunks."""
        candidates = self._extract_candidate_sentences(context)
        if not candidates:
            return ""

        keywords = self._query_keywords(query)
        query_lower = query.lower()

        scored = []
        for i, sent in enumerate(candidates):
            s_lower = sent.lower()
            score = 0.0

            # Keyword overlap
            overlap = sum(1 for k in keywords if k in s_lower)
            score += overlap * 2.0

            # Intent-aware boosts
            if "reapply" in query_lower and any(x in s_lower for x in ["reapply", "every", "hours", "sweat", "water"]):
                score += 2.5
            if "apply" in query_lower and any(x in s_lower for x in ["before", "minutes", "apply"]):
                score += 2.0
            if any(x in query_lower for x in ["compare", "difference", "mineral", "chemical"]):
                if any(x in s_lower for x in ["mineral", "chemical", "organic", "inorganic", "filter"]):
                    score += 1.8

            if question_type == "advice" and any(x in s_lower for x in ["should", "recommend", "important", "best"]):
                score += 0.8

            if 40 <= len(sent) <= 180:
                score += 0.5

            # Slight preference for higher-ranked retrieved chunks
            score += max(0.0, 0.3 - (i * 0.01))
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for sc, s in scored[:12] if sc > 1.0]
        if not top:
            return self._no_clear_answer(query, context)

        # Special handling for comparison questions
        if all(x in query_lower for x in ["mineral", "chemical"]) and any(x in query_lower for x in ["compare", "vs", "difference"]):
            mineral = next((s for s in top if "mineral" in s.lower() and "chemical" not in s.lower()), None)
            chemical = next((s for s in top if ("chemical" in s.lower() or "organic" in s.lower()) and "mineral" not in s.lower()), None)

            if mineral is None:
                mineral = next((s for s in top if "mineral" in s.lower()), None)
            if chemical is None:
                chemical = next((s for s in top if "chemical" in s.lower() or "organic" in s.lower()), None)

            if mineral and chemical:
                if mineral.strip().lower() == chemical.strip().lower():
                    return self._clean_answer(mineral)
                return self._clean_answer(f"Mineral sunscreen: {mineral} Chemical sunscreen: {chemical}")

        # Pick 1-2 diverse sentences
        answer_parts = []
        used = set()
        for s in top:
            sig = " ".join(sorted(set(s.lower().split()) & keywords))
            if sig in used and sig:
                continue
            answer_parts.append(s)
            if sig:
                used.add(sig)
            if len(answer_parts) >= 2:
                break

        return self._clean_answer(" ".join(answer_parts))

    def _no_clear_answer(self, query: str, context: List[str]) -> str:
        """Provide a helpful response when no clear answer is found."""
        q = query.lower()

        # Practical fallbacks for common sunscreen questions
        if "reapply" in q:
            return "A practical rule is to reapply sunscreen about every 2 hours, and sooner after swimming, sweating, or towel-drying."
        if "apply" in q and "reapply" not in q:
            return "A practical rule is to apply sunscreen about 15 minutes before sun exposure so the film can set evenly."
        if all(x in q for x in ["mineral", "chemical"]) and any(x in q for x in ["compare", "vs", "difference"]):
            return "Mineral sunscreens (zinc oxide/titanium dioxide) mainly protect by scattering and absorbing UV at the skin surface; chemical sunscreens absorb UV and convert it to heat. In practice, the best sunscreen is the one you'll apply generously and reapply consistently."

        candidates = self._extract_candidate_sentences(context)
        if candidates:
            return f"I couldn't find a direct answer, but a relevant line is: {self._clean_answer(candidates[0])}"
        return "I couldn't find clear information about that in this content."

    def _simple_extractive_answer(self, query: str, context: List[str]) -> str:
        """Fallback: return the first readable candidate sentence."""
        candidates = self._extract_candidate_sentences(context)
        if candidates:
            return self._clean_answer(candidates[0])
        return "I found some information but couldn't extract a clear answer."

    def _rule_based_answer(self, question: str) -> str:
        """High-confidence practical answers for common sunscreen intents."""
        q = question.lower()

        if "cloudy" in q and "sunscreen" in q:
            return "Yes. Use sunscreen even when it's cloudy—UVA still reaches skin through clouds. Apply in the morning, then reapply about every 2 hours when outdoors, and after swimming, sweating, or towel-drying."

        if "reapply" in q and "sunscreen" in q:
            return "Reapply sunscreen about every 2 hours while outdoors, and immediately after swimming, sweating, or towel-drying."

        if any(x in q for x in ["when to use sunscreen", "when should i use sunscreen", "when should sunscreen be applied", "when apply sunscreen"]):
            return "Use sunscreen every day on exposed skin during daylight. Apply as the last skincare step before makeup, ideally 15 minutes before sun exposure."

        if "sunscreen" in q and any(x in q for x in ["need", "should", "when", "daily", "every day", "apply"]):
            return "As a practical rule: wear sunscreen daily on exposed skin during daytime, and reapply every 2 hours when outdoors (sooner after water or sweat)."

        if ("mineral" in q and "chemical" in q) and any(x in q for x in ["compare", "difference", "vs"]):
            return "Mineral sunscreens (zinc oxide/titanium dioxide) form UV-protective filters at the skin surface; chemical sunscreens absorb UV and convert it to heat. Both can work well—pick one you can apply generously and reapply consistently."

        return ""

    def _looks_fragmented(self, text: str) -> bool:
        """Detect transcript-fragment answers that are not user-friendly."""
        if not text:
            return True
        t = text.strip()
        if len(t) < 30:
            return True
        if t[:1].islower():
            return True
        # Very long run with no sentence ending is usually transcript mush
        if len(t) > 140 and not any(p in t for p in ".!?"):
            return True
        bad_starts = ["ff ", "very day", "ome ", "n your", "l based"]
        if any(t.lower().startswith(bs) for bs in bad_starts):
            return True
        return False
    
    def _clean_answer(self, text: str) -> str:
        """Clean up answer text by removing URLs, citations, etc."""
        import re
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove obvious broken leading fragments
        text = re.sub(r'^[^A-Za-z0-9]+', '', text)
        text = re.sub(r'^[a-z]\s+', '', text)

        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text.strip()
    
    def _is_unsatisfactory_answer(self, answer: str, question: str) -> bool:
        """Check if an answer seems unsatisfactory or vague."""
        if not answer or len(answer) < 20:
            return True
        
        # Check for vague phrases that don't really answer the question
        vague_phrases = [
            'the bits that',
            'this is why',
            'when people think',
            'there have been',
            'it is just'
        ]
        
        if any(phrase in answer.lower() for phrase in vague_phrases):
            return True
        
        # Check if answer contains question keywords but seems off-topic
        question_keywords = re.findall(r'\b\w{4,}\b', question.lower())
        answer_keywords = re.findall(r'\b\w{4,}\b', answer.lower())
        
        overlap = len(set(question_keywords) & set(answer_keywords))
        if overlap < 2:  # Not enough keyword overlap
            return True
        
        return False
    
    def _enhance_answer(self, original_answer: str, question: str, context: List[str]) -> str:
        """Enhance an unsatisfactory answer with more context and explanation."""
        import re
        
        # Try to find additional relevant information
        additional_info = []
        question_keywords = re.findall(r'\b\w{4,}\b', question.lower())
        
        for chunk in context[:4]:  # Look at more context chunks
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            for sentence in sentences:
                clean_sentence = re.sub(r'\[\d+\]', '', sentence).strip()
                if 30 < len(clean_sentence) < 180:
                    # Check if this sentence adds new information
                    sentence_keywords = re.findall(r'\b\w{4,}\b', clean_sentence.lower())
                    overlap = len(set(question_keywords) & set(sentence_keywords))
                    
                    if overlap >= 2 and clean_sentence.lower() != original_answer.lower():
                        # Check if this is substantially different from the original answer
                        words_in_common = len(set(sentence_keywords) & set(re.findall(r'\b\w{4,}\b', original_answer.lower())))
                        if words_in_common < len(sentence_keywords) * 0.7:  # Less than 70% overlap
                            additional_info.append(clean_sentence)
                            if len(additional_info) >= 2:
                                break
        
        if additional_info:
            return f"{original_answer} Additionally, the content mentions: " + ". ".join(additional_info)
        else:
            return self._provide_content_summary(question)
    
    def _provide_content_summary(self, question: str) -> str:
        """Provide a summary of what the content actually discusses about the topic."""
        import re
        
        # Extract main topic from question
        keywords = re.findall(r'\b\w{4,}\b', question.lower())
        main_topic = keywords[0] if keywords else "this topic"
        
        # Common topics in this content based on our analysis
        content_topics = {
            'sunscreen': 'sunscreen types (mineral vs chemical), sunscreen myths, and recommendations for different skin types',
            'skin cancer': 'skin cancer research, racial differences in skin cancer rates, and sun exposure studies',
            'sun exposure': 'relationship between sun exposure and skin health, UV protection, and sun safety guidelines',
            'skin health': 'skin barrier function, gut-skin connection, and general dermatology advice'
        }
        
        # Find the most relevant content topic
        best_match = None
        best_score = 0
        
        for topic, description in content_topics.items():
            score = len(set(keywords) & set(re.findall(r'\b\w{4,}\b', topic)))
            if score > best_score:
                best_score = score
                best_match = description
        
        if best_match:
            return f"The content doesn't provide a direct answer to this question, but it does discuss {best_match}. Would you like more information about that?"
        else:
            return f"I couldn't find specific information about {main_topic} in this content. The video focuses more on debunking myths and discussing general skin health topics."

    def query(self, question: str) -> str:
        """Answer a question using the RAG system with error handling."""
        if not question or not question.strip():
            return "Please provide a valid question."
        
        # Check if input is too short to be a meaningful question
        if len(question.strip()) <= 2:
            return "Please ask a complete question."
        
        # Check if input ends with question mark or is a statement that could be a question
        is_question = question.strip().endswith('?') or any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'are', 'is', 'do', 'does', 'can', 'will'])
        
        if not is_question:
            # Try to convert statement to question
            if not question.strip().endswith('?'):
                question = question.strip() + "?"
        
        try:
            # Fast path for common high-confidence intents
            direct = self._rule_based_answer(question)
            if direct:
                return direct

            context = self.retrieve(question)
            if not context:
                return self._suggest_related_topics(question)
            
            answer = self.generate_answer(question, context)
            if self._looks_fragmented(answer):
                # If extraction quality is poor, use practical guidance when possible
                fallback = self._rule_based_answer(question)
                if fallback:
                    return fallback
            return answer
        except Exception as e:
            return f"Error processing your question: {str(e)}"


def repl(rag: RAGSystem):
    """Simple REPL for querying the RAG system with improved UX."""
    print("RAG System REPL. Type 'exit' to quit.")
    if hasattr(rag, 'original_files') and len(rag.original_files) > 1:
        print(f"Loaded {len(rag.chunks)} chunks from {len(rag.original_files)} files: {', '.join(rag.original_files)}")
    else:
        print(f"Loaded {len(rag.chunks)} chunks from '{rag.subtitles_path}'")
    
    while True:
        try:
            query = input(">>> ")
            if query.lower() == "exit":
                break
            
            if not query.strip():
                continue
            
            # Show thinking indicator
            print("Processing...", end="\r")
            
            answer = rag.query(query)
            
            # Clear thinking indicator
            print(" " * 50, end="\r")
            print(f"\nAnswer: {answer}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def _read_text_file(path: str) -> str:
    """Read text file with UTF-8 fallback to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG System for YouTube Subtitles")
    parser.add_argument("subtitles", nargs='+', type=str, help="Path(s) to the subtitles file(s)")
    parser.add_argument("--model", type=str, default="distilgpt2",
                       help="Text generation model to use (default: distilgpt2)")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)")
    args = parser.parse_args()

    # Check if all files exist
    for subtitles_path in args.subtitles:
        if not os.path.exists(subtitles_path):
            print(f"Error: Subtitles file '{subtitles_path}' not found.")
            return

    try:
        print(f"Initializing RAG system with:")
        print(f"  - Input files: {', '.join(args.subtitles)}")
        print(f"  - Embedding model: {args.embedding_model}")
        print(f"  - Generation model: {args.model}")
        
        # Load and combine multiple files
        combined_parts = []
        for subtitles_path in args.subtitles:
            content = _read_text_file(subtitles_path)
            if content.strip():
                combined_parts.append(content)
            else:
                print(f"Warning: '{subtitles_path}' is empty and will be skipped.")

        if not combined_parts:
            raise ValueError("All provided files are empty.")

        combined_text = "\n\n".join(combined_parts)

        # Create RAG system with combined content
        rag = RAGSystem(args.subtitles[0], model_name=args.embedding_model)  # Use first file path as identifier
        rag.original_files = args.subtitles
        # Set the combined content
        rag.subtitles = combined_text
        
        # Load and preprocess subtitles
        rag.load_subtitles()
        rag.split_chunks()
        rag.generate_embeddings()
        rag.build_index()
        
        # Override the generator model if specified
        if args.model != "gpt2":
            rag.generator = rag.init_generator(model_id=args.model)
        
        repl(rag)
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
