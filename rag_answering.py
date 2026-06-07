"""Answer synthesis and query-handling mixin for the RAG system."""

import re
from typing import List


class AnsweringMixin:
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

        if any(word in query_lower for word in ["when", "should", "must", "need"]):
            return "advice"
        elif any(word in query_lower for word in ["what", "describe", "explain"]):
            return "factual"
        elif any(word in query_lower for word in ["why", "how", "does", "is", "are"]):
            return "explanatory"
        elif any(word in query_lower for word in ["types", "kinds", "sorts", "categories"]):
            return "categorical"
        else:
            return "general"

    def _strip_markdown_noise(self, text: str) -> str:
        """Convert markdown-ish source text into plain, answer-friendly text."""
        if not text:
            return ""

        text = text.replace("\n", " ")
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)  # markdown links
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text)  # headings
        text = re.sub(r"^\s*[-*+]\s+", "", text)  # list bullets
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_candidate_sentences(self, context: List[str]) -> List[str]:
        """Extract readable candidate sentences/passages from retrieved chunks."""
        candidates = []
        for chunk in context:
            chunk = self._strip_markdown_noise(chunk)
            chunk = re.sub(r"\s+", " ", chunk).strip()
            if not chunk:
                continue

            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", chunk) if p.strip()]
            if not parts:
                parts = [chunk]

            for part in parts:
                part = self._strip_markdown_noise(part)
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
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "about",
            "from",
            "than",
            "then",
            "into",
            "your",
            "you",
            "it",
            "this",
            "that",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        return {w for w in words if w not in stopwords}

    def _synthesize_answer(self, query: str, context: List[str], question_type: str) -> str:
        """Synthesize a cleaner extractive answer from multiple chunks."""
        candidates = self._extract_candidate_sentences(context)
        if not candidates:
            return ""

        keywords = self._query_keywords(query)

        scored = []
        for i, sent in enumerate(candidates):
            s_lower = sent.lower()
            score = 0.0

            # Keyword overlap (word-level, avoid substring false positives)
            overlap = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", s_lower))
            score += overlap * 2.0

            if question_type == "advice" and any(
                x in s_lower for x in ["should", "recommend", "important", "best"]
            ):
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
        candidates = self._extract_candidate_sentences(context)
        if candidates:
            return (
                "I couldn't find a direct answer, but a relevant line is: "
                f"{self._clean_answer(candidates[0])}"
            )
        return "I couldn't find clear information about that in this content."

    def _simple_extractive_answer(self, query: str, context: List[str]) -> str:
        """Fallback: return the first readable candidate sentence."""
        candidates = self._extract_candidate_sentences(context)
        if candidates:
            return self._clean_answer(candidates[0])
        return "I found some information but couldn't extract a clear answer."

    def _rule_based_answer(self, question: str) -> str:
        """Optional rule-based override for known intents (disabled by default)."""
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
        """Clean up answer text by removing URLs, markdown, citations, etc."""
        text = self._strip_markdown_noise(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove obvious broken leading fragments
        text = re.sub(r"^[^A-Za-z0-9]+", "", text)
        text = re.sub(r"^[a-z]\s+", "", text)

        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text.strip(" *`_-\t\n")

    def _is_unsatisfactory_answer(self, answer: str, question: str) -> bool:
        """Check if an answer seems unsatisfactory or vague."""
        if not answer or len(answer) < 20:
            return True

        # Check for vague phrases that don't really answer the question
        vague_phrases = [
            "the bits that",
            "this is why",
            "when people think",
            "there have been",
            "it is just",
        ]

        if any(phrase in answer.lower() for phrase in vague_phrases):
            return True

        # Use content-bearing query words (exclude question words/stopwords)
        question_keywords = self._query_keywords(question)
        answer_keywords = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower()))

        if not question_keywords:
            return False

        overlap = len(question_keywords & answer_keywords)
        required_overlap = 1 if len(question_keywords) <= 2 else 2
        if overlap < required_overlap:
            return True

        return False

    def _enhance_answer(self, original_answer: str, question: str, context: List[str]) -> str:
        """Enhance an unsatisfactory answer with more context and explanation."""
        # Try to find additional relevant information
        additional_info = []
        question_keywords = re.findall(r"\b\w{4,}\b", question.lower())

        for chunk in context[:4]:  # Look at more context chunks
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for sentence in sentences:
                clean_sentence = self._clean_answer(re.sub(r"\[\d+\]", "", sentence).strip())
                if 30 < len(clean_sentence) < 180:
                    # Check if this sentence adds new information
                    sentence_keywords = re.findall(r"\b\w{4,}\b", clean_sentence.lower())
                    overlap = len(set(question_keywords) & set(sentence_keywords))

                    if overlap >= 2 and clean_sentence.lower() != original_answer.lower():
                        # Check if this is substantially different from the original answer
                        words_in_common = len(
                            set(sentence_keywords)
                            & set(re.findall(r"\b\w{4,}\b", original_answer.lower()))
                        )
                        if words_in_common < len(sentence_keywords) * 0.7:  # Less than 70% overlap
                            additional_info.append(clean_sentence)
                            if len(additional_info) >= 2:
                                break

        if additional_info:
            return (
                f"{original_answer} Additionally, the content mentions: "
                + ". ".join(additional_info)
            )
        return self._provide_content_summary(question)

    def _provide_content_summary(self, question: str) -> str:
        """Provide a generic summary fallback when a direct answer is unavailable."""
        keywords = re.findall(r"\b\w{4,}\b", question.lower())
        main_topic = keywords[0] if keywords else "this topic"
        return (
            f"I couldn't find a direct answer about {main_topic} in this content. "
            "Try rephrasing your question or asking for a specific definition, example, or comparison."
        )

    def _suggest_related_topics(self, question: str) -> str:
        """Fallback when retrieval finds no relevant chunks."""
        keywords = re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        hint = f" around '{keywords[0]}'" if keywords else ""
        return (
            "I couldn't find relevant passages for that question"
            f"{hint}. Try using exact terms from the transcript or ask a narrower question."
        )

    def query(self, question: str) -> str:
        """Answer a question using the RAG system with error handling."""
        if not question or not question.strip():
            return "Please provide a valid question."

        # Check if input is too short to be a meaningful question
        if len(question.strip()) <= 2:
            return "Please ask a complete question."

        # Check if input ends with question mark or contains question words as full tokens
        q_lower = question.lower().strip()
        has_question_word = re.search(
            r"\b(what|how|why|when|where|who|which|are|is|do|does|can|will|should|could)\b",
            q_lower,
        ) is not None
        is_question = q_lower.endswith("?") or has_question_word

        if not is_question and not q_lower.endswith("?"):
            # Try to convert statement to question
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
            if self._is_unsatisfactory_answer(answer, question):
                answer = self._enhance_answer(answer, question, context)

            if self._looks_fragmented(answer):
                # If extraction quality is poor, use practical guidance when possible
                fallback = self._rule_based_answer(question)
                if fallback:
                    return fallback
            return answer
        except Exception as e:
            return f"Error processing your question: {str(e)}"
