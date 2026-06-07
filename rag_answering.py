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
                # For definition questions, ensure we have a complete answer
                query_lower = query.lower()
                if (query_lower.startswith("what is") or query_lower.startswith("what are")) and \
                   self._is_unsatisfactory_answer(best_answer, query):
                    # Try to enhance the definition answer
                    enhanced = self._enhance_definition_answer(best_answer, query, context)
                    if enhanced:
                        return enhanced
                return best_answer
        except Exception as e:
            print(f"Answer synthesis failed: {e}")

        # If synthesis fails, fall back to simple extractive approach
        return self._simple_extractive_answer(query, context)

    def _enhance_definition_answer(self, original_answer: str, question: str, context: List[str]) -> str:
        """Enhance definition answers with additional context."""
        # Extract the main term from the question
        question_lower = question.lower()
        if question_lower.startswith("what is "):
            term = question_lower[8:].strip()
        elif question_lower.startswith("what are "):
            term = question_lower[9:].strip()
        else:
            return ""
        
        # Look for additional explanatory context
        term_keywords = set(re.findall(r"\b[a-zA-Z]{3,}\b", term.lower()))
        if not term_keywords:
            return ""
        
        additional_context = []
        for chunk in context[:5]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for sentence in sentences:
                clean_sentence = self._clean_answer(sentence.strip())
                if 20 < len(clean_sentence) < 200:
                    sentence_lower = clean_sentence.lower()
                    
                    # Check if this sentence provides additional context about the term
                    sentence_keywords = set(re.findall(r"\b[a-zA-Z]{3,}\b", sentence_lower))
                    overlap = len(term_keywords & sentence_keywords)
                    
                    if overlap >= 1:
                        # Check for explanatory patterns
                        if any(pattern in sentence_lower for pattern in [
                            "because", "since", "due to", "reason", "explain", 
                            "purpose", "goal", "principle", "idea", "concept"
                        ]):
                            # Check that it's not just repeating the definition
                            words_in_common = len(
                                set(re.findall(r"\b[a-zA-Z]{4,}\b", sentence_lower)) &
                                set(re.findall(r"\b[a-zA-Z]{4,}\b", original_answer.lower()))
                            )
                            if words_in_common < len(re.findall(r"\b[a-zA-Z]{4,}\b", sentence_lower)) * 0.8:
                                additional_context.append(clean_sentence)
                                if len(additional_context) >= 2:
                                    break
        
        if additional_context:
            return f"{original_answer} This means that {'. '.join(additional_context)}"
        
        return ""

    def _analyze_question(self, query: str) -> str:
        """Analyze question type to determine best answering strategy."""
        query_lower = query.lower().strip()
        
        # Remove question marks and normalize
        query_clean = query_lower.rstrip('?')
        
        # Check for specific question patterns first
        if any(word in query_clean.split() for word in ["when", "should", "must", "need", "recommend", "best"]):
            return "advice"
        elif any(word in query_clean.split() for word in ["what", "describe", "define", "means", "meaning"]) or query_clean.startswith("define "):
            return "factual"
        elif any(word in query_clean.split() for word in ["why", "how", "does", "do", "can", "could", "would", "explain"]) or query_clean.startswith("explain "):
            return "explanatory"
        elif any(word in query_clean.split() for word in ["types", "kinds", "sorts", "categories", "list", "examples"]):
            return "categorical"
        elif any(word in query_clean.split() for word in ["compare", "vs", "versus", "difference", "similar"]):
            return "comparison"
        elif any(word in query_clean.split() for word in ["pro", "con", "advantage", "disadvantage", "benefit"]):
            return "pros_cons"
        else:
            # Default to general, but try to infer from structure
            if query_clean.startswith("how to") or query_clean.startswith("how can"):
                return "explanatory"
            elif query_clean.startswith("what is") or query_clean.startswith("what are") or query_clean.startswith("what's"):
                return "factual"
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
            
            # Special handling for markdown headings which often contain definitions
            heading_match = re.match(r"^(#{1,6}\s*)(.+)", chunk)
            if heading_match:
                # Put heading content first as it's often the main definition
                heading_content = heading_match.group(2).strip()
                if heading_content:
                    parts.insert(0, heading_content)

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
        query_lower = query.lower()

        # Special handling for "what is" questions
        if query_lower.startswith("what is") or query_lower.startswith("what are"):
            return self._handle_definition_question(query, candidates, keywords)

        scored = []
        for i, sent in enumerate(candidates):
            s_lower = sent.lower()
            score = 0.0

            # Keyword overlap (word-level, avoid substring false positives)
            overlap = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", s_lower))
            score += overlap * 2.0

            # Question type specific scoring
            if question_type == "advice" and any(
                x in s_lower for x in ["should", "recommend", "important", "best", "must", "need"]
            ):
                score += 1.0
            elif question_type == "factual" and any(
                x in s_lower for x in ["is", "are", "was", "were", "defines", "means"]
            ):
                score += 0.8
            elif question_type == "explanatory" and any(
                x in s_lower for x in ["because", "since", "due to", "reason", "explain"]
            ):
                score += 0.9
            elif question_type == "categorical" and any(
                x in s_lower for x in ["types", "kinds", "categories", "include", "such as"]
            ):
                score += 0.8

            # Length preference: medium-length sentences are often better
            if 40 <= len(sent) <= 180:
                score += 0.5
            elif 180 < len(sent) <= 250:
                score += 0.3  # Slightly prefer shorter over very long

            # Position in context: earlier chunks are often more relevant
            score += max(0.0, 0.4 - (i * 0.02))
            
            # Boost chunks that start with headings (often contain main definitions)
            if re.match(r"^#{1,6}\s", sent):
                score += 1.5
            
            # Strong boost for quote blocks (core definitions)
            if re.match(r"^>", sent):
                score += 2.5
            
            # Check for direct question answering patterns
            if any(pattern in s_lower for pattern in [
                f"{query_lower}",
                f"answer to {query_lower}",
                f"response to {query_lower}"
            ]):
                score += 1.5
            
            # Penalize "see also" and cross-reference chunks for definition questions
            if question_type == "factual" and any(phrase in s_lower for phrase in [
                "see also:", "see also ", "[see also]", "related:", "for more see",
                "*see also:", "* see also:", "*see also ", "* see also "
            ]):
                return 0.0  # Completely exclude cross-references from definition answers
            
            # Boost chunks that actually define the term
            if question_type == "factual":
                # Extract likely term from question
                terms = [k for k in keywords if len(k) > 3]  # Filter short keywords
                for term in terms:
                    # Look for definition patterns with the term
                    if re.search(rf"\b{re.escape(term)}\s+(is|are|means|refers to|defines|is defined as)\b", s_lower):
                        score += 2.0
                        break
            
            scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Dynamic threshold based on scores
        if scored:
            max_score = scored[0][0]
            # If top score is very high, we can be more selective
            if max_score > 5.0:
                threshold = max(2.0, max_score * 0.4)
            else:
                threshold = max(1.0, max_score * 0.6)
        else:
            threshold = 1.0
            
        top = [s for sc, s in scored[:15] if sc >= threshold]
        if not top:
            return self._no_clear_answer(query, context)

        # Pick 1-3 diverse sentences with better diversity checking
        answer_parts = []
        used = set()
        used_phrases = set()
        
        for s in top:
            s_lower = s.lower()
            
            # Check if this sentence is too similar to what we already have
            is_duplicate = False
            for existing in answer_parts:
                existing_lower = existing.lower()
                
                # Check for substantial phrase overlap (indicates repetition)
                common_words = set(s_lower.split()) & set(existing_lower.split())
                if len(common_words) > 4:
                    is_duplicate = True
                    break
                
                # Check for key phrase repetition (e.g., "Greenspun vs YAGNI")
                for phrase in [" vs ", " versus ", " compared to ", " and "]:
                    if phrase in s_lower and phrase in existing_lower:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    break
            
            if is_duplicate:
                continue
            
            # Check for phrase-level diversity
            sentence_phrases = set()
            words = s_lower.split()
            for j in range(len(words) - 2):
                phrase = " ".join(words[j:j+3])
                if phrase in used_phrases:
                    is_duplicate = True
                    break
                sentence_phrases.add(phrase)
            
            if is_duplicate:
                continue
            
            # Also use the original keyword-based signature for additional diversity
            sig = " ".join(sorted(set(s_lower.split()) & keywords))
            if sig in used and sig:
                continue
            
            answer_parts.append(s)
            if sig:
                used.add(sig)
            used_phrases.update(sentence_phrases)
            if len(answer_parts) >= 3:
                break

        return self._clean_answer(" ".join(answer_parts))

    def _handle_definition_question(self, query: str, candidates: List[str], keywords: set) -> str:
        """Special handling for 'what is' definition questions."""
        # Look for sentences that define the term
        definition_candidates = []
        
        main_keyword = next(iter(keywords)) if keywords else ""
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Skip cross-references and "see also" chunks
            if any(phrase in candidate_lower for phrase in [
                "see also:", "see also ", "[see also]", "related:", "for more see",
                "*see also:", "* see also:", "*see also ", "* see also ",
                "see [", "see #", "see section"
            ]):
                continue
            
            # Skip source/bibliographic chunks for definition answers
            if candidate_lower.startswith("**source.**") or candidate_lower.startswith("**source:**"):
                continue
            if "named " in candidate_lower and " by " in candidate_lower and len(candidate) < 100:
                continue
            
            # Check for definition patterns
            # Look for term followed by definition
            term_found = False
            if main_keyword and re.search(rf"\b{re.escape(main_keyword)}\b", candidate_lower):
                term_found = True
            
            # Check for strong definition patterns
            strong_definition = False
            if any(pattern in candidate_lower for pattern in [
                "is defined as ", "refers to ", "means that ",
                "is the principle that ", "states that ", "is the idea that "
            ]):
                strong_definition = True
            
            # Check for basic definition patterns
            basic_definition = False
            if any(pattern in candidate_lower for pattern in [
                "is ", "are ", "is a ", "are a ", "is the ", "are the "
            ]):
                basic_definition = True
            
            # Also accept quote blocks as definitions
            quote_block = re.match(r"^>", candidate)
            
            if (term_found and (strong_definition or basic_definition)) or quote_block:
                # Score the definition quality
                score = 0
                if strong_definition:
                    score += 3
                elif basic_definition:
                    score += 1
                elif quote_block:
                    score += 2  # Quote blocks are often definitions
                
                # Prefer longer, more complete definitions
                if 50 <= len(candidate) <= 200:
                    score += 2
                elif len(candidate) > 200:
                    score += 1
                
                # Check if it starts with the term (often better definitions)
                if candidate_lower.startswith(f"{main_keyword.lower()} ") or \
                   candidate_lower.startswith(f"the {main_keyword.lower()} ") or \
                   candidate_lower.startswith(f"{main_keyword.lower()} is ") or \
                   candidate_lower.startswith(f"{main_keyword.lower()} are "):
                    score += 2
                
                definition_candidates.append((score, candidate))
        
        if definition_candidates:
            # Sort by score and return the best definition
            definition_candidates.sort(key=lambda x: x[0], reverse=True)
            best_definition = definition_candidates[0][1]
            
            # Try to combine with additional context
            additional_info = []
            for score, candidate in definition_candidates[1:3]:  # Get 2nd and 3rd best
                # Check if it adds new information
                if len(set(candidate.lower().split()) & set(best_definition.lower().split())) < len(candidate.split()) * 0.7:
                    additional_info.append(candidate)
                    if len(additional_info) >= 2:
                        break
            
            if additional_info:
                return self._clean_answer(f"{best_definition} " + " ".join(additional_info))
            else:
                return self._clean_answer(best_definition)
        
        # If no clear definition, fall back to general synthesis
        return ""

    def _no_clear_answer(self, query: str, context: List[str]) -> str:
        """Provide a helpful response when no clear answer is found."""
        question_type = self._analyze_question(query)
        candidates = self._extract_candidate_sentences(context)
        if candidates:
            # Try to find the most relevant candidate
            keywords = self._query_keywords(query)
            best_candidate = None
            best_score = 0
            
            for candidate in candidates[:5]:
                score = sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", candidate.lower()))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate and best_score > 0:
                return (
                    f"I couldn't find a direct answer, but here's a relevant passage: "
                    f"{self._clean_answer(best_candidate)}"
                )
        
        # Use enhanced summary based on question type
        return self._provide_enhanced_summary(query, question_type)

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

    def _looks_like_definition(self, text: str) -> bool:
        """Check if text looks like a definition (quote, complete sentence, etc.)."""
        if not text or len(text) < 20:
            return False
            
        # Check if it starts with a quote marker
        if text.strip().startswith(">"):
            return True
            
        # Check if it's a complete sentence
        if text[0].isupper() and any(p in text for p in ".!?"):
            return True
            
        # Check for definition-like patterns
        lower_text = text.lower()
        if any(pattern in lower_text for pattern in [
            " is ", " are ", " means ", " refers to ", " defines ",
            " the principle that ", " states that ", " can be described as "
        ]):
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

        question_lower = question.lower().strip()
        question_type = self._analyze_question(question)  # Get question type for later use
        
        # Special handling for definition questions
        if question_lower.startswith("what is") or question_lower.startswith("what are"):
            # Check if the answer actually defines the term
            answer_lower = answer.lower()
            
            # Extract the term
            term = question_lower.split("what is ", 1)[1].split()[0] if "what is " in question_lower else \
                   question_lower.split("what are ", 1)[1].split()[0] if "what are " in question_lower else ""
            
            if term:
                # For definition questions, be more lenient about term matching
                # The answer might be a quote or definition that doesn't repeat the term
                term_normalized = term.replace("'", "").lower()
                answer_normalized = answer_lower.replace("'", "")
                
                # Check if term appears (with or without possessive)
                term_found = re.search(rf"\b{re.escape(term_normalized)}\b", answer_normalized)
                
                # For short answers that look like definitions, don't require term repetition
                if len(answer) < 100 and any(p in answer for p in ".!?") and \
                   (term_found or self._looks_like_definition(answer)):
                    # This looks like a valid definition, even if term isn't repeated
                    return False
                
                # If term is found, check for definition patterns
                if term_found:
                    has_definition = any(pattern in answer_lower for pattern in [
                        f"{term} is ", f"{term} are ", f"{term} means ", 
                        f"{term} refers to ", f"{term} is defined as "
                    ])
                    
                    if has_definition:
                        return False
                
                # Check if it's just a fragment or incomplete
                if len(answer) < 40 or not any(p in answer for p in ".!?"):
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

        # For explanatory questions, exclude the word "explain" from required keywords
        if question_type == "explanatory":
            question_keywords = {k for k in question_keywords if k != "explain"}

        if not question_keywords:  # After filtering, might be empty
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
        question_keywords = self._query_keywords(question)
        question_type = self._analyze_question(question)

        # Look at more context chunks and use better selection criteria
        for chunk in context[:6]:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            for sentence in sentences:
                clean_sentence = self._clean_answer(re.sub(r"\[\d+\]", "", sentence).strip())
                if 25 < len(clean_sentence) < 200:  # Wider range for additional context
                    # Check if this sentence adds new information
                    sentence_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", clean_sentence.lower()))
                    overlap = len(question_keywords & sentence_keywords)

                    if overlap >= 1 and clean_sentence.lower() != original_answer.lower():
                        # Skip cross-references in enhancement too
                        sentence_lower = clean_sentence.lower()
                        if any(phrase in sentence_lower for phrase in [
                            "see also:", "see also ", "*see also:", "* see also:",
                            "related:", "for more see"
                        ]):
                            continue
                        
                        # Check if this is substantially different from the original answer
                        original_keywords = set(re.findall(r"\b[a-zA-Z]{4,}\b", original_answer.lower()))
                        words_in_common = len(sentence_keywords & original_keywords)
                        
                        # More lenient overlap threshold for additional context
                        if words_in_common < len(sentence_keywords) * 0.8:
                            # Check for meaningful content (not just stopwords)
                            meaningful_words = [w for w in sentence_keywords if len(w) > 3 and w not in {"this", "that", "these", "those", "with", "from", "about"}]
                            if len(meaningful_words) >= 3:
                                additional_info.append(clean_sentence)
                                if len(additional_info) >= 3:
                                    break

        if additional_info:
            # Use different connectors based on question type
            if question_type in ["explanatory", "advice"]:
                connector = " To elaborate, "
            elif question_type in ["factual", "categorical"]:
                connector = " Additionally, "
            else:
                connector = " The content also mentions that "
                
            return f"{original_answer}{connector}" + ". ".join(additional_info)
        
        # If no additional info found, provide a more helpful summary
        return self._provide_enhanced_summary(question, question_type)

    def _provide_enhanced_summary(self, question: str, question_type: str) -> str:
        """Provide an enhanced summary fallback when a direct answer is unavailable."""
        keywords = self._query_keywords(question)
        main_topic = keywords.pop() if keywords else "this topic"
        
        # Provide more specific guidance based on question type
        if question_type == "factual":
            return (
                f"I couldn't find a clear definition or explanation of {main_topic} in this content. "
                "Try asking about specific aspects of {main_topic}, its characteristics, or examples of how it's used."
            )
        elif question_type == "explanatory":
            return (
                f"The content doesn't provide a detailed explanation for why or how {main_topic} works. "
                "You might find more information by asking about the principles behind {main_topic}, "
                "its purpose, or how it relates to other concepts mentioned in the material."
            )
        elif question_type == "advice":
            return (
                f"There isn't specific guidance about {main_topic} in this content. "
                "Consider asking about best practices related to {main_topic}, "
                "common approaches, or what to consider when dealing with {main_topic}."
            )
        elif question_type == "categorical":
            return (
                f"I couldn't find a list or categorization of {main_topic} in this content. "
                "Try asking about specific types of {main_topic}, examples, or how {main_topic} is classified."
            )
        elif question_type == "comparison":
            return (
                f"The content doesn't directly compare {main_topic} with other concepts. "
                "You might ask about how {main_topic} differs from related ideas, "
                "its advantages and disadvantages, or when to use {main_topic} versus alternatives."
            )
        else:
            return (
                f"I couldn't find relevant information about {main_topic} in this content. "
                "Try rephrasing your question with more specific terms from the material, "
                "or ask about particular aspects, examples, or applications of {main_topic}."
            )

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
