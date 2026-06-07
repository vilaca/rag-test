"""Answering module combining all answer-related functionality."""

from .question_analysis import QuestionAnalysisMixin
from .answer_generation import AnswerGenerationMixin
from .answer_validation import AnswerValidationMixin
from .answer_postprocessing import AnswerPostprocessingMixin
from .specialized_answers import SpecializedAnswersMixin
from .answer_utils import AnswerUtilsMixin


class AnsweringMixin(
    QuestionAnalysisMixin,
    AnswerGenerationMixin,
    AnswerValidationMixin,
    AnswerPostprocessingMixin,
    SpecializedAnswersMixin,
    AnswerUtilsMixin
):
    """Comprehensive answering mixin combining all answer-related functionality."""
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate an answer using advanced synthesis from retrieved context."""
        if not context:
            return "I couldn't find any relevant information about that in the content."
        
        # Analyze question type for synthesis
        question_type = self._analyze_question(query)
        
        # Generate answer using the existing synthesis method
        generated_answer = self._synthesize_answer(query, context, question_type)
        
        # Hallucination guard: check if answer can be supported by context
        if not self._can_answer_be_supported(generated_answer, query, context):
            return f"I couldn't find enough information in the indexed documents to confidently answer that question."
        
        # Apply post-processing
        return self._post_process_answer(generated_answer, query, context)
    
    def query(self, question: str) -> str:
        """Main query method that handles the full answer generation pipeline."""
        # Retrieve context (this would be handled by the retrieval mixin)
        context = self.retrieve(question, k=12)
        
        # Generate answer
        return self.generate_answer(question, context)
