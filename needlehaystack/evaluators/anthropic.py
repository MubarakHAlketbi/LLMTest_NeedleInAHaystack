import time
import random
import os
from typing import Dict, Any
from anthropic import Anthropic as AnthropicClient
from .evaluator import Evaluator

class AnthropicEvaluator(Evaluator):
    CRITERIA: Dict[str, str] = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Only respond with a numerical score"""
    }

    DEFAULT_MODEL_KWARGS: Dict[str, Any] = {"max_tokens": 300, "temperature": 0}

    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",  # Updated default model
        model_kwargs: Dict[str, Any] = None,
        true_answer: str = None,
        question_asked: str = None
    ):
        if not true_answer or not question_asked:
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs or self.DEFAULT_MODEL_KWARGS
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if not api_key:
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using Anthropic evaluator.")

        self.client = AnthropicClient(api_key=api_key)

    def evaluate_response(self, response: str) -> int:
        prompt = f"""You are an AI evaluator. Your task is to evaluate the accuracy of a given answer based on a reference answer and a question.

Question: {self.question_asked}
Reference Answer: {self.true_answer}
Given Answer: {response}

Evaluate the given answer based on the following criteria:
{self.CRITERIA['accuracy']}

Your evaluation:"""

        result = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.model_kwargs.get("max_tokens", 300),
            temperature=self.model_kwargs.get("temperature", 0),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        try:
            score = int(result.content[0].text.strip())
            return score
        except ValueError:
            print(f"Warning: Unable to parse score from Anthropic response: {result.content[0].text}")
            return 0  # or some default value

    def evaluate_response_with_retry(self, response: str, max_retries: int = 5, base_delay: float = 1) -> int:
        for attempt in range(max_retries):
            try:
                return self.evaluate_response(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt * base_delay) + (random.random() * base_delay)
                print(f"Anthropic API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        raise Exception("Max retries reached")
