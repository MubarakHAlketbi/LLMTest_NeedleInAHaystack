import os
import asyncio
import time
import random
from typing import Dict, Any
from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI
from .evaluator import Evaluator

class OpenAIEvaluator(Evaluator):
    CRITERIA: Dict[str, str] = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Only respond with a numerical score"""
    }

    DEFAULT_MODEL_KWARGS: Dict[str, Any] = {"temperature": 0}

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        model_kwargs: Dict[str, Any] = None,
        true_answer: str = None,
        question_asked: str = None,
    ):
        if not true_answer or not question_asked:
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs or self.DEFAULT_MODEL_KWARGS
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_key = os.getenv('NIAH_EVALUATOR_API_KEY')
        if not api_key:
            raise ValueError("NIAH_EVALUATOR_API_KEY must be in env for using OpenAI evaluator.")

        self.api_key = api_key
        
        self.evaluator = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            **self.model_kwargs
        )

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            prediction=response,
            reference=self.true_answer,
            input=self.question_asked,
        )

        return int(eval_result['score'])

    def evaluate_response_with_retry(self, response: str, max_retries: int = 5, base_delay: float = 1) -> int:
        for attempt in range(max_retries):
            try:
                return self.evaluate_response(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt * base_delay) + (random.random() * base_delay)
                print(f"OpenAI API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        raise Exception("Max retries reached")
