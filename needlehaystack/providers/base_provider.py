import os
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseProvider(ABC):
    def __init__(self, model_name: str, model_kwargs: dict, env_var_name: str):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = os.getenv(env_var_name)
        if not self.api_key:
            raise ValueError(f"{env_var_name} must be in env.")

    @abstractmethod
    def evaluate_model(self, prompt: Any) -> str:
        pass

    @abstractmethod
    def generate_prompt(self, context: str, retrieval_question: str) -> Any:
        pass

    @abstractmethod
    def encode_text_to_tokens(self, text: str) -> list[int]:
        pass

    @abstractmethod
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def get_langchain_runnable(self, context: str) -> Any:
        pass

    def evaluate_model_with_retry(self, prompt: Any, max_retries: int = 5, base_delay: float = 1) -> str:
        for attempt in range(max_retries):
            try:
                return self.evaluate_model(prompt)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt * base_delay) + (random.random() * base_delay)
                print(f"API error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        raise Exception("Max retries reached")
