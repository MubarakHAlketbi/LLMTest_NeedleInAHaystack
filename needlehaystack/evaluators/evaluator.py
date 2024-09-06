from abc import ABC, abstractmethod

class Evaluator(ABC):
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numerical score"""}

    @abstractmethod
    async def evaluate_response(self, response: str) -> int:
        pass

    @abstractmethod
    async def evaluate_response_with_retry(self, response: str, max_retries: int, base_delay: float) -> int:
        pass