import time
import random
from typing import Union, List, Dict, Any
import uuid
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.smith import RunEvalConfig
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith.client import Client
from langsmith.evaluation import EvaluationResult, run_evaluator
from langsmith.schemas import Example, Run
from .evaluator import Evaluator

@run_evaluator
def score_relevance(run: Run, example: Union[Example, None] = None) -> EvaluationResult:
    student_answer = run.outputs["output"]
    reference = example.outputs["answer"]

    template = """You are an expert grader of student answers relative to a reference answer. \n 
            The reference answer is a single ingredient or a list of ingredients related to pizza \n 
            toppings. The grade is the number of correctly returned ingredient relative to the reference. \n 
            For example, if the reference has 5 ingredients and the student returns 3, then the grade is 3. \n
            Here is the student answer: \n --- --- --- \n {answer}
            Here is the reference answer: \n --- --- --- \n {reference}"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["answer", "reference"],
    )

    class Grade(BaseModel):
        """Grade output"""
        score: int = Field(description="Score from grader")
    
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    
    grade_tool_oai = convert_to_openai_tool(Grade)
    
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "Grade"}},
    )
    
    parser_tool = PydanticToolsParser(tools=[Grade])
    
    chain = prompt | llm_with_tool | parser_tool

    score = chain.invoke({"answer": student_answer, "reference": reference})

    return EvaluationResult(key="needles_retrieved", score=score[0].score)

class LangSmithEvaluator(Evaluator):
    CRITERIA: Dict[str, str] = {
        "accuracy": """
        Score 1: The answer is completely unrelated to the reference.
        Score 3: The answer has minor relevance but does not align with the reference.
        Score 5: The answer has moderate relevance but contains inaccuracies.
        Score 7: The answer aligns with the reference but has minor omissions.
        Score 10: The answer is completely accurate and aligns perfectly with the reference.
        Only respond with a numerical score"""
    }

    def __init__(self):
        self.client = Client()

    def evaluate_response(self, response: str) -> int:
        # This method is added for consistency with other evaluators,
        # but it's not used in the current implementation of LangSmith
        print("Warning: LangSmithEvaluator.evaluate_response() is called, but it's not implemented.")
        return 0  # Return a default score

    def evaluate_response_with_retry(self, response: str, max_retries: int = 5, base_delay: float = 1) -> int:
        for attempt in range(max_retries):
            try:
                return self.evaluate_response(response)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = (2 ** attempt * base_delay) + (random.random() * base_delay)
                print(f"LangSmith evaluation error: {str(e)}. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
        raise Exception("Max retries reached")

    def evaluate_chain(
        self,
        chain: Any,
        context_length: int,
        depth_percent: float,
        model_name: str,
        eval_set: str,
        num_needles: int,
        needles: List[str],
        insertion_percentages: List[float]
    ) -> None:
        evaluation_config = RunEvalConfig(
            custom_evaluators=[score_relevance],
        )

        run_id = uuid.uuid4().hex[:4]
        project_name = eval_set
        self.client.run_on_dataset(
            dataset_name=eval_set,
            llm_or_chain_factory=chain,
            project_metadata={
                "context_length": context_length, 
                "depth_percent": depth_percent, 
                "num_needles": num_needles,
                "needles": needles,
                "insertion_percentages": insertion_percentages,
                "model_name": model_name
            },
            evaluation=evaluation_config,
            project_name=f"{context_length}-{depth_percent}--{model_name}--{project_name}--{run_id}",
        )
