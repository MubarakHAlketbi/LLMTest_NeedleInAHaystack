from dataclasses import dataclass, field
from typing import Optional, List

from dotenv import load_dotenv
from jsonargparse import CLI

from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from .evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator, AnthropicEvaluator
from .providers import Anthropic, BaseProvider, OpenAI, Cohere

load_dotenv()

@dataclass
class CommandArgs:
    provider: str = "openai"
    evaluator: str = "openai"
    model_name: str = "gpt-3.5-turbo-0125"
    evaluator_model_name: Optional[str] = "gpt-3.5-turbo-0125"
    needle: Optional[str] = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    haystack_dir: Optional[str] = "PaulGrahamEssays"
    retrieval_question: Optional[str] = "What is the best thing to do in San Francisco?"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 1000
    context_lengths_max: Optional[int] = 16000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[List[int]] = None
    document_depth_percent_min: Optional[int] = 0
    document_depth_percent_max: Optional[int] = 100
    document_depth_percent_intervals: Optional[int] = 35
    document_depth_percents: Optional[List[int]] = None
    document_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = True
    max_retries: int = 5
    base_delay: float = 1.0
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False
    needles: List[str] = field(default_factory=lambda: [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])

def get_model_to_test(args: CommandArgs) -> BaseProvider:
    match args.provider.lower():
        case "openai":
            return OpenAI(model_name=args.model_name)
        case "anthropic":
            return Anthropic(model_name=args.model_name)
        case "cohere":
            return Cohere(model_name=args.model_name)
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    match args.evaluator.lower():
        case "openai":
            return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                   question_asked=args.retrieval_question,
                                   true_answer=args.needle)
        case "langsmith":
            return LangSmithEvaluator()
        case "anthropic":
            return AnthropicEvaluator(model_name="claude-3-haiku-20240307",
                                      question_asked=args.retrieval_question,
                                      true_answer=args.needle)
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    
    args_dict = args.__dict__.copy()
    for key in ['model_to_test', 'evaluator', 'max_retries', 'base_delay']:
        args_dict.pop(key, None)

    if args.multi_needle:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(
            model_to_test=args.model_to_test,
            evaluator=args.evaluator,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            **args_dict
        )
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(
            model_to_test=args.model_to_test,
            evaluator=args.evaluator,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            **args_dict
        )
    tester.start_test()

if __name__ == "__main__":
    main()
