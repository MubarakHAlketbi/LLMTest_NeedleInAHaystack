import time
from typing import List, Optional
from tqdm import tqdm
import numpy as np

from .evaluators import Evaluator, LangSmithEvaluator
from .llm_needle_haystack_tester import LLMNeedleHaystackTester
from .providers import BaseProvider

class LLMMultiNeedleHaystackTester(LLMNeedleHaystackTester):
    def __init__(self, *args, 
                 needles: List[str] = [], 
                 model_to_test: BaseProvider = None,
                 evaluator: Evaluator = None, 
                 print_ongoing_status: bool = True,
                 eval_set: str = "multi-needle-eval-sf",
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 **kwargs):
        super().__init__(*args, model_to_test=model_to_test, max_retries=max_retries, base_delay=base_delay, **kwargs)

        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not evaluator:
            raise ValueError("An evaluator must be provided.")
        if not self.needle or not self.haystack_dir or not self.retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needles = needles
        self.evaluator = evaluator
        self.model_to_test = model_to_test
        self.eval_set = eval_set
        self.model_name = self.model_to_test.model_name
        self.print_ongoing_status = print_ongoing_status
        self.insertion_percentages = []

    def insert_needles(self, context: str, depth_percent: float, context_length: int) -> str:
        tokens_context = self.model_to_test.encode_text_to_tokens(context)
        context_length -= self.final_context_length_buffer

        total_needles_length = sum(len(self.model_to_test.encode_text_to_tokens(needle)) for needle in self.needles)

        if len(tokens_context) + total_needles_length > context_length:
            tokens_context = tokens_context[:context_length - total_needles_length]
        
        depth_percent_interval = (100 - depth_percent) / len(self.needles)
        
        self.insertion_percentages = []

        for needle in self.needles:
            tokens_needle = self.model_to_test.encode_text_to_tokens(needle)

            if depth_percent == 100:
                tokens_context = tokens_context + tokens_needle
            else:
                insertion_point = int(len(tokens_context) * (depth_percent / 100))
                tokens_new_context = tokens_context[:insertion_point]
                period_tokens = self.model_to_test.encode_text_to_tokens('.')
                
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]
                    
                tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

                insertion_percentage = (insertion_point / len(tokens_context)) * 100
                self.insertion_percentages.append(insertion_percentage)
                print(f"Inserted '{needle}' at {insertion_percentage:.2f}% of the context, total length now: {len(tokens_context)} tokens")
                
                depth_percent += depth_percent_interval  

        new_context = self.model_to_test.decode_tokens(tokens_context)
        return new_context

    def generate_context(self, context_length: int, depth_percent: float) -> str:
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needles(context, depth_percent, context_length)
        return context
    
    def evaluate_and_log(self, context_length: int, depth_percent: float, pbar: tqdm) -> None:
        if self.print_ongoing_status:
            print(f"\nStarting test: Context Length = {context_length}, Depth = {depth_percent}%")

        if self.save_results and self.result_exists(context_length, depth_percent):
            print(f"Result already exists for Context Length = {context_length}, Depth = {depth_percent}%. Skipping.")
            self.completed_tests += 1
            pbar.update(1)
            return

        context = self.generate_context(context_length, depth_percent)

        test_start_time = time.time()

        if isinstance(self.evaluator, LangSmithEvaluator):
            print("EVALUATOR: LANGSMITH")
            chain = self.model_to_test.get_langchain_runnable(context)
            self.evaluator.evaluate_chain(chain, context_length, depth_percent, self.model_to_test.model_name, 
                                          self.eval_set, len(self.needles), self.needles, self.insertion_percentages)
            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time
            score = None  # LangSmith doesn't provide an immediate score
            response = None
        else:
            print("EVALUATOR: Standard Model")
            prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)
            
            try:
                response = self.model_to_test.evaluate_model_with_retry(prompt, self.max_retries, self.base_delay)
            except Exception as e:
                print(f"Failed to get model response: {str(e)}")
                response = f"Error: Failed to get model response. Details: {str(e)}"

            try:
                score = self.evaluator.evaluate_response_with_retry(response, self.max_retries, self.base_delay)
            except Exception as e:
                print(f"Failed to evaluate response: {str(e)}")
                score = 0

            test_end_time = time.time()
            test_elapsed_time = test_end_time - test_start_time

        results = {
            'model': self.model_to_test.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needles': self.needles,
            'model_response': response,
            'score': score,
            'test_duration_seconds': test_elapsed_time,
            'test_timestamp_utc': self.get_current_utc_time()
        }

        self.testing_results.append(results)
        self.completed_tests += 1
        pbar.update(1)

        if self.print_ongoing_status:
            self.print_progress_update()
            self.print_test_summary(test_elapsed_time, context_length, depth_percent, score, response)

        self.save_results_and_contexts(results, context, context_length, depth_percent)

        if self.seconds_to_sleep_between_completions:
            time.sleep(self.seconds_to_sleep_between_completions)

    def print_start_test_summary(self) -> None:
        print("\nStarting Multi-Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needles: {self.needles}")
        print(f"- Max Retries: {self.max_retries}")
        print(f"- Base Delay: {self.base_delay}")
        print("\n")

    def start_test(self) -> None:
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()
