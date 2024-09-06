import glob
import json
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import numpy as np

from .evaluators import Evaluator
from .providers import BaseProvider

class LLMNeedleHaystackTester:
    def __init__(self,
                 model_to_test: BaseProvider = None,
                 evaluator: Evaluator = None,
                 needle: Optional[str] = None,
                 haystack_dir: str = "PaulGrahamEssays",
                 retrieval_question: Optional[str] = None,
                 results_version: int = 1,
                 context_lengths_min: int = 1000,
                 context_lengths_max: int = 16000,
                 context_lengths_num_intervals: int = 35,
                 context_lengths: Optional[List[int]] = None,
                 document_depth_percent_min: int = 0,
                 document_depth_percent_max: int = 100,
                 document_depth_percent_intervals: int = 35,
                 document_depth_percents: Optional[List[int]] = None,
                 document_depth_percent_interval_type: str = "linear",
                 num_concurrent_requests: int = 1,
                 save_results: bool = True,
                 save_contexts: bool = True,
                 final_context_length_buffer: int = 200,
                 seconds_to_sleep_between_completions: Optional[float] = None,
                 print_ongoing_status: bool = True,
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 **kwargs):

        if not model_to_test:
            raise ValueError("A language model must be provided to test.")
        if not evaluator:
            raise ValueError("An evaluator must be provided.")
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.model_to_test = model_to_test
        self.evaluator = evaluator

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.context_lengths = self.generate_context_lengths(context_lengths, context_lengths_min, context_lengths_max, context_lengths_num_intervals)
        self.document_depth_percents = self.generate_depth_percents(document_depth_percents, document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals, document_depth_percent_interval_type)
        
        self.model_name = self.model_to_test.model_name
        
        self.total_tests = len(self.context_lengths) * len(self.document_depth_percents)
        self.completed_tests = 0
        self.start_time = None
        self.rate_limit_warnings = 0

    def generate_context_lengths(self, context_lengths, min_length, max_length, num_intervals):
        if context_lengths is None:
            if min_length is None or max_length is None or num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            return np.round(np.linspace(min_length, max_length, num=num_intervals, endpoint=True)).astype(int)
        return context_lengths

    def generate_depth_percents(self, depth_percents, min_depth, max_depth, num_intervals, interval_type):
        if depth_percents is None:
            if min_depth is None or max_depth is None or num_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            
            if interval_type == 'linear':
                return np.round(np.linspace(min_depth, max_depth, num=num_intervals, endpoint=True)).astype(int)
            elif interval_type == 'sigmoid':
                return [self.logistic(x) for x in np.linspace(min_depth, max_depth, num_intervals)]
            else:
                raise ValueError("document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
        return depth_percents

    def logistic(self, x: float, L: float = 100, x0: float = 50, k: float = 0.1) -> float:
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * self.sigmoid(x), 3)
    
    def sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def run_test(self):
        self.start_time = time.time()
        
        pbar = tqdm(total=self.total_tests, desc="Overall Progress")

        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent, pbar)

        pbar.close()

        self.print_final_summary()

    def evaluate_and_log(self, context_length: int, depth_percent: float, pbar: tqdm) -> None:
        if self.print_ongoing_status:
            print(f"\nStarting test: Context Length = {context_length}, Depth = {depth_percent}%")

        if self.save_results and self.result_exists(context_length, depth_percent):
            print(f"Result already exists for Context Length = {context_length}, Depth = {depth_percent}%. Skipping.")
            self.completed_tests += 1
            pbar.update(1)
            return

        try:
            context = self.generate_context(context_length, depth_percent)
        except Exception as e:
            print(f"Error generating context: {str(e)}")
            return

        prompt = self.model_to_test.generate_prompt(context, self.retrieval_question)

        test_start_time = time.time()

        try:
            response = self.model_to_test.evaluate_model_with_retry(prompt, self.max_retries, self.base_delay)
        except Exception as e:
            print(f"Failed to get model response after {self.max_retries} retries: {str(e)}")
            response = "Error: Failed to get model response"

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        try:
            score = self.evaluator.evaluate_response_with_retry(response, self.max_retries, self.base_delay)
        except Exception as e:
            print(f"Failed to evaluate response after {self.max_retries} retries: {str(e)}")
            score = 0

        results = {
            'model': self.model_name,
            'context_length': int(context_length),
            'depth_percent': float(depth_percent),
            'version': self.results_version,
            'needle': self.needle,
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

    def print_progress_update(self) -> None:
        elapsed_time = time.time() - self.start_time
        avg_time_per_test = elapsed_time / self.completed_tests if self.completed_tests > 0 else 0
        estimated_time_remaining = avg_time_per_test * (self.total_tests - self.completed_tests)

        progress_str = f"\rProgress: {self.completed_tests}/{self.total_tests} tests | "
        progress_str += f"Elapsed: {elapsed_time:.2f}s | "
        progress_str += f"Remaining: {estimated_time_remaining:.2f}s | "
        progress_str += f"Rate Limit Warnings: {self.rate_limit_warnings}"
        
        print(progress_str, end='', flush=True)

    def print_final_summary(self) -> None:
        total_time = time.time() - self.start_time
        print("\nTest Completed!")
        print(f"Total tests run: {self.total_tests}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per test: {total_time/self.total_tests:.2f} seconds")
        print(f"Total Rate Limit Warnings: {self.rate_limit_warnings}")

    def result_exists(self, context_length: int, depth_percent: float) -> bool:
        results_dir = 'results/'
        if not os.path.exists(results_dir):
            return False
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    if (result['context_length'] == context_length and
                        result['depth_percent'] == depth_percent and
                        result.get('version', 1) == self.results_version and
                        result['model'] == self.model_name):
                        return True
        return False

    def generate_context(self, context_length: int, depth_percent: float) -> str:
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)
        return context
    
    def insert_needle(self, context: str, depth_percent: float, context_length: int) -> str:
        tokens_needle = self.model_to_test.encode_text_to_tokens(self.needle)
        tokens_context = self.model_to_test.encode_text_to_tokens(context)

        context_length -= self.final_context_length_buffer

        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            
            period_tokens = self.model_to_test.encode_text_to_tokens('.')
            
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        new_context = self.model_to_test.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context: str) -> int:
        return len(self.model_to_test.encode_text_to_tokens(context))

    def read_context_files(self) -> str:
        context = ""
        max_context_length = max(self.context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, self.haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def encode_and_trim(self, context: str, context_length: int) -> str:
        tokens = self.model_to_test.encode_text_to_tokens(context)
        if len(tokens) > context_length:
            context = self.model_to_test.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self) -> List[Dict[str, Any]]:
        return self.testing_results
    
    def print_start_test_summary(self) -> None:
        print("\nStarting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f"- Needle: {self.needle.strip()}")
        print("\n")

    def start_test(self) -> None:
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()

    def print_test_summary(self, test_elapsed_time: float, context_length: int, depth_percent: float, score: int, response: str) -> None:
        print(f"-- Test Summary -- ")
        print(f"Duration: {test_elapsed_time:.1f} seconds")
        print(f"Context: {context_length} tokens")
        print(f"Depth: {depth_percent}%")
        print(f"Score: {score}")
        print(f"Response: {response}\n")

    def save_results_and_contexts(self, results: Dict[str, Any], context: str, context_length: int, depth_percent: float) -> None:
        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] = context_file_location
            if not os.path.exists('contexts'):
                os.makedirs('contexts')
            with open(f'contexts/{context_file_location}_context.txt', 'w') as f:
                f.write(context)

        if self.save_results:
            if not os.path.exists('results'):
                os.makedirs('results')
            with open(f'results/{context_file_location}_results.json', 'w') as f:
                json.dump(results, f)

    @staticmethod
    def get_current_utc_time() -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')