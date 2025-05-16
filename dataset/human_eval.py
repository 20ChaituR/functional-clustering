import json
import csv
import ast

from typing import Any, List, Tuple
from datasets import load_dataset
from func_timeout import func_timeout, FunctionTimedOut
from .dataset import Dataset


class HumanEval(Dataset):
    TIMEOUT = 1.0

    def __init__(self):
        super().__init__()
        self.ds = load_dataset("openai_humaneval")["test"]
        self.updates = {}
        self.skip_unupdated = False
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "prompt": self.get_function_prompt(idx),
            "test_prompt": self.get_test_prompt(idx),
        }
    
    def get_question_content(self, idx):
        return self.ds[idx]["prompt"]
    
    def get_function_prompt(self, idx):
        entry_point, function_docstring = self._get_docstring(idx)
        return (
            f"Complete the function '{entry_point}' in the following code snippet. "
            "Provide a detailed explanation of your reasoning in the 'explanation' field, "
            "and the complete code in the 'code' field. Think carefully about the problem, "
            "considering edge cases and the best approach to implement the function. If the "
            "code snippet contains any examples, think through those examples to verify "
            "whether your reasoning about the problem is correct. Use those examples to correct "
            "any misunderstandings you may have about the problem. Do not add new imports or define "
            "any new functions that were not included in the provided snippet. "
            "Output your response as a JSON object with fields 'explanation' and 'code'.\n"
            f"```{function_docstring}```"
        )

    def get_test_prompt(self, idx):
        entry_point, function_docstring = self._get_docstring(idx)
        return (
            f"Generate a comprehensive list of valid input test cases for the function '{entry_point}' in the following code. "
            f"The test cases should cover all possible valid scenarios, including edge cases and typical use cases. Provide only "
            f"the inputs to each test case. Each set of inputs should be a string that can be parsed with json.loads into a valid "
            f"dictionary with the function's parameter names as keys:\n"
            f"```{function_docstring}```"
        )
    
    def test_correctness(self, idx, completion):
        def run_test():
            local_namespace = {}
            
            entry_point, function_docstring = self._get_docstring(idx)
            test_code_str = self.ds[idx]["test"]

            preamble = f"{function_docstring}    pass"
            exec(preamble, local_namespace)
            
            test_code_str = f"{completion}\n{test_code_str}"
            exec(test_code_str, local_namespace)
            exec(f"check({entry_point})", local_namespace)
            return "AC"
        
        try:
            return func_timeout(HumanEval.TIMEOUT, run_test)
        except FunctionTimedOut:
            return "TL"
        except Exception:
            return "WA"

    def parse_inputs(self, inputs):
        if len(inputs) == 0:
            raise Exception("No inputs generated")
        
        return [json.loads(test) for test in inputs]

    def test_inputs(self, idx, completion, inputs):
        entry_point, function_docstring = self._get_docstring(idx)

        def run_test(func, test):
            return str(func(**test))

        try:
            local_namespace = {}
            preamble = f"{function_docstring}    pass"
            exec(preamble, local_namespace)
            exec(completion, {}, local_namespace)

            func = local_namespace.get(entry_point)
            if func is None:
                return None
            
            results = []
            for test in inputs:
                results.append(func_timeout(HumanEval.TIMEOUT, run_test, args=(func, test)))
            
            return results
        except (Exception, FunctionTimedOut):
            return None
        
    def extract_candidate_arg_tuples(self, check_fn: str) -> List[Tuple[Any, ...]]:
        """
        Parse the given Python code, find all assert statements of the form
            assert candidate(... ) == ...
        and return a list of tuples containing the arguments passed to candidate.
        """
        tree = ast.parse(check_fn)
        arg_tuples: List[Tuple[Any, ...]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                test = node.test
                if isinstance(test, ast.Compare):
                    left = test.left
                    if isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == 'candidate':
                        try:
                            values = [ast.literal_eval(arg) for arg in left.args]
                        except Exception:
                            continue
                        arg_tuples.append(tuple(values))

        return arg_tuples
    
    def test_absolute(self, idx, completion):
        entry_point, function_docstring = self._get_docstring(idx)

        def run_test(func, test):
            return str(func(*test))

        results = []
        try:
            local_namespace = {}
            preamble = f"{function_docstring}    pass"
            exec(preamble, local_namespace)
            exec(completion, {}, local_namespace)

            test_code_str = self.ds[idx]["test"]
            inputs = self.extract_candidate_arg_tuples(test_code_str)

            func = local_namespace.get(entry_point)
            if func is None:
                return None
            for test in inputs:
                results.append(func_timeout(HumanEval.TIMEOUT, run_test, args=(func, test)))
        except Exception as e:
            return "ER", str(e)
        except FunctionTimedOut:
            return "TL", None
        
        def run_check():
            local_namespace = {}
            
            entry_point, function_docstring = self._get_docstring(idx)
            test_code_str = self.ds[idx]["test"]
            preamble = f"{function_docstring}    pass"
            exec(preamble, local_namespace)
            
            test_code_str = f"{completion}\n{test_code_str}"
            exec(test_code_str, local_namespace)
            exec(f"check({entry_point})", local_namespace)

            return "AC", str(results)
        
        try:
            return func_timeout(HumanEval.TIMEOUT, run_check)
        except FunctionTimedOut:
            return "TL", None
        except Exception as e:
            return "WA", str(results)

    def update_prompts(self, updated_prompts):
        with open(updated_prompts, mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                idx = int(row["idx"])
                prompt = row["prompt"]
                self.updates[idx] = prompt
    
    def skip(self, idx):
        if idx == 76 or idx == 129: return True
        return self.skip_unupdated and idx in self.updates

    def _get_docstring(self, idx):
        entry_point = self.ds[idx]["entry_point"]
        if idx in self.updates:
            docstring = self.updates[idx]
        else:
            docstring = self.ds[idx]["prompt"]
        
        return entry_point, docstring