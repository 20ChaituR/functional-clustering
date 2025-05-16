import subprocess
import tempfile
import os
import ast

from datasets import load_dataset
from func_timeout import func_timeout, FunctionTimedOut
from .dataset import Dataset

def run_subprocess(process, test_input):
        return process.communicate(input=test_input)

class LiveCodeBench(Dataset):
    TIMEOUT = 1.0
    
    def __init__(self):
        super().__init__()
        self.ds = load_dataset("livecodebench/code_generation")['test']

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return {
            "idx": idx,
            "prompt": self.get_function_prompt(idx),
            "test_prompt": self.get_test_prompt(idx),
        }

    def get_question_content(self, idx):
        return self.ds[idx]['question_content']

    def get_function_prompt(self, idx):
        question_content = self.ds[idx]['question_content']
        return (
            f"Write Python code to solve the following problem. Read from standard input and output to standard output. "
            "Provide a detailed explanation of your reasoning in the 'explanation' field, "
            "and the complete code in the 'code' field. Think carefully about the problem, "
            "considering edge cases and the best approach to implement the function. If the "
            "provided problem contains any examples, think through those examples to verify "
            "whether your reasoning about the problem is correct. Use those examples to correct "
            "any misunderstandings you may have about the problem. "
            "Output your response as a JSON object with fields 'explanation' and 'code'.\n"
            "### Problem Statement:\n"
            f"{question_content}"
        )

    def get_test_prompt(self, idx):
        question_content = self.ds[idx]['question_content']
        return (
            f"Generate a comprehensive list of valid input test cases for the given problem statement. "
            f"The test cases should cover all possible valid scenarios, including edge cases and typical use cases. Provide only "
            f"the inputs to each test case. Each input should be a string in the provided test format that will be passed into a program through standard input.\n"
            "### Problem Statement:\n"
            f"{question_content}"
        )
    
    def test_correctness(self, idx, completion):
        test_cases = ast.literal_eval(self.ds[idx]['private_test_cases'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(completion)
            temp_filename = f.name

        try:
            for test_case in test_cases:
                test_in = test_case["input"]
                test_out = test_case["output"]

                process = subprocess.Popen(
                    ['python', temp_filename],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                try:
                    stdout, stderr = func_timeout(LiveCodeBench.TIMEOUT, run_subprocess, args=(process, test_in))
                except FunctionTimedOut:
                    process.kill()
                    return "TL"
                if stderr:
                    return "ER"
                if stdout != test_out:
                    return "WA"
        
        finally:
            os.unlink(temp_filename)
        return "AC"

    def parse_inputs(self, inputs):
        return [ast.literal_eval(inp) for inp in inputs]
    
    def test_inputs(self, idx, completion, inputs):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(completion)
            temp_filename = f.name

        outputs = []
        try:
            for test_input in inputs:
                process = subprocess.Popen(
                    ['python', temp_filename],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                try:
                    # print("Testing", test_input)
                    stdout, stderr = func_timeout(LiveCodeBench.TIMEOUT, run_subprocess, args=(process, test_input))
                except FunctionTimedOut:
                    process.kill()
                    # print("Timeout")
                    return None
                if stderr:
                    # print("Error", stderr)
                    return None
                # print("Output", stdout)
                outputs.append(stdout)
        
        finally:
            os.unlink(temp_filename)

        return outputs
    
    def test_absolute(self, idx, completion):
        test_cases = ast.literal_eval(self.ds[idx]['private_test_cases'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(completion)
            temp_filename = f.name

        outputs = []
        all_ac = True
        try:
            for test_case in test_cases:
                test_in = test_case["input"]
                test_out = test_case["output"]

                process = subprocess.Popen(
                    ['python', temp_filename],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                try:
                    stdout, stderr = func_timeout(LiveCodeBench.TIMEOUT, run_subprocess, args=(process, test_in))
                except FunctionTimedOut:
                    process.kill()
                    return "TL", (test_in, test_out)
                if stderr:
                    return "ER", (stderr, test_in, test_out)
                if stdout != test_out:
                    all_ac = False
                outputs.append(stdout)
        
        finally:
            os.unlink(temp_filename)
        
        if all_ac:
            return "AC", tuple(outputs)
        else:
            return "WA", tuple(outputs)

    def update_prompts(self, updated_prompts):
        pass

    def skip(self, idx):
        return ast.literal_eval(self.ds[idx]['private_test_cases'])[0]['testtype'] != 'stdin'