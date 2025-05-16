import json
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from .model import Model


class ChatGPTModel(Model):
    MODELS = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]
    STRUCTURED_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"]

    def __init__(self, model: str, api_key: str, temperature: float = 1.0):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        self.client = OpenAI(api_key=api_key)
        self.uses_structured_outputs = model in ChatGPTModel.STRUCTURED_MODELS
    
    def custom_prompt(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return messages + [{"role": "assistant", "content": completion.choices[0].message.content}]

    def generate_function_completions(self, prompt, n=1):
        system_message = (
            "You are an expert Python programmer and coding assistant. Your task is to solve the "
            "given problem in Python, providing both a detailed "
            "explanation of your reasoning and the code. Think through the problem step by step, "
            "considering any edge cases and ensuring the code meets the requirements. If the problem "
            "contains any examples, simulate running those examples against your code to verify whether your reasoning about "
            "the problem is correct. Make sure to not have any misunderstandings about the problem."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        class FunctionCompletion(BaseModel):
            explanation: str
            code: str

        if self.uses_structured_outputs:
            response_format = FunctionCompletion
        else:
            response_format = {"type": "json_object"}

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                n=n,
                temperature=self.temperature
            )

            completions = []
            for choice in completion.choices:
                if self.uses_structured_outputs:
                    response_data = choice.message.parsed
                    code = response_data.code
                    completions.append(code)
                else:
                    response_data = choice.message.content

                    try:
                        response_data = json.loads(response_data)
                        code = response_data.get("code", "")
                        completions.append(code)
                    except json.JSONDecodeError as e:
                        continue

        except Exception as e:
            print(e)
            completions = []
        finally:
            return completions
    
    def generate_test_completions(self, prompt):
        class TestCasesResponse(BaseModel):
            test_cases: List[str] = Field(
                ...,
                description=(
                    "A list of strings, where each string is a JSON object representing input arguments for the function."
                ),
            )
        
        system_prompt = (
            "You are a coding assistant that generates comprehensive valid input test cases for a problem given its "
            "signature and docstring. You will only be generating the input for these problems, not the output. Your "
            "goal is to create test cases that thoroughly cover all possible valid scenarios, including edge cases and "
            "typical use cases."
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=TestCasesResponse,
        )

        response_data = response.choices[0].message.parsed
        return response_data.test_cases
