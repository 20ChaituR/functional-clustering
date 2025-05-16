import asyncio
import json
from anthropic import AsyncAnthropic
from aiolimiter import AsyncLimiter
from .model import Model

class ClaudeModel(Model):
    MODELS = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]
    MAX_REQUESTS_PER_MINUTE = 50

    def __init__(self, model: str, api_key: str, temperature: float = 1.0):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        self.client = AsyncAnthropic(api_key=api_key)
        self.limiter = AsyncLimiter(ClaudeModel.MAX_REQUESTS_PER_MINUTE, time_period=60)
    
    def generate_function_completions(self, prompt, n = 1):
        return asyncio.run(self._generate_function_completions_async(prompt, n))
    
    async def _generate_function_completions_async(self, prompt, n):
        system_prompt = (
"""You are an expert Python programmer and coding assistant. Your goal is to generate a response **strictly** in JSON format with exactly two top-level fields: \"explanation\" and \"code\". Both fields must be valid JSON strings that include all necessary escape characters (e.g., newlines as \\n). 

- The \"explanation\" field should provide a detailed, step-by-step reasoning of how you derived your solution. 
- The \"code\" field must contain **only** valid Python code that can be copied and run directly in a Python file without modifications or additional text. 
- Do not include any fields other than \"explanation\" and \"code\".
- Do not include any text before or after the JSON object. 
- Do not include markdown formatting (like triple backticks). 
- If newlines are needed within the strings, represent them with '\\n'.

Your output should look like:

{
  \"explanation\": \"...\",
  \"code\": \"...\"
}

And nothing else.

Follow these instructions carefully to ensure the output is in the correct format.""")

        messages = [{"role": "user", "content": prompt}]

        async def _get_completion(task_id):
            async with self.limiter:
                try:
                    response = await self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        messages=messages,
                        system=system_prompt,
                        temperature=self.temperature,
                    )
                    completion_text = response.content[0].text

                    response_data = json.loads(completion_text)
                    return response_data['code']
                except Exception:
                    return None
        
        tasks = [_get_completion(i) for i in range(n)]
        completions = await asyncio.gather(*tasks)
        return [completion for completion in completions if completion is not None]

    def generate_test_completions(self, prompt):
        raise NotImplementedError()