import os
from dotenv import load_dotenv
load_dotenv(override=True)

from .model import Model
from .chatgpt import ChatGPTModel
from .claude import ClaudeModel


def model_mapping(model: str, temperature: int) -> Model:
    if model in ChatGPTModel.MODELS:
        return ChatGPTModel(model, os.environ.get("OPENAI_API_KEY"), temperature)
    elif model in ClaudeModel.MODELS:
        return ClaudeModel(model, os.environ.get("CLAUDE_API_KEY"), temperature)
    else:
        raise Exception(f"Model {model} is not supported.")