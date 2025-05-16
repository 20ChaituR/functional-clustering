from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    @abstractmethod
    def generate_function_completions(self, prompt: str, n: int = 1) -> List[str]:
        pass

    @abstractmethod
    def generate_test_completions(self, prompt: str) -> List[str]:
        pass
