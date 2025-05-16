from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Any


# use system prompt to differentiate between different llms
# use user prompt for specific dataset

class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        pass

    @abstractmethod
    def get_function_prompt(self, idx: int) -> str:
        pass

    @abstractmethod
    def get_test_prompt(self, idx: int) -> str:
        pass

    @abstractmethod
    def test_correctness(self, idx: int, completion: str) -> str:
        pass

    @abstractmethod
    def parse_inputs(self, inputs: List[str]) -> List:
        pass
    
    @abstractmethod
    def test_inputs(self, idx: int, completion: str, inputs: List) -> Optional[List[str]]:
        pass

    @abstractmethod
    def test_absolute(self, idx: int, completion: str) -> Tuple[str, Any]:
        pass

    @abstractmethod
    def update_prompts(self, updated_prompts: str):
        pass

    def skip(self, idx: int) -> bool:
        return False
