from .dataset import Dataset
from .human_eval import HumanEval
from .live_code_bench import LiveCodeBench

def dataset_mapping(dataset: str) -> Dataset:
    if dataset == 'Human Eval':
        return HumanEval()
    elif dataset == 'Live Code Bench':
        return LiveCodeBench()
    else:
        raise Exception(f"Dataset {dataset} is not supported.")