# Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering

This repository is the official implementation of [Eliminating Hallucination-Induced Errors in LLM Code Generation with Functional Clustering]().

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Create a `.env` file in the root directory with your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key
CLAUDE_API_KEY=your-anthropic-api-key
```

## Usage

Everything is driven by **one config JSON** (see format below).  
The typical flow is:

1. **Generate functions**: `scripts/gen_functions.py`  
2. **Cluster by I/O behaviour**  
   * single file: `scripts/cluster_functions.py`  
   * many files: `scripts/cluster_multiple_functions.py`  
3. **Evaluate clusters**: `scripts/eval_clusters.py`  
   * prints expected accuracy, calibrated error, τ-thresholds  
   * `-i <idx>` shows a single task; `-f <dir>` dumps exemplar code  
4. **Visualize**  
   * scatter: `scripts/plot_clusters.py -s`  
   * cumulative: `scripts/plot_clusters.py -l`  


### Config format

```jsonc
{
  "dataset": "Human Eval",              // one of ["Human Eval", "Live Code Bench"]
  "models": [                           // generation models
    {"name": "gpt-4o", "temperature": 1.1}
  ],
  "n": 50,                              // total programs to sample
  "test_model": {"name": "gpt-4o", "temperature": 1.0}, // test-suite generator
  "test_retries": 5,                    // retries for flaky test generation
  "function_file": "results/human-eval/functions.csv",
  "function_files": "results/human-eval/function",       // prefix for multi-files
  "cluster_file": "results/human-eval/clusters.csv",
  "reset": false                        // true = start fresh, false = resume
}
```

### Example Commands

HumanEval:
```bash
# 1 Generate
python scripts/gen_functions.py -c config/humaneval-gpt.json

# 2a Cluster one file
python scripts/cluster_functions.py -c config/humaneval-gpt.json

# 3 Evaluate
python scripts/eval_clusters.py -c config/humaneval-gpt.json

#   └── inspect task 163 + dump exemplar solutions
python scripts/eval_clusters.py -c config/humaneval-gpt.json \
       -i 163 -f results/human-eval/solution_explanations

# 4 Visualize
python scripts/plot_clusters.py -c config/humaneval-gpt.json -s \
       -o results/plots/humaneval-conf-scatter.png

python scripts/plot_clusters.py -c config/humaneval-gpt.json -l \
       -o results/plots/humaneval-cumulative.png
```

LiveCodeBench:

```bash
# 1 Generate
python scripts/gen_functions.py -c config/lcb-gpt-claude.json

# 2b Cluster multiple files (if you generated chunks separately)
python scripts/cluster_multiple_functions.py -c config/lcb-gpt-claude.json

# or: cluster a single file
python scripts/cluster_functions.py -c config/lcb-gpt-claude.json

# 3 Evaluate
python scripts/eval_clusters.py -c config/lcb-gpt-claude.json

#   └── inspect task 163
python scripts/eval_clusters.py -c config/lcb-gpt-claude.json \
       -i 163 -f results/human-eval/solution_explanations

# 4 Visualize
python scripts/plot_clusters.py -c config/lcb-gpt-claude.json -s \
       -o results/plots/lcb-conf-scatter.png

python scripts/plot_clusters.py -c config/lcb-gpt-claude.json -l \
       -o results/plots/lcb-cumulative.png
```

### Adding a new dataset

Create `dataset/my_dataset.py` that implements every abstract method in `Dataset` (see `dataset/dataset.py` for the interface). Use `LiveCodeBench` as a template and keep the public API identical so all existing scripts work unchanged.

Edit `dataset/__init__.py` so the factory can return your new class:

```python
from .dataset import Dataset
from .human_eval import HumanEval
from .live_code_bench import LiveCodeBench
from .my_dataset import MyDataset          # NEW

def dataset_mapping(dataset: str) -> Dataset:
    if dataset == 'Human Eval':
        return HumanEval()
    elif dataset == 'Live Code Bench':
        return LiveCodeBench()
    elif dataset == 'My Dataset':            # NEW
        return MyDataset()
    raise Exception(f"Dataset {dataset} is not supported.")
```

### Adding a new model

Create `model/my_model.py` that implements   `generate_function_completions()` and `generate_test_completions()` exactly as specified in `model/model.py`. Use `ChatGPTModel` (`model/chatgpt.py`) as a guide for structure, authentication, and temperature handling.

Inside your class, define  
```python
class MyModel(Model):
    MODELS = ["my-model-small", "my-model-large"]
    ...
```

Each string in `MODELS` is a value that can appear in a config JSON.

Edit `model/__init__.py` so the factory can instantiate it:
```python
from .model import Model
from .chatgpt import ChatGPTModel
from .claude   import ClaudeModel
from .my_model import MyModel            # NEW

def model_mapping(model: str, temperature: int) -> Model:
    if model in ChatGPTModel.MODELS:
        return ChatGPTModel(model, os.environ["OPENAI_API_KEY"], temperature)
    elif model in ClaudeModel.MODELS:
        return ClaudeModel(model, os.environ["CLAUDE_API_KEY"], temperature)
    elif model in MyModel.MODELS:        # NEW
        return MyModel(model, "MY_API_KEY", temperature)
    raise Exception(f"Model {model} is not supported.")
```

## Results

When we apply our verifier to [Live Code Bench](https://livecodebench.github.io/) and set each model’s confidence
threshold so that it matches its own expected accuracy, we get the calibrated
error rates below (lower is better).

| Model            | Expected Accuracy (%) | Error Rate (%) |
| ---------------- | -------------------- | -------------- |
| GPT-4.1-mini     | 67.76                | 11.90          |
| GPT-4.1          | 66.48                | 9.52           |
| GPT-4.1-nano     | 45.95                | 9.52           |
| GPT-4o           | 36.93                | **4.11**       |
| Claude-3-Haiku   | 21.71                | 12.33          |

## License

This repository (code + docs) is released under the  
**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license. You may share and adapt the material for any purpose, even commercially, as long as you: 

1. give appropriate credit,  
2. provide a link to the license, and  
3. indicate if changes were made.  

Full text: <https://creativecommons.org/licenses/by-sa/4.0/>