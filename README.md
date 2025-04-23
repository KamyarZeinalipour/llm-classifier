
# llm-classifier
A Python package for annotating text using LLMs (OpenAI & DeepSeek), extracting JSON outputs, evaluating performance (including Jaccard for multi-label), and plotting results—all in an object-oriented, reusable way.

## Features
- **Flexible LLM support**: OpenAI (`gpt-4o-mini`, etc.) and DeepSeek (`deepseek-chat`, `deepseek-reasoner`).
- **Resumeable annotations**: Automatically resumes interrupted runs.
- **Dynamic prompts**: Use `{column_name}` placeholders to inject CSV values into prompts.
- **JSON parsing**: Safely extracts and normalizes JSON from model outputs.
- **Evaluation**: Exact-match accuracy, sample-wise Jaccard, micro precision/recall/F1, with detailed reports.
- **CLI**: Simple commands: `annotate` and `evaluate`.

## Installation
```bash
# Install from PyPI
pip install llm-classifier


### Local development
```bash
# Clone the repo
git clone https://github.com/yourusername/llm-classifier.git
cd llm-classifier
# Install in editable mode
pip install -e .
```

## Quickstart
```python
from llm_classifier import Annotator, Parser, Evaluator

# 1) Annotate
annot = Annotator(
    openai_api_key="YOUR_OPENAI_KEY", 
    deepseek_api_key="YOUR_DEEPSEEK_KEY",  # optional
    models=["gpt-4o-mini","deepseek-chat"],
    system_message_path="configs/system_message.txt",
    prompt_template_path="configs/prompt_template.txt",
)
annot.run(input_csv="data/df_cleanv3.csv", output_dir="results")

# 2) Parse
parser = Parser()
df_parsed = parser.parse_csv("results/output_gpt-4o-mini.csv")

# 3) Evaluate
evaluator = Evaluator()
metrics_df = evaluator.evaluate(df_parsed, ground_truth_path="data/df_cleanv3.csv")
print(metrics_df)
```
