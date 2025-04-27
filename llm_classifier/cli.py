import argparse
import sys
from llm_classifier import Annotator, Parser, Evaluator


def main():
    parser = argparse.ArgumentParser(prog='llm-classifier')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Annotate command
    ann = subparsers.add_parser('annotate', help='Run annotation')
    ann.add_argument('-i', '--input', required=True, help='Input CSV path')
    ann.add_argument('-o', '--output', default='results', help='Output directory')
    ann.add_argument('-m', '--models', nargs='+', required=True, help='List of model names')
    ann.add_argument('--openai-key', required=True, help='OpenAI API key')
    ann.add_argument('--deepseek-key', help='DeepSeek API key (optional)')
    ann.add_argument('--system-message', help='Path to system message file')
    ann.add_argument('--prompt-template', help='Path to prompt template file')
    ann.add_argument('--prompt-columns', nargs='+', help='CSV column names to inject into prompt')
    ann.add_argument('-d', '--debug', action='store_true', help='Print model name and a sample filled-in prompt for each model')

    # Evaluate command
    ev = subparsers.add_parser('evaluate', help='Compute evaluation metrics')
    ev.add_argument('-p', '--predictions', required=True, help='Predictions CSV path')
    ev.add_argument('-g', '--ground-truth', required=True, help='Ground truth CSV path')

    args = parser.parse_args()
    if args.command == 'annotate':
        annotator = Annotator(
            openai_api_key=args.openai_key,
            deepseek_api_key=args.deepseek_key,
            models=args.models,
            system_message_path=args.system_message,
            prompt_template_path=args.prompt_template,
            prompt_columns=args.prompt_columns,
            debug=args.debug,
        )
        annotator.run(args.input, args.output)
    else:
        parser_ = Parser()
        df_pred = parser_.parse_csv(args.predictions)
        evaluator = Evaluator()
        print(evaluator.evaluate(df_pred, args.ground_truth))
        sys.exit(0)
