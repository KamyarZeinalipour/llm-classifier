import argparse
import sys
from llm_classifier import Annotator, Parser, Evaluator


def main():
    ap = argparse.ArgumentParser(prog='llm-classifier')
    subs = ap.add_subparsers(dest='cmd', required=True)

    a = subs.add_parser('annotate')
    a.add_argument('-i','--input', required=True)
    a.add_argument('-o','--output', default='results')
    a.add_argument('-m','--models', nargs='+', required=True)
    a.add_argument('--openai-key', required=True)
    a.add_argument('--deepseek-key')
    a.add_argument('--system-message')
    a.add_argument('--prompt-template')

    e = subs.add_parser('evaluate')
    e.add_argument('-p','--predictions', required=True)
    e.add_argument('-g','--ground-truth', required=True)

    args = ap.parse_args()
    if args.cmd == 'annotate':
        ann = Annotator(
            openai_api_key=args.openai_key,
            deepseek_api_key=args.deepseek_key,
            models=args.models,
            system_message_path=args.system_message,
            prompt_template_path=args.prompt_template
        )
        ann.run(args.input, args.output)
    else:
        pr = Parser()
        dfp = pr.parse_csv(args.predictions)
        ev = Evaluator()
        print(ev.evaluate(dfp, args.ground_truth))
        sys.exit(0)
