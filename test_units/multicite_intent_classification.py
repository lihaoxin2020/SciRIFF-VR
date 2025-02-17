# # NOTE(davidw) A lot of these fail for the base tulu model; this is always because the
# # model `</s>`, which is pretty much the same as predicting no categories.

import argparse

from metrics.list_f1 import ListF1
from metrics.json_parser import JSONParser
from tasks.util import get_raw_predictions, parse_predictions


def multicite(entries):
    res = {}
    json_parser = JSONParser(
        default=[], task="MultiCite"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = ListF1()
    res["f1"] = {}
    res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["f1"]["all"] = evaluator.evaluate(predictions["all"])
    res["json_counts"] = json_counts
    
    return res['f1']['all']['f1']['f1']
    

def main():
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/multicite_intent_classification/raw_predictions.jsonl", 
        help="Path to the prediction directory."
    )
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        default=None, 
        help="Path to the evaluation directory."
    )
    parser.add_argument(
        "--max_instances", 
        type=int, 
        default=None, 
        help="Maximum number of instances to evaluate."
    )

    args = parser.parse_args()

    # Load example outputs
    entries = get_raw_predictions(fname=args.pred_dir, max_instances=args.max_instances)

    ###########################
    # Evaluation snippet
    ###########################
    res = {}
    json_parser = JSONParser(
        default=[], task="MultiCite"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = ListF1()
    res["f1"] = {}
    res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["f1"]["all"] = evaluator.evaluate(predictions["all"])
    res["json_counts"] = json_counts
    
    print("Results:\n", res)
    print("Primary metric ['f1']['all']['f1']['f1']:\n", res['f1']['all']['f1']['f1'])

if __name__ == "__main__":
    main()