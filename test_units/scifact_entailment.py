import argparse

from metrics.fact_checking_eval import FactCheckingEval
from metrics.json_parser import JSONParser
from tasks.util import get_raw_predictions, parse_predictions


def scifact_entailment(entries, default):
    res = {}
    json_parser = JSONParser(
        default=default, task="SciFact"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = FactCheckingEval()
    res["f1"] = {}
    res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["f1"]["all"] = evaluator.evaluate(predictions["all"])
    res["json_counts"] = json_counts
    
    print("Results:\n", res)
    metrics_summary = {
        "f1_label": res["f1"]["all"]["f1"]["label"]["f1"],
        "f1_evidence_sent": res["f1"]["all"]["f1"]["evidence_sent"]["f1"],
        "f1_evidence_token": res["f1"]["all"]["f1"]["evidence_tok"]["f1"],
    }
    print("Primary metric:\n", metrics_summary)


def main():
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/scifact_entailment/raw_predictions.jsonl", 
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
        default = {"verdict": "NEI", "evidence": []}, task="SciFact"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = FactCheckingEval()
    res["f1"] = {}
    res["f1"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["f1"]["all"] = evaluator.evaluate(predictions["all"])
    res["json_counts"] = json_counts
    
    print("Results:\n", res)
    metrics_summary = {
        "f1_label": res["f1"]["all"]["f1"]["label"]["f1"],
        "f1_evidence_sent": res["f1"]["all"]["f1"]["evidence_sent"]["f1"],
        "f1_evidence_token": res["f1"]["all"]["f1"]["evidence_tok"]["f1"],
    }
    print("Primary metric:\n", metrics_summary)

if __name__ == "__main__":
    main()