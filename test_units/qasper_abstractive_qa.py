import argparse

from pathlib import Path
from metrics.attributed_qa_eval import AttributedQAEval
from metrics.json_parser import JSONParser
from tasks.util import get_raw_predictions, parse_predictions


def attributed_eval(entries, eval_dir):
    res = {}
    json_parser = JSONParser(
        default = {"answer": None, "evidence": None}, task="Qasper"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = AttributedQAEval()
    lm_judge_file = Path(eval_dir) / f"lm_judge.json"
    lm_judge_raw_file = Path(eval_dir) / f"lm_judge_raw.json"
    lm_judge_mapping = Path(eval_dir) / f"lm_judge_mapping.json"

    res["results"] = {}
    res["results"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["results"]["all"] = evaluator.evaluate(
        predictions["all"], 
        lm_judge_file=lm_judge_file, 
        lm_judge_raw_file=lm_judge_raw_file, 
        lm_judge_mapping=lm_judge_mapping, 
        use_batch_api=True,
        eval_type="reference_free"
    )
    res["json_counts"] = json_counts
    
    return res["results"]["all"]["scores"].get("lm_judge", None), \
        res["results"]["all"]["scores"]["f1_evidence_all"]

def main():
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/qasper_abstractive_qa/raw_predictions.jsonl", 
        help="Path to the prediction directory."
    )
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/qasper_abstractive_qa", 
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
        default = {"answer": None, "evidence": None}, task="Qasper"
    )
    predictions, json_counts = parse_predictions(entries, json_parser)

    evaluator = AttributedQAEval()
    lm_judge_file = Path(args.eval_dir) / f"lm_judge.json"
    lm_judge_raw_file = Path(args.eval_dir) / f"lm_judge_raw.json"
    lm_judge_mapping = Path(args.eval_dir) / f"lm_judge_mapping.json"

    res["results"] = {}
    res["results"]["valid_json"] = evaluator.evaluate(predictions["parsed"])
    res["results"]["all"] = evaluator.evaluate(
        predictions["all"], 
        lm_judge_file=lm_judge_file, 
        lm_judge_raw_file=lm_judge_raw_file, 
        lm_judge_mapping=lm_judge_mapping, 
        use_batch_api=True,
        eval_type="reference_free"
    )
    res["json_counts"] = json_counts
    
    print("Results:\n", res)
    metrics_summary = {
        "lm_judge_answer": res["results"]["all"]["scores"].get("lm_judge", None),
        "f1_evidence": res["results"]["all"]["scores"]["f1_evidence_all"],
    }
    print("Primary metric:\n", metrics_summary)

if __name__ == "__main__":
    main()