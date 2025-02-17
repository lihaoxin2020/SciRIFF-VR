import argparse

from pathlib import Path
from metrics.summary_comparison import SummaryComparison
from tasks.util import get_raw_predictions


def summ_comparison(entries, eval_dir):
    predictions = [entry["pred"] for entry in entries]
    references = [entry["ref"] for entry in entries]
    prompts = [entry["prompt"] for entry in entries]

    # wrap all the info for the llm judge into a dict
    instances = {}
    for i in range(len(predictions)):
        instances[i] = {}
        instances[i]["prediction"] = predictions[i]
        instances[i]["prompt"] = prompts[i]
        instances[i]["reference"] = references[i]

    res = {}

    # use model judge to compare `predictions` against `baselines` and against gold standard summaries
    # Number of samples to evaluate for each task; use more for the ref comparison since it uses gpt-3.5.
    n_sample_lookup = {"model_comparison": 50,
                        "reference_comparison": 100, 
                        "reference_free": 500
                        }

    res["lm_judge"] = {}
    eval_type = "reference_free"

    # set output directories
    lm_judge_raw_results_file = Path(eval_dir) / f"lm_judge_{eval_type}_raw.json"
    lm_judge_agg_results_file = Path(eval_dir) / f"lm_judge_{eval_type}.json"
    lm_judge_outputs = Path(eval_dir) / f"lm_judge_raw.json"
    lm_judge_mapping = Path(eval_dir) / "lm_judge_mapping.json"

    evaluator = SummaryComparison()
    llm_judge_results = evaluator.evaluate(
        instances,
        lm_judge_raw_results_file,
        lm_judge_agg_results_file,
        eval_type,
        n_sample_lookup[eval_type],
        lm_judge_outputs,
        lm_judge_mapping=lm_judge_mapping,
        use_batch_api=True
    )
    if not llm_judge_results:
        return
    res["lm_judge"][eval_type] = llm_judge_results

    return res["lm_judge"]["reference_free"]["avg_rating"]


def main():
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/mup_single_document_summarization/raw_predictions.jsonl", 
        help="Path to the prediction directory."
    )
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/mup_single_document_summarization", 
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
    raw_predictions = get_raw_predictions(fname=args.pred_dir, max_instances=args.max_instances)
    

    ###########################
    # Evaluation snippet
    ###########################

    predictions = [entry["pred"] for entry in raw_predictions]
    references = [entry["ref"] for entry in raw_predictions]
    prompts = [entry["prompt"] for entry in raw_predictions]

    # wrap all the info for the llm judge into a dict
    instances = {}
    for i in range(len(predictions)):
        instances[i] = {}
        instances[i]["prediction"] = predictions[i]
        instances[i]["prompt"] = prompts[i]
        instances[i]["reference"] = references[i]

    res = {}

    # use model judge to compare `predictions` against `baselines` and against gold standard summaries
    # Number of samples to evaluate for each task; use more for the ref comparison since it uses gpt-3.5.
    n_sample_lookup = {"model_comparison": 50,
                        "reference_comparison": 100, 
                        "reference_free": 500
                        }

    res["lm_judge"] = {}
    eval_type = "reference_free"

    # set output directories
    lm_judge_raw_results_file = Path(args.eval_dir) / f"lm_judge_{eval_type}_raw.json"
    lm_judge_agg_results_file = Path(args.eval_dir) / f"lm_judge_{eval_type}.json"
    lm_judge_outputs = Path(args.eval_dir) / f"lm_judge_raw.json"
    lm_judge_mapping = Path(args.eval_dir) / "lm_judge_mapping.json"

    evaluator = SummaryComparison()
    llm_judge_results = evaluator.evaluate(
        instances,
        lm_judge_raw_results_file,
        lm_judge_agg_results_file,
        eval_type,
        n_sample_lookup[eval_type],
        lm_judge_outputs,
        lm_judge_mapping=lm_judge_mapping,
        use_batch_api=True
    )
    if not llm_judge_results:
        return
    res["lm_judge"][eval_type] = llm_judge_results

    
    print("Results:\n", res)
    metrics_summary = {
        "lm_judge_reference": res["lm_judge"]["reference_free"]["avg_rating"]
    }
    print("Primary metric:\n", metrics_summary)

if __name__ == "__main__":
    main()