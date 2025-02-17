import argparse, evaluate

from tasks.util import get_raw_predictions


def bleu(entries):
    bleu_scorer = evaluate.load("bleu")
    predictions = [entry["pred"] for entry in entries]
    references = [entry["ref"] for entry in entries]

    res = bleu_scorer.compute(predictions=predictions, references=references)
    return res["bleu"]


def main():
    parser = argparse.ArgumentParser(description="Run BioASQ evaluation.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="example/Qwen2.5-7B-Instruct/discomat_te/raw_predictions.jsonl", 
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
    raw_predictions = get_raw_predictions(fname=args.pred_dir, max_instances=args.max_instances)


    ###########################
    # Evaluation snippet
    ###########################
    bleu_scorer = evaluate.load("bleu")
    predictions = [entry["pred"] for entry in raw_predictions]
    references = [entry["ref"] for entry in raw_predictions]

    res = bleu_scorer.compute(predictions=predictions, references=references)
    res = {"bleu": res["bleu"]}
    
    print("Results:\n", res)

if __name__ == "__main__":
    main()