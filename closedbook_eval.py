import argparse
import json
import os
import re
import string

from tqdm import tqdm


def normalize_answer(text):
    text = str(text).lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"^(\d{4})-01-01$", r"\1", text)
    text = re.sub(r"^(\d{4})-\d{2}-\d{2} 05:12:00$", r"\1", text)
    text = text.strip(string.punctuation + " ")
    text = re.sub(r"\s+", " ", text)

    articles = {"a", "an", "the"}
    tokens = [tok for tok in text.split() if tok not in articles]
    return " ".join(tokens)


def valid_webqsp_parse(parse):
    comment = parse.get("AnnotatorComment", {})
    return (
        comment.get("QuestionQuality") == "Good"
        and comment.get("ParseQuality") == "Complete"
    )


def load_webqsp_gold(gold_file):
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = {}

    for item in data["Questions"]:
        qid = item["QuestionId"]
        parse_gold_sets = []

        for parse in item.get("Parses", []):
            if not valid_webqsp_parse(parse):
                continue

            answers = []
            for ans in parse.get("Answers", []):
                value = ans.get("EntityName") or ans.get("AnswerArgument")
                if value:
                    answers.append(value)

            parse_gold_sets.append(answers)

        if parse_gold_sets:
            gold[qid] = parse_gold_sets

    return gold


def load_cwq_gold(gold_file, use_labels):
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = {}

    get_label = None
    if use_labels:
        from executor.sparql_executor import get_label_with_odbc

        get_label = get_label_with_odbc

    for item in tqdm(data, desc="Loading CWQ gold"):
        answers = []
        for ans in item.get("answer", []):
            value = ans
            if get_label is not None and isinstance(ans, str) and ans.startswith(("m.", "g.")):
                try:
                    label = get_label(ans)
                    if label:
                        value = label
                except Exception:
                    value = ans
            answers.append(value)

        gold[item["ID"]] = [answers]

    return gold


def score_one(pred_answers, gold_answers):
    pred_norm = [normalize_answer(x) for x in pred_answers if normalize_answer(x)]
    gold_norm = [normalize_answer(x) for x in gold_answers if normalize_answer(x)]

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0, 1, 1
    if not gold_set:
        return 0.0, 1.0, 0.0, 0, 0
    if not pred_set:
        return 1.0, 0.0, 0.0, 0, 0

    overlap = pred_set & gold_set
    precision = len(overlap) / len(pred_set)
    recall = len(overlap) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    hits1 = 1 if pred_norm and pred_norm[0] in gold_set else 0
    em = 1 if pred_set == gold_set else 0
    return precision, recall, f1, hits1, em


def evaluate(pred_file, gold):
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    rows = []
    p_sum = r_sum = f_sum = hits_sum = em_sum = 0.0
    total = 0

    for pred in preds:
        qid = pred["qid"]
        if qid not in gold:
            continue

        pred_answers = pred.get("pred_answers", [])
        best = None

        for gold_answers in gold[qid]:
            score = score_one(pred_answers, gold_answers)
            if best is None or score[2] > best[2]:
                best = score + (gold_answers,)

        precision, recall, f1, hits1, em, best_gold_answers = best

        rows.append(
            {
                "qid": qid,
                "question": pred.get("question"),
                "pred_answers": pred_answers,
                "gold_answers": best_gold_answers,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hits1": hits1,
                "em": em,
            }
        )

        p_sum += precision
        r_sum += recall
        f_sum += f1
        hits_sum += hits1
        em_sum += em
        total += 1

    metrics = {
        "total": total,
        "avg_precision": p_sum / total if total else 0.0,
        "avg_recall": r_sum / total if total else 0.0,
        "avg_f1": f_sum / total if total else 0.0,
        "hits1": hits_sum / total if total else 0.0,
        "em": em_sum / total if total else 0.0,
    }

    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["WebQSP", "CWQ"])
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cwq_use_labels", action="store_true")
    args = parser.parse_args()

    if args.dataset == "WebQSP":
        gold = load_webqsp_gold(args.gold_file)
    else:
        gold = load_cwq_gold(args.gold_file, use_labels=args.cwq_use_labels)

    metrics, rows = evaluate(args.pred_file, gold)

    output_dir = args.output_dir or os.path.dirname(args.pred_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(args.pred_file)
    metrics_file = os.path.join(output_dir, f"{base}_metrics.json")
    details_file = os.path.join(output_dir, f"{base}_details.json")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
