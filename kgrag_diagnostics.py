import argparse
import json
import os

from closedbook_eval import (
    load_cwq_gold,
    load_webqsp_gold,
    normalize_answer,
)


def load_gold(dataset, gold_file, cwq_use_labels=False, cwq_entity_list_file=None):
    if dataset == "WebQSP":
        return load_webqsp_gold(gold_file)
    if dataset == "CWQ":
        return load_cwq_gold(
            gold_file,
            use_labels=cwq_use_labels,
            entity_list_file=cwq_entity_list_file,
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def endpoint_set(triples):
    values = set()
    for triple in triples:
        if isinstance(triple, dict):
            head = triple.get("head")
            tail = triple.get("tail")
        elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
            head, _relation, tail = triple[:3]
        else:
            continue

        for value in (head, tail):
            norm = normalize_answer(value)
            if norm:
                values.add(norm)

    return values


def coverage_for_gold_sets(gold_sets, endpoints):
    best = {
        "gold_answers": [],
        "gold_coverage": 0.0,
        "gold_covered_any": False,
        "gold_covered_count": 0,
        "gold_total_count": 0,
    }

    for gold_answers in gold_sets:
        gold_norm = [normalize_answer(x) for x in gold_answers if normalize_answer(x)]
        if not gold_norm:
            continue

        covered = [answer for answer in gold_norm if answer in endpoints]
        coverage = len(set(covered)) / len(set(gold_norm))

        if coverage > best["gold_coverage"]:
            best = {
                "gold_answers": gold_answers,
                "gold_coverage": coverage,
                "gold_covered_any": bool(covered),
                "gold_covered_count": len(set(covered)),
                "gold_total_count": len(set(gold_norm)),
            }

    return best


def prediction_faithfulness(pred_answers, endpoints):
    pred_norm = [normalize_answer(x) for x in pred_answers if normalize_answer(x)]
    if not pred_norm:
        return {
            "pred_faithfulness": 0.0,
            "pred_covered_any": False,
            "pred_all_in_triples": False,
            "pred_covered_count": 0,
            "pred_total_count": 0,
        }

    pred_set = set(pred_norm)
    covered = pred_set & endpoints
    return {
        "pred_faithfulness": len(covered) / len(pred_set),
        "pred_covered_any": bool(covered),
        "pred_all_in_triples": len(covered) == len(pred_set),
        "pred_covered_count": len(covered),
        "pred_total_count": len(pred_set),
    }


def safe_avg(values):
    return sum(values) / len(values) if values else 0.0


def diagnose(pred_file, gold):
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    rows = []
    for pred in preds:
        qid = pred["qid"]
        if qid not in gold:
            continue

        triples = pred.get("triples_used", [])
        endpoints = endpoint_set(triples)
        gold_diag = coverage_for_gold_sets(gold[qid], endpoints)
        pred_diag = prediction_faithfulness(pred.get("pred_answers", []), endpoints)

        rows.append(
            {
                "qid": qid,
                "question": pred.get("question"),
                "num_triples_used": pred.get("num_triples_used", len(triples)),
                "pred_answers": pred.get("pred_answers", []),
                **gold_diag,
                **pred_diag,
            }
        )

    total = len(rows)
    metrics = {
        "total": total,
        "avg_gold_coverage": safe_avg([row["gold_coverage"] for row in rows]),
        "gold_covered_any_rate": safe_avg([1.0 if row["gold_covered_any"] else 0.0 for row in rows]),
        "avg_pred_faithfulness": safe_avg([row["pred_faithfulness"] for row in rows]),
        "pred_covered_any_rate": safe_avg([1.0 if row["pred_covered_any"] else 0.0 for row in rows]),
        "pred_all_in_triples_rate": safe_avg([1.0 if row["pred_all_in_triples"] else 0.0 for row in rows]),
        "empty_prediction_rate": safe_avg([1.0 if not row["pred_answers"] else 0.0 for row in rows]),
        "avg_num_triples_used": safe_avg([row["num_triples_used"] for row in rows]),
    }

    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["WebQSP", "CWQ"])
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cwq_use_labels", action="store_true")
    parser.add_argument(
        "--cwq_entity_list_file",
        default=None,
        help="Optional Freebase entity list file for mapping CWQ answer MIDs to names.",
    )
    args = parser.parse_args()

    gold = load_gold(
        args.dataset,
        args.gold_file,
        cwq_use_labels=args.cwq_use_labels,
        cwq_entity_list_file=args.cwq_entity_list_file,
    )
    metrics, rows = diagnose(args.pred_file, gold)

    output_dir = args.output_dir or os.path.dirname(args.pred_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base = os.path.basename(args.pred_file)
    metrics_file = os.path.join(output_dir, f"{base}_kgrag_diagnostics.json")
    details_file = os.path.join(output_dir, f"{base}_kgrag_diagnostics_details.json")

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
