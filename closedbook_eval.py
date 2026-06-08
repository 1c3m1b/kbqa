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


def collect_cwq_answer_ids(data):
    answer_ids = []
    seen = set()
    for item in data:
        for ans in item.get("answer", []):
            if (
                isinstance(ans, str)
                and ans.startswith(("m.", "g."))
                and ans not in seen
            ):
                seen.add(ans)
                answer_ids.append(ans)
    return answer_ids


def load_entity_names_from_file(entity_list_file, target_ids):
    names = {}
    if not entity_list_file:
        return names
    if not os.path.exists(entity_list_file):
        raise FileNotFoundError(f"Entity list file not found: {entity_list_file}")

    with open(entity_list_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading entity names"):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            mid, name = cols[0], cols[1]
            if mid in target_ids and name:
                names[mid] = name
                if len(names) == len(target_ids):
                    break

    return names


def query_one_label_with_odbc(mid):
    import executor.sparql_executor as se

    if se.odbc_conn is None:
        se.initialize_odbc_connection()

    queries = [
        f"""
SPARQL
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?label WHERE {{
  ns:{mid} rdfs:label ?label .
  FILTER (langMatches(lang(?label), "EN"))
}}
LIMIT 1
""",
        f"""
SPARQL
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?label WHERE {{
  ns:{mid} ns:type.object.name ?label .
}}
LIMIT 1
""",
    ]

    for query in queries:
        try:
            with se.odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchmany(10)
        except Exception:
            continue

        for row in rows:
            if row and row[0] is not None:
                return str(row[0])

    return None


def load_entity_names_from_odbc(target_ids):
    names = {}
    missing = 0

    for mid in tqdm(target_ids, desc="Loading ODBC entity labels"):
        label = query_one_label_with_odbc(mid)
        if label:
            names[mid] = label
        else:
            missing += 1

    print(
        "ODBC label map:",
        f"total={len(target_ids)}",
        f"hit={len(names)}",
        f"miss={missing}",
    )
    return names


def load_cwq_gold(gold_file, use_labels, entity_list_file=None):
    with open(gold_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gold = {}
    answer_ids = collect_cwq_answer_ids(data)
    entity_name_map = load_entity_names_from_file(
        entity_list_file,
        answer_ids,
    )

    if use_labels:
        missing_ids = [mid for mid in answer_ids if mid not in entity_name_map]
        entity_name_map.update(load_entity_names_from_odbc(missing_ids))

    unresolved = 0

    for item in tqdm(data, desc="Loading CWQ gold"):
        answers = []
        for ans in item.get("answer", []):
            value = ans
            if isinstance(ans, str) and ans in entity_name_map:
                value = entity_name_map[ans]
            elif isinstance(ans, str) and ans.startswith(("m.", "g.")):
                unresolved += 1
            answers.append(value)

        gold[item["ID"]] = [answers]

    if unresolved:
        print(f"Warning: {unresolved} CWQ gold answers remain unresolved MIDs.")

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
    parser.add_argument(
        "--cwq_entity_list_file",
        default=None,
        help="Optional Freebase entity list file for mapping CWQ answer MIDs to names.",
    )
    args = parser.parse_args()

    if args.dataset == "WebQSP":
        gold = load_webqsp_gold(args.gold_file)
    else:
        gold = load_cwq_gold(
            args.gold_file,
            use_labels=args.cwq_use_labels,
            entity_list_file=args.cwq_entity_list_file,
        )

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
