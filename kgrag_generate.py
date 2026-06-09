import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You answer knowledge-base-style factual questions using the provided knowledge graph triplets.
Use only the triplets as evidence. If the triplets do not support any answer, return an empty list.
Return only a JSON object with an "answers" list.
Each answer must be a short canonical entity name, value, date, number, or type.
Do not return full sentences.
Do not include explanations."""


TRIPLET_FIELD_CANDIDATES = (
    "scored_triples",
    "scored_triplets",
    "triples",
    "triplets",
    "retrieved_triples",
    "retrieved_triplets",
    "graph",
)


def load_questions(dataset, input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    if dataset == "WebQSP":
        for item in data["Questions"]:
            qid = item["QuestionId"]
            question = item.get("ProcessedQuestion") or item.get("RawQuestion")
            if question:
                examples.append({"qid": qid, "question": question})
    elif dataset == "CWQ":
        for item in data:
            examples.append({"qid": item["ID"], "question": item["question"]})
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return examples


def parse_answers(text):
    text = text.strip()

    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return []

    answers = obj.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(answers, list):
        return []

    return [str(x).strip() for x in answers if str(x).strip()]


def to_python_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            return value.item()
        except Exception:
            pass
    return value


def is_sequence_like(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def stringify(value):
    value = to_python_value(value)
    return str(value).strip()


def parse_score(value):
    value = to_python_value(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def normalize_triplet(raw):
    """Return (head, relation, tail, score) or None."""
    if isinstance(raw, Mapping):
        head = raw.get("head", raw.get("h", raw.get("subject", raw.get("src"))))
        relation = raw.get("relation", raw.get("rel", raw.get("r", raw.get("predicate"))))
        tail = raw.get("tail", raw.get("t", raw.get("object", raw.get("dst"))))
        if head is None or relation is None or tail is None:
            return None
        score = raw.get("score", raw.get("logit", raw.get("prob")))
        return stringify(head), stringify(relation), stringify(tail), parse_score(score)

    if not is_sequence_like(raw):
        return None

    items = list(raw)
    if len(items) == 2:
        first, second = items
        if is_sequence_like(first) and len(first) >= 3:
            head, relation, tail = list(first)[:3]
            return stringify(head), stringify(relation), stringify(tail), parse_score(second)
        if is_sequence_like(second) and len(second) >= 3:
            head, relation, tail = list(second)[:3]
            return stringify(head), stringify(relation), stringify(tail), parse_score(first)

    if len(items) >= 3:
        head, relation, tail = items[:3]
        score = parse_score(items[3]) if len(items) >= 4 else None
        return stringify(head), stringify(relation), stringify(tail), score

    return None


def extract_triplets(record, preferred_field=None):
    if preferred_field:
        fields = (preferred_field,) + tuple(
            field for field in TRIPLET_FIELD_CANDIDATES if field != preferred_field
        )
    else:
        fields = TRIPLET_FIELD_CANDIDATES

    raw_triplets = None
    if isinstance(record, Mapping):
        for field in fields:
            if field in record:
                raw_triplets = record[field]
                break
    elif is_sequence_like(record):
        raw_triplets = record

    if raw_triplets is None:
        return []

    triplets = []
    for raw in raw_triplets:
        triplet = normalize_triplet(raw)
        if triplet is not None and all(triplet[:3]):
            triplets.append(triplet)

    return triplets


def iter_records(obj):
    if isinstance(obj, Mapping):
        # HuggingFace/torch files are usually {qid: {"scored_triples": [...]}}.
        if any(field in obj for field in TRIPLET_FIELD_CANDIDATES):
            yield None, obj
        else:
            for key, value in obj.items():
                yield key, value
    elif isinstance(obj, list):
        for value in obj:
            key = None
            if isinstance(value, Mapping):
                key = (
                    value.get("id")
                    or value.get("qid")
                    or value.get("question_id")
                    or value.get("QuestionId")
                    or value.get("ID")
                )
            yield key, value
    else:
        raise ValueError(f"Unsupported triplet bank type: {type(obj)}")


def load_triplet_bank(triples_file, preferred_field=None):
    try:
        bank_obj = torch.load(triples_file, map_location="cpu", weights_only=False)
    except TypeError:
        bank_obj = torch.load(triples_file, map_location="cpu")

    bank = {}
    question_bank = {}
    for key, record in iter_records(bank_obj):
        if isinstance(record, Mapping):
            key = (
                key
                or record.get("id")
                or record.get("qid")
                or record.get("question_id")
                or record.get("QuestionId")
                or record.get("ID")
            )

        if key is None:
            continue

        triplets = extract_triplets(record, preferred_field=preferred_field)
        bank[str(key)] = triplets

        if isinstance(record, Mapping) and record.get("question"):
            question_bank[normalize_text(record["question"])] = triplets

    return bank, question_bank


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower().strip().rstrip("?"))


def unique_triplets(triplets):
    seen = set()
    unique = []
    for head, relation, tail, score in triplets:
        key = (head, relation, tail)
        if key in seen:
            continue
        seen.add(key)
        unique.append((head, relation, tail, score))
    return unique


def select_triplets(ex, triplet_bank, question_bank, top_k, score_threshold=None):
    triplets = triplet_bank.get(ex["qid"])
    if triplets is None:
        triplets = triplet_bank.get(ex["qid"].lower())
    if triplets is None:
        triplets = question_bank.get(normalize_text(ex["question"]))
    if triplets is None:
        return [], 0

    available = len(triplets)
    selected = []
    for head, relation, tail, score in unique_triplets(triplets):
        if score_threshold is not None and score is not None and score < score_threshold:
            continue
        selected.append((head, relation, tail, score))
        if len(selected) >= top_k:
            break

    return selected, available


def format_triplet_for_prompt(triplet):
    head, relation, tail, _score = triplet
    return f"({head}, {relation}, {tail})"


def triplet_for_output(triplet):
    head, relation, tail, score = triplet
    obj = {"head": head, "relation": relation, "tail": tail}
    if score is not None:
        obj["score"] = score
    return obj


def build_messages(question, triplets):
    if triplets:
        triplet_block = "\n".join(format_triplet_for_prompt(triplet) for triplet in triplets)
    else:
        triplet_block = "(no triplets)"

    user_content = (
        "Triplets:\n"
        f"{triplet_block}\n\n"
        f"Question: {question}\n\n"
        'Output format:\n{"answers": ["..."]}'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_prompt(tokenizer, question, triplets):
    messages = build_messages(question, triplets)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_batch(model, tokenizer, batch_items, max_new_tokens, max_input_tokens=None):
    prompts = [
        build_prompt(tokenizer, item["question"], item["triplets"])
        for item in batch_items
    ]

    tokenizer_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": max_input_tokens is not None,
    }
    if max_input_tokens is not None:
        tokenizer_kwargs["max_length"] = max_input_tokens

    inputs = tokenizer(prompts, **tokenizer_kwargs).to(model.device)
    prompt_width = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for output in outputs:
        generated = output[prompt_width:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["WebQSP", "CWQ"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--triples_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--triplet_field", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--missing_triples",
        default="empty",
        choices=["empty", "skip", "error"],
        help="What to do when a question has no matching retrieved triplets.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    examples = load_questions(args.dataset, args.input_file)
    triplet_bank, question_bank = load_triplet_bank(
        args.triples_file,
        preferred_field=args.triplet_field,
    )
    triplet_bank.update({key.lower(): value for key, value in list(triplet_bank.items())})

    prepared = []
    missing = []
    for ex in examples:
        selected, available = select_triplets(
            ex,
            triplet_bank,
            question_bank,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
        )

        if not selected:
            missing.append(ex["qid"])
            if args.missing_triples == "error":
                raise KeyError(f"No retrieved triplets found for qid={ex['qid']}")
            if args.missing_triples == "skip":
                continue

        prepared.append(
            {
                "qid": ex["qid"],
                "question": ex["question"],
                "triplets": selected,
                "num_triples_available": available,
            }
        )

    print(
        "KG-RAG examples:",
        f"dataset={args.dataset}",
        f"questions={len(examples)}",
        f"prepared={len(prepared)}",
        f"missing_triplets={len(missing)}",
        f"top_k={args.top_k}",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for start in tqdm(range(0, len(prepared), args.batch_size), desc=f"KG-RAG {args.dataset}"):
        batch = prepared[start:start + args.batch_size]
        raws = generate_batch(
            model,
            tokenizer,
            batch,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
        )

        for item, raw in zip(batch, raws):
            results.append(
                {
                    "qid": item["qid"],
                    "question": item["question"],
                    "triples_file": args.triples_file,
                    "top_k": args.top_k,
                    "score_threshold": args.score_threshold,
                    "num_triples_available": item["num_triples_available"],
                    "num_triples_used": len(item["triplets"]),
                    "triples_used": [triplet_for_output(triplet) for triplet in item["triplets"]],
                    "prediction_raw": raw,
                    "pred_answers": parse_answers(raw),
                }
            )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if missing:
        missing_file = f"{args.output_file}.missing_triplets.json"
        with open(missing_file, "w", encoding="utf-8") as f:
            json.dump(missing, f, indent=2, ensure_ascii=False)
        print(f"Wrote missing qids to {missing_file}")


if __name__ == "__main__":
    main()
