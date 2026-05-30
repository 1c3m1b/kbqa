import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You answer knowledge-base-style factual questions.
Do not use tools, search, retrieved documents, or external knowledge bases.
Return only a JSON object with an "answers" list.
Do not include explanations."""


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
            qid = item["ID"]
            question = item["question"]
            examples.append({"qid": qid, "question": question})

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return examples


def build_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f'Question: {question}\n\nOutput format:\n{{"answers": ["..."]}}',
        },
    ]


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


def generate_one(model, tokenizer, question, max_new_tokens):
    messages = build_messages(question)

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["WebQSP", "CWQ"])
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    examples = load_questions(args.dataset, args.input_file)

    results = []
    for ex in tqdm(examples, desc=f"Generating {args.dataset}"):
        raw = generate_one(model, tokenizer, ex["question"], args.max_new_tokens)
        pred_answers = parse_answers(raw)

        results.append(
            {
                "qid": ex["qid"],
                "question": ex["question"],
                "prediction_raw": raw,
                "pred_answers": pred_answers,
            }
        )

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
