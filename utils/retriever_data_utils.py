from typing import Dict, List

import torch


def build_training_pairs(data: List[Dict]) -> List[Dict]:
    pairs: List[Dict] = []
    for row in data:
        question = row.get("question", "")
        positives = row.get("positive_chunks", [])
        negatives = row.get("negative_chunks", [])

        if not question or not positives or not negatives:
            continue

        pair_count = min(len(positives), len(negatives))
        for idx in range(pair_count):
            pos = positives[idx]
            neg = negatives[idx]

            if not (isinstance(pos, str) and pos.strip()):
                continue
            if not (isinstance(neg, str) and neg.strip()):
                continue

            pairs.append(
                {
                    "question": question,
                    "positive": pos,
                    "negatives": [neg],
                }
            )

    return pairs


def tokenize_one_negative_pairs(
    pairs: List[Dict],
    tokenizer,
    max_q_len: int,
    max_c_len: int,
) -> Dict[str, torch.Tensor]:
    questions = [row["question"] for row in pairs]
    positives = [row["positive"] for row in pairs]

    negatives: List[str] = []
    for row in pairs:
        neg_list = row.get("negatives", [])
        if not neg_list:
            raise ValueError("Each sample must contain exactly one negative.")
        negatives.append(neg_list[0])

    q_batch = tokenizer(
        questions,
        padding="max_length",
        truncation=True,
        max_length=max_q_len,
        return_tensors="pt",
    )
    p_batch = tokenizer(
        positives,
        padding="max_length",
        truncation=True,
        max_length=max_c_len,
        return_tensors="pt",
    )
    n_batch = tokenizer(
        negatives,
        padding="max_length",
        truncation=True,
        max_length=max_c_len,
        return_tensors="pt",
    )

    return {
        "q_input_ids": q_batch["input_ids"],
        "q_attention_mask": q_batch["attention_mask"],
        "p_input_ids": p_batch["input_ids"],
        "p_attention_mask": p_batch["attention_mask"],
        "n_input_ids": n_batch["input_ids"],
        "n_attention_mask": n_batch["attention_mask"],
    }
