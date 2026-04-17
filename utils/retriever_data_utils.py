from typing import Dict, List

import torch


def build_training_pairs(data: List[Dict], num_negatives: int = 5) -> List[Dict]:
    if num_negatives <= 0:
        raise ValueError("num_negatives must be > 0")

    pairs: List[Dict] = []
    for row in data:
        question = row.get("question", "")
        positive = row.get("positive", [])
        negatives = row.get("negatives", [])

        if not question or not positive or not negatives:
            continue

        clean_negs = [n for n in negatives if isinstance(n, str) and n.strip()]
        if not clean_negs:
            continue

        # Use one positive and many negatives per sample.
        positive = positive if isinstance(positive, str) and positive.strip() else None
        if positive is None:
            continue

        pairs.append(
            {
                "question": question,
                "positive": positive,
                "negatives": clean_negs,
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


def tokenize_multi_negative_pairs(
    pairs: List[Dict],
    tokenizer,
    max_q_len: int,
    max_c_len: int,
) -> Dict[str, torch.Tensor]:
    questions = [row["question"] for row in pairs]
    positives = [row["positive"] for row in pairs]

    negatives_per_sample = []
    for row in pairs:
        neg_list = [n for n in row.get("negatives", []) if isinstance(n, str) and n.strip()]
        if not neg_list:
            raise ValueError("Each sample must contain at least one negative.")
        negatives_per_sample.append(neg_list)

    num_negatives = len(negatives_per_sample[0])
    if num_negatives <= 0:
        raise ValueError("num_negatives must be > 0")
    for neg_list in negatives_per_sample:
        if len(neg_list) != num_negatives:
            raise ValueError("All samples must have the same number of negatives.")

    flat_negatives = [neg for neg_list in negatives_per_sample for neg in neg_list]

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
        flat_negatives,
        padding="max_length",
        truncation=True,
        max_length=max_c_len,
        return_tensors="pt",
    )

    batch_size = len(pairs)
    n_input_ids = n_batch["input_ids"].view(batch_size, num_negatives, -1)
    n_attention_mask = n_batch["attention_mask"].view(batch_size, num_negatives, -1)

    return {
        "q_input_ids": q_batch["input_ids"],
        "q_attention_mask": q_batch["attention_mask"],
        "p_input_ids": p_batch["input_ids"],
        "p_attention_mask": p_batch["attention_mask"],
        "n_input_ids": n_input_ids,
        "n_attention_mask": n_attention_mask,
    }
