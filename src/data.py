"""
数据处理 — 加载训练数据，ChatML 格式化，过采样
"""

import json
import numpy as np
from collections import Counter
from typing import List, Tuple
from torch.utils.data import Dataset

from .config import SYSTEM_PROMPT


def load_and_prepare_data(
    data_path: str,
    val_split: float = 0.05,
    seed: int = 42,
    oversample_threshold: int = 200,
    oversample_multiplier: int = 5,
) -> Tuple[List[dict], List[dict]]:
    """加载数据 → ChatML格式化 → 过采样低频谓词 → 划分 train/val"""

    print(f"Loading data: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 统计谓词频率
    pred_counter = Counter()
    for sample in raw_data:
        for t in sample.get("primitive-ontology-triples", []):
            if len(t) == 3:
                pred_counter[t[1]] += 1

    print(f"  Predicates: {len(pred_counter)} types, top 5:")
    for pred, cnt in pred_counter.most_common(5):
        print(f"    {pred}: {cnt}")

    # ChatML 格式化
    formatted = []
    for sample in raw_data:
        triples = sample.get("primitive-ontology-triples", [])
        if not triples:
            continue

        target = json.dumps({"triples": triples}, ensure_ascii=False)
        full_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{sample['context']}<|im_end|>\n"
            f"<|im_start|>assistant\n{target}<|im_end|>"
        )

        has_rare = any(
            pred_counter[t[1]] < oversample_threshold
            for t in triples if len(t) == 3
        )

        formatted.append({
            "text": full_text,
            "num_triples": len(triples),
            "has_rare_predicate": has_rare,
        })

    # 过采样低频谓词样本
    rare_samples = [d for d in formatted if d["has_rare_predicate"]]
    print(f"  Original: {len(formatted)}, rare-predicate samples: {len(rare_samples)}")
    for _ in range(oversample_multiplier - 1):
        formatted.extend(rare_samples)
    print(f"  After {oversample_multiplier}x oversampling: {len(formatted)}")

    # 划分
    np.random.seed(seed)
    indices = np.random.permutation(len(formatted))
    val_size = max(1, int(len(formatted) * val_split))
    train_data = [formatted[i] for i in indices[val_size:].tolist()]
    val_data = [formatted[i] for i in indices[:val_size].tolist()]

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


class OntologyDataset(Dataset):
    """Tokenized 数据集，直接返回 input_ids + labels."""

    def __init__(self, data: List[dict], tokenizer, max_length: int = 4096):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.data[idx]["text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok
