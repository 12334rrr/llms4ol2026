#!/usr/bin/env python
"""
LLMs4OL 2026 — 推理 + 提交生成

Usage:
    python scripts/predict.py --model_path output/ontology_lora/final_model
    python scripts/predict.py --model_path ... --num_sc 3
    python scripts/predict.py --no_lora  # zero-shot 基线
"""

import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src import SYSTEM_PROMPT, postprocess_triples, find_local_model

MAX_NEW_TOKENS = 768
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(model_path=None, use_lora=True, base_model=None):
    """加载 4-bit 模型 + LoRA adapter. 自动查找本地模型."""
    base = base_model or DEFAULT_MODEL
    base_path = find_local_model(base)
    is_local = os.path.isdir(base_path)

    print(f"Loading base model: {base_path}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
        torch_dtype=torch.float16, local_files_only=is_local,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, local_files_only=is_local,
    )
    if use_lora and model_path and os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Loading LoRA: {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def extract_triples(text):
    if not text:
        return []
    for pat in [r'\{[^{}]*"triples"\s*:\s*\[.*?\][^{}]*\}', r'\[\[.*?\]\]']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if isinstance(data, dict):
                    return [[str(x) for x in t] for t in data.get("triples", []) if len(t) == 3]
                if isinstance(data, list):
                    return [[str(x) for x in t] for t in data if len(t) == 3]
            except:
                pass
    return [list(t) for t in re.findall(r'\[\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]', text)]


def generate(model, tokenizer, context, temp=0.1):
    prompt = (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
              f"<|im_start|>user\n{context}<|im_end|>\n<|im_start|>assistant\n")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=temp,
                             top_p=0.9, do_sample=(temp > 0),
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_triples(text.replace("<|im_end|>", "").strip())


def generate_ensemble(model, tokenizer, context, n=3, temp=0.3, threshold=0.5):
    all_triples = [generate(model, tokenizer, context, temp) for _ in range(n)]
    if n == 1:
        return all_triples[0]
    votes = Counter()
    for triples in all_triples:
        for t in triples:
            if len(t) == 3:
                votes[(t[0].strip(), t[1].strip(), t[2].strip())] += 1
    return [[s, p, o] for (s, p, o), c in votes.items() if c / n >= threshold]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default="data/test_task_a_input.json")
    parser.add_argument("--output", type=str, default="submission.json")
    parser.add_argument("--num_sc", type=int, default=1)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model (default: Qwen/Qwen2.5-7B-Instruct, auto-finds local)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, not args.no_lora, args.base_model)
    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")

    preds, total, failed = [], 0, 0
    for i, s in enumerate(tqdm(test_data)):
        triples = (generate_ensemble(model, tokenizer, s["context"], n=args.num_sc, temp=0.3)
                   if args.num_sc > 1 else generate(model, tokenizer, s["context"]))
        triples = postprocess_triples(triples)
        if not triples:
            failed += 1
        total += len(triples)
        preds.append({"id": s["id"], "primitive-ontology-triples": triples})
        if i < 3:
            print(f"\n  Sample {i+1}: {len(triples)} triples")
            for t in triples[:3]:
                print(f"    {t}")

    print(f"\nAvg triples/sample: {total / len(test_data):.1f}, Failed: {failed}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"Saved: {args.output} ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
