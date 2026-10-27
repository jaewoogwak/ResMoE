"""Real activation collection and end-to-end Mixtral evaluation."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import install_layer
from .progress import progress_step


def load_original(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory = None
    if torch.cuda.is_available():
        free_gib = int(torch.cuda.mem_get_info()[0] / 2**30)
        max_memory = {0: f"{max(1, free_gib - 16)}GiB", "cpu": "240GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, low_cpu_mem_usage=True).eval()
    return model, tokenizer


@torch.inference_mode()
def collect_activations(model_name: str, layers: list[int], root: Path,
                        samples: int, seq_len: int) -> dict[int, dict[int, torch.Tensor]]:
    """Capture true routed Mixtral input activations from WikiText text."""
    print(f"[analyze] loading original model for WikiText calibration "
          f"({len(layers)} layers, target {samples} routed tokens/expert)", flush=True)
    model, tokenizer = load_original(model_name)
    collected = {layer: {expert: [] for expert in range(8)} for layer in layers}
    handles = []
    for layer in layers:
        moe = model.model.layers[layer].block_sparse_moe

        def capture(module, inputs, layer_id=layer):
            hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            chosen = module.gate(hidden).topk(2, dim=-1).indices
            for expert in range(8):
                left = samples - len(collected[layer_id][expert])
                if left <= 0:
                    continue
                indices = (chosen == expert).any(dim=-1).nonzero().flatten()[:left]
                collected[layer_id][expert].extend(hidden[indices].cpu().unbind(0))

        handles.append(moe.register_forward_pre_hook(capture))
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    seen = 0
    input_device = model.model.embed_tokens.weight.device
    with progress_step("[analyze] WikiText activation collection"):
        for row in dataset:
            if not row["text"].strip():
                continue
            ids = tokenizer(row["text"], return_tensors="pt", truncation=True,
                            max_length=seq_len)["input_ids"].to(input_device)
            if ids.numel() < 2:
                continue
            model(input_ids=ids, use_cache=False)
            seen += 1
            if seen % 10 == 0:
                print(f"[analyze] WikiText documents {seen}/{samples}", flush=True)
            if seen >= samples or all(len(collected[layer][e]) >= samples
                                      for layer in layers for e in range(8)):
                break
    for handle in handles:
        handle.remove()
    result = {layer: {expert: torch.stack(values) if values else torch.empty(0, model.config.hidden_size)
                      for expert, values in by_expert.items()}
              for layer, by_expert in collected.items()}
    target = root / "calibration"
    target.mkdir(parents=True, exist_ok=True)
    for layer, by_expert in result.items():
        torch.save(by_expert, target / f"layer_{layer}.pt")
        (target / f"layer_{layer}.json").write_text(json.dumps({
            "model": model_name, "dataset": "wikitext-103-raw-v1/train",
            "calibration_samples": samples, "calibration_seq_len": seq_len,
            "routed_tokens_per_expert": {str(expert): len(values)
                                         for expert, values in by_expert.items()}}, indent=2) + "\n")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def wikitext_perplexity(model, tokenizer, max_chunks: int | None,
                        seq_len: int = 1024) -> float:
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(row for row in dataset["text"] if row.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    input_device = model.model.embed_tokens.weight.device
    total_loss, total_tokens = 0.0, 0
    for start in range(0, ids.numel() - 1, seq_len):
        if max_chunks is not None and start // seq_len >= max_chunks:
            break
        chunk = ids[start:start + seq_len + 1].unsqueeze(0).to(input_device)
        if chunk.shape[1] < 2:
            continue
        logits = model(input_ids=chunk[:, :-1], use_cache=False).logits[:, - (chunk.shape[1] - 1):]
        labels = chunk[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                                 labels.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tokens += labels.numel()
    if not total_tokens:
        raise ValueError("no WikiText tokens evaluated")
    return math.exp(total_loss / total_tokens)


def evaluate_model(model, tokenizer, tasks: set[str], batch_size: int,
                   limit: int | None) -> dict:
    result = {}
    mc = sorted(tasks & {"winogrande", "piqa"})
    if mc:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        harness = HFLM(pretrained=model, tokenizer=tokenizer,
                       batch_size=batch_size, device="cuda:0")
        raw = evaluator.simple_evaluate(model=harness, tasks=mc, num_fewshot=0,
                                        batch_size=batch_size, limit=limit)
        for task in mc:
            score = raw["results"][task]
            result[task] = float(score.get("acc,none", score.get("acc_norm,none")))
        del harness
    if "wikitext" in tasks:
        result["wikitext_ppl"] = wikitext_perplexity(model, tokenizer, limit)
    return result


def evaluate_all(model_name: str, root: Path, methods: list[str], layers: list[int],
                 dimension: int, seed: int, tasks: set[str], batch_size: int,
                 limit: int | None) -> list[dict]:
    print(f"[eval] loading original model for D={dimension}", flush=True)
    model, tokenizer = load_original(model_name)
    with progress_step(f"[eval] original BF16 D={dimension}"):
        original_scores = evaluate_model(model, tokenizer, tasks, batch_size, limit)
    rows = [{"method": "original-bf16", "dimension": 0, "seed": seed,
             "layers": ",".join(map(str, layers)), **original_scores}]
    for method_number, method in enumerate(methods, start=1):
        print(f"[eval] method {method_number}/{len(methods)}: {method} D={dimension} "
              f"installing {len(layers)} layers", flush=True)
        for layer in layers:
            install_layer(model, root, method, dimension, seed, layer)
            print(f"[eval] {method} D={dimension}: installed layer {layer}", flush=True)
        with progress_step(f"[eval] {method} D={dimension} tasks {','.join(sorted(tasks))}"):
            scores = evaluate_model(model, tokenizer, tasks, batch_size, limit)
        rows.append({"method": method,
                     "dimension": 0 if method == "resmoe-bf16" else dimension,
                     "seed": 0 if method == "resmoe-bf16" else seed,
                     "layers": ",".join(map(str, layers)),
                     **scores})
    return rows
