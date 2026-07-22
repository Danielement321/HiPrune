"""POPE accuracy benchmark for per-request token pruning in vLLM.

Runs a balanced subset of POPE (yes/no object-hallucination questions on
COCO images, from the lmms-lab/POPE Hugging Face dataset) against a
vLLM server for ONE pruning method and measures, per configuration:

- accuracy, precision, recall, F1, and yes-rate (POPE's standard
  hallucination indicators)
- per-category accuracy (random / popular / adversarial)
- actual retained visual tokens from `token_pruning_metadata`
  (mean/median retained, mean effective keep ratio)
- optionally mean/median TTFT (--timing-samples > 0; off by default)

`token_pruning` is a KEEP ratio: 0.14 retains 14% of visual tokens.
Methods:
- baseline: no pruning fields at all
- hiprune / hydart / hiprune_pp / dart / anchorprune: swept over
  --ratios (requested keep ratios)
- nprune: swept over --strides (its keep count is ceil(H/s)*ceil(W/s),
  so requested ratios cannot be matched; effective keep is reported)
- checkered: one fixed ~50% pattern

Pass --manifest to pin the exact sample IDs (written once by
run_method_sweep.py) so every method sees identical examples.

Decoding is greedy (temperature 0) with "Answer yes or no." appended to
each question, so replies parse trivially.

Outputs into --out-dir:
- pope_results.jsonl — one line per request
- pope_summary.json  — per-config aggregates (input for plot_retention.py)
- pope_summary.txt   — human-readable table

Usage (next to the server):
    python3 pope_eval.py --url http://localhost:8123 \
        --method hiprune --num-samples 400 \
        --ratios 0.75 0.5 0.25 0.14 \
        --manifest benchmarks/pope_methods/sample_manifest.json \
        --concurrency 8 --out-dir ./pope_out

Requires: datasets, pillow (pip install datasets pillow).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prune_bench_common import (
    ALL_METHODS, DEFAULT_RATIOS, DEFAULT_STRIDES, apply_manifest,
    build_configs, config_row_fields, parse_token_metadata,
    token_aggregates,
)


def encode_image(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_pope_subset(num_samples: int, seed: int) -> list[dict]:
    """Load POPE and sample balanced across (category, answer) cells."""
    from datasets import load_dataset

    print("loading lmms-lab/POPE (test split)...")
    ds = load_dataset("lmms-lab/POPE", split="test")
    by_cell: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(ds):
        cell = (row["category"], row["answer"].strip().lower())
        by_cell.setdefault(cell, []).append(i)

    rng = random.Random(seed)
    cells = sorted(by_cell)
    per_cell = num_samples // len(cells)
    picked: list[int] = []
    for cell in cells:
        idxs = by_cell[cell]
        rng.shuffle(idxs)
        picked += idxs[:per_cell]

    samples = []
    for i in picked:
        row = ds[i]
        samples.append({
            "id": f"{row['category']}/{row['question_id']}",
            "category": row["category"],
            "question": row["question"].strip(),
            "answer": row["answer"].strip().lower(),
            "image_b64": encode_image(row["image"]),
        })
    rng.shuffle(samples)
    print(f"sampled {len(samples)} questions "
          f"({per_cell} per (category, answer) cell)")
    return samples


def build_body(sample: dict, model: str, body_fields: dict,
               stream: bool = False) -> dict:
    body: dict = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{sample['image_b64']}"}},
                {"type": "text",
                 "text": f"{sample['question']} Answer yes or no."},
            ],
        }],
        "max_tokens": 10,
        "temperature": 0,
    }
    body.update(body_fields)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        # Bust the prefix cache so TTFT reflects a full prefill.
        body["cache_salt"] = uuid.uuid4().hex
    return body


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def parse_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    has_yes = "yes" in t
    has_no = "no" in t.replace("not", " ").split() or t.startswith("no")
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


def measure_ttft(url: str, body: dict) -> float:
    """Serial streaming request; returns seconds to first content chunk."""
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                return time.perf_counter() - t0
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8123")
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--method", default="hiprune", choices=ALL_METHODS)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=DEFAULT_RATIOS,
                        help="requested keep ratios (ratio methods only)")
    parser.add_argument("--strides", type=int, nargs="+",
                        default=DEFAULT_STRIDES,
                        help="lattice strides (nprune only)")
    parser.add_argument("--manifest",
                        help="shared sample-ID manifest (see docstring)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timing-samples", type=int, default=0,
                        help="serial streaming requests per config for "
                             "TTFT; 0 (default) skips the timing pass")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="./pope_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_pope_subset(args.num_samples, args.seed)
    if args.manifest:
        samples = apply_manifest(samples, args.manifest, "pope", args.seed)

    configs = build_configs(args.method, args.ratios, args.strides)
    results_path = out_dir / "pope_results.jsonl"
    results_f = open(results_path, "w")
    summaries = []

    for cfg in configs:
        label = cfg["label"]
        print(f"\n=== {args.method} {label}: accuracy pass "
              f"({len(samples)} questions, concurrency {args.concurrency}) ===")

        def run_one(sample: dict, cfg: dict = cfg) -> dict:
            body = build_body(sample, args.model, cfg["body_fields"])
            t0 = time.perf_counter()
            resp = post_json(args.url, body)
            elapsed = time.perf_counter() - t0
            raw = resp["choices"][0]["message"]["content"]
            parsed = parse_yes_no(raw)
            return {
                "id": sample["id"],
                "category": sample["category"],
                **config_row_fields(args.method, cfg),
                "ground_truth": sample["answer"],
                "raw_answer": raw,
                "parsed": parsed,
                "correct": parsed == sample["answer"],
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
                **parse_token_metadata(resp),
                "latency_s": round(elapsed, 4),
            }

        t_pass = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = list(pool.map(run_one, samples))
        pass_time = time.perf_counter() - t_pass
        for rec in records:
            results_f.write(json.dumps(rec) + "\n")
        results_f.flush()

        n = len(records)
        correct = sum(r["correct"] for r in records)
        invalid = sum(r["parsed"] is None for r in records)
        tp = sum(r["parsed"] == "yes" and r["ground_truth"] == "yes"
                 for r in records)
        fp = sum(r["parsed"] == "yes" and r["ground_truth"] == "no"
                 for r in records)
        fn = sum(r["parsed"] != "yes" and r["ground_truth"] == "yes"
                 for r in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_category = {}
        for cat in sorted({r["category"] for r in records}):
            cat_recs = [r for r in records if r["category"] == cat]
            per_category[cat] = sum(r["correct"] for r in cat_recs) / len(cat_recs)

        print(f"accuracy {correct / n:.3f} | f1 {f1:.3f} | "
              f"invalid {invalid} | pass took {pass_time:.0f}s")

        ttfts = []
        if args.timing_samples > 0:
            print(f"=== {args.method} {label}: timing pass "
                  f"({args.timing_samples} serial streaming requests) ===")
            for sample in samples[:args.timing_samples]:
                body = build_body(sample, args.model, cfg["body_fields"],
                                  stream=True)
                ttfts.append(measure_ttft(args.url, body))
        prompt_tokens = [r["prompt_tokens"] for r in records
                         if r["prompt_tokens"] is not None]

        summary = {
            "method": args.method,
            "label": label,
            "requested_keep": cfg["requested_keep"],
            # legacy key kept for older tooling (plot_pope.py)
            "ratio": cfg["requested_keep"],
            "num_samples": n,
            "accuracy": correct / n,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_rate": sum(r["parsed"] == "yes" for r in records) / n,
            "invalid": invalid,
            "per_category_accuracy": per_category,
            "mean_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
            **token_aggregates(records, args.method == "baseline"),
            "accuracy_pass_seconds": pass_time,
        }
        if cfg["stride"] is not None:
            summary["stride"] = cfg["stride"]
        if ttfts:
            summary["mean_ttft_s"] = statistics.mean(ttfts)
            summary["median_ttft_s"] = statistics.median(ttfts)
        summaries.append(summary)

    results_f.close()
    summary_path = out_dir / "pope_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": "lmms-lab/POPE (test)",
            "benchmark": "pope",
            "model": args.model,
            "method": args.method,
            "num_samples": len(samples),
            "seed": args.seed,
            "manifest": args.manifest,
            "results": summaries,
        }, f, indent=2)

    def fmt(v, spec=".3f", default="—"):
        return format(v, spec) if v is not None else default

    header = (f"{'config':>10} {'req keep':>9} {'eff keep':>9} "
              f"{'retained':>9} {'accuracy':>9} {'f1':>7} {'yes-rate':>9} "
              f"{'invalid':>8}")
    lines = [
        f"POPE | {args.model} | method {args.method} | "
        f"{len(samples)} balanced samples",
        "", header, "-" * len(header),
    ]
    for s in summaries:
        lines.append(
            f"{s['label']:>10} {s['requested_keep']:>9.2f} "
            f"{fmt(s['mean_effective_keep']):>9} "
            f"{fmt(s['mean_retained_tokens'], '.1f'):>9} "
            f"{s['accuracy']:>9.3f} {s['f1']:>7.3f} "
            f"{s['yes_rate']:>9.3f} {s['invalid']:>8}")
    lines += ["", "per-category accuracy:"]
    for s in summaries:
        cats = " | ".join(f"{c} {a:.3f}"
                          for c, a in s["per_category_accuracy"].items())
        lines.append(f"  {s['label']:>10}: {cats}")
    table = "\n".join(lines)
    with open(out_dir / "pope_summary.txt", "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {results_path}, {summary_path}, "
          f"{out_dir / 'pope_summary.txt'}")


if __name__ == "__main__":
    main()
