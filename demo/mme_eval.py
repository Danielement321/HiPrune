"""MME benchmark for per-request token pruning in vLLM.

Runs the full MME benchmark (yes/no perception + cognition questions,
from the lmms-lab/MME Hugging Face dataset; 2,374 questions over 1,187
images, 2 questions per image) against a vLLM server for ONE pruning
method and reports, per configuration:

- overall accuracy and the official MME scores: per category
  score = 100 * (acc + acc+), where acc+ counts an image only if BOTH
  its questions are answered correctly; perception score sums the 10
  perception categories (max 2000), cognition the 4 cognition
  categories (max 800)
- actual retained visual tokens from `token_pruning_metadata`
- optionally a small serial TTFT pass (--timing-samples > 0; off by
  default)

`token_pruning` is a KEEP ratio: 0.14 retains 14% of visual tokens.
Method semantics match pope_eval.py (see prune_bench_common.py).
MME always runs its full, deterministic question set, so no sample
manifest is needed.

Decoding is greedy (temperature 0); MME questions already end with
"Please answer yes or no.", so replies parse trivially.

Outputs into --out-dir:
- mme_results.jsonl — one line per request
- mme_summary.json  — per-config aggregates
- mme_summary.txt   — human-readable table

Usage (next to the server):
    python3 mme_eval.py --url http://localhost:8124 \
        --model google/gemma-4-e4b-it --method hiprune \
        --ratios 0.75 0.5 0.25 0.14 --concurrency 8 --out-dir ./mme_out

Requires: datasets, pillow.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prune_bench_common import (
    ALL_METHODS, DEFAULT_RATIOS, DEFAULT_STRIDES, build_configs,
    config_row_fields, parse_token_metadata, token_aggregates,
)

COGNITION_CATEGORIES = {
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning",
}


def encode_image(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def load_mme() -> list[dict]:
    from datasets import load_dataset

    print("loading lmms-lab/MME (test split)...")
    ds = load_dataset("lmms-lab/MME", split="test")
    samples = []
    # Two consecutive rows share one image (yes + no question); encode
    # each unique image once and reuse the base64.
    b64_cache: dict[str, str] = {}
    for row in ds:
        key = f"{row['category']}/{row['question_id']}"
        if key not in b64_cache:
            b64_cache[key] = encode_image(row["image"])
        samples.append({
            "id": key,
            "category": row["category"],
            "question": row["question"].strip(),
            "answer": row["answer"].strip().lower(),
            "image_b64": b64_cache[key],
        })
    print(f"loaded {len(samples)} questions, "
          f"{len(b64_cache)} images, "
          f"{len({s['category'] for s in samples})} categories")
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
                {"type": "text", "text": sample["question"]},
            ],
        }],
        "max_tokens": 10,
        "temperature": 0,
    }
    body.update(body_fields)
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        body["cache_salt"] = uuid.uuid4().hex
    return body


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
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
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
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


def mme_scores(records: list[dict]) -> dict:
    """Official MME scoring: per category, score = 100*(acc + acc+)."""
    per_category = {}
    for cat in sorted({r["category"] for r in records}):
        cat_recs = [r for r in records if r["category"] == cat]
        acc = sum(r["correct"] for r in cat_recs) / len(cat_recs)
        by_image: dict[str, list[bool]] = {}
        for r in cat_recs:
            by_image.setdefault(r["id"], []).append(r["correct"])
        acc_plus = sum(all(v) for v in by_image.values()) / len(by_image)
        per_category[cat] = {
            "acc": acc,
            "acc_plus": acc_plus,
            "score": 100 * (acc + acc_plus),
        }
    perception = sum(v["score"] for c, v in per_category.items()
                     if c not in COGNITION_CATEGORIES)
    cognition = sum(v["score"] for c, v in per_category.items()
                    if c in COGNITION_CATEGORIES)
    return {
        "per_category": per_category,
        "perception_score": perception,   # max 2000
        "cognition_score": cognition,     # max 800
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8124")
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--method", default="hiprune", choices=ALL_METHODS)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=DEFAULT_RATIOS,
                        help="requested keep ratios (ratio methods only)")
    parser.add_argument("--strides", type=int, nargs="+",
                        default=DEFAULT_STRIDES,
                        help="lattice strides (nprune only)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timing-samples", type=int, default=0,
                        help="serial streaming requests per config for "
                             "TTFT; 0 (default) skips the timing pass")
    parser.add_argument("--out-dir", default="./mme_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_mme()

    configs = build_configs(args.method, args.ratios, args.strides)
    results_path = out_dir / "mme_results.jsonl"
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
                "question": sample["question"],
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
        scores = mme_scores(records)
        print(f"accuracy {correct / n:.3f} | "
              f"perception {scores['perception_score']:.1f}/2000 | "
              f"cognition {scores['cognition_score']:.1f}/800 | "
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
            "ratio": cfg["requested_keep"],
            "num_samples": n,
            "accuracy": correct / n,
            "invalid": invalid,
            "perception_score": scores["perception_score"],
            "cognition_score": scores["cognition_score"],
            "per_category": scores["per_category"],
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
    summary_path = out_dir / "mme_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": "lmms-lab/MME (test, full)",
            "benchmark": "mme",
            "model": args.model,
            "method": args.method,
            "num_samples": len(samples),
            "results": summaries,
        }, f, indent=2)

    def fmt(v, spec=".3f", default="—"):
        return format(v, spec) if v is not None else default

    header = (f"{'config':>10} {'req keep':>9} {'eff keep':>9} "
              f"{'retained':>9} {'accuracy':>9} {'perception':>11} "
              f"{'cognition':>10} {'invalid':>8}")
    lines = [
        f"MME | {args.model} | method {args.method} | "
        f"{len(samples)} questions (full benchmark)",
        "", header, "-" * len(header),
    ]
    for s in summaries:
        lines.append(
            f"{s['label']:>10} {s['requested_keep']:>9.2f} "
            f"{fmt(s['mean_effective_keep']):>9} "
            f"{fmt(s['mean_retained_tokens'], '.1f'):>9} "
            f"{s['accuracy']:>9.3f} "
            f"{s['perception_score']:>11.1f} {s['cognition_score']:>10.1f} "
            f"{s['invalid']:>8}")
    table = "\n".join(lines)
    with open(out_dir / "mme_summary.txt", "w") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nwrote {results_path}, {summary_path}, "
          f"{out_dir / 'mme_summary.txt'}")


if __name__ == "__main__":
    main()
