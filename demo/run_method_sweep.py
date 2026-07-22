"""Run the full pruning-method sweep for one benchmark.

Drives pope_eval.py / mme_eval.py / textvqa_eval.py once per method so
every method is scored on identical examples (POPE/TextVQA pin their
sample IDs through a shared manifest written on the first run; MME
always runs its full question set):

1. no-op smoke  — baseline vs every ratio method at nominal keep 1.0 on
   the repo demo images; asserts the enabled prune path removes zero
   tokens and answers match baseline (catches a method whose code path
   changes the representation even at full retention)
2. baseline     — no pruning fields             -> <out-root>/baseline/
3. ratio methods (hiprune, hydart, hiprune_pp, dart, anchorprune)
   at requested keep 0.75/0.50/0.25/0.14        -> <out-root>/<method>/
4. nprune       — lattice strides 2, 3, 4       -> <out-root>/nprune/
5. checkered    — fixed ~50% pattern            -> <out-root>/checkered/

Afterwards, plot with:
    python3 plot_retention.py --root <out-root>

Usage:
    python3 run_method_sweep.py --benchmark pope \
        --url http://localhost:8123 --model google/gemma-4-e4b-it \
        --num-samples 400 --seed 0 --out-root benchmarks/pope_methods
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from prune_bench_common import RATIO_METHODS

EVAL_SCRIPTS = {
    "pope": "pope_eval.py",
    "mme": "mme_eval.py",
    "textvqa": "textvqa_eval.py",
}

SMOKE_IMAGES = ["dog.jpg", "pyramids.jpg"]
SMOKE_PROMPT = "Describe this image in one short sentence."


def post_json(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def smoke_body(image_b64: str, model: str, method: str | None) -> dict:
    body: dict = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": SMOKE_PROMPT},
            ],
        }],
        "max_tokens": 24,
        "temperature": 0,
    }
    if method is not None:
        # Enabled prune path at nominal full retention.
        body["token_pruning"] = 1.0
        body["token_pruning_method"] = method
    return body


def noop_smoke(url: str, model: str, demo_dir: Path) -> bool:
    """Baseline vs each ratio method at keep 1.0; returns True on pass."""
    images = [p for p in (demo_dir / n for n in SMOKE_IMAGES) if p.exists()]
    if not images:
        print("no-op smoke: no demo images found, skipping")
        return True

    ok = True
    for img_path in images:
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        base_resp = post_json(url, smoke_body(b64, model, None))
        base_answer = base_resp["choices"][0]["message"]["content"].strip()
        for method in RATIO_METHODS:
            resp = post_json(url, smoke_body(b64, model, method))
            answer = resp["choices"][0]["message"]["content"].strip()
            md = (resp.get("token_pruning_metadata") or [None])[0]
            # At keep 1.0 the server may short-circuit the prune path and
            # return no metadata (treated as a no-op), or return metadata
            # with an empty pruned list. Either is fine as long as the
            # answer matches baseline.
            if md is None:
                pruned = None
                zero = True
            else:
                pruned = len(md.get("pruned") or [])
                zero = pruned == 0
            same = answer == base_answer
            status = "PASS" if (same and zero) else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {status}: {img_path.name} {method} @ keep 1.0 "
                  f"(pruned={pruned}, answer match={same})")
    return ok


def run_eval(script: Path, args_list: list[str], log_path: Path) -> None:
    cmd = [sys.executable, str(script)] + args_list
    print(f"\n>>> {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"{script.name} failed (exit {proc.returncode}), "
                         f"see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(EVAL_SCRIPTS),
                        default="pope")
    parser.add_argument("--url", default="http://localhost:8123")
    parser.add_argument("--model", default="google/gemma-4-e4b-it")
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.75, 0.50, 0.25, 0.14])
    parser.add_argument("--strides", type=int, nargs="+", default=[2],
                        help="nprune strides (deployed server allows 1 or 2)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default="benchmarks/pope_methods")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--methods", nargs="+",
                        default=["baseline", *RATIO_METHODS, "nprune",
                                 "checkered"],
                        help="subset of methods to (re)run")
    args = parser.parse_args()

    demo_dir = Path(__file__).resolve().parent
    script = demo_dir / EVAL_SCRIPTS[args.benchmark]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "sample_manifest.json"

    if not args.skip_smoke:
        print("=== no-op smoke: enabled prune path at keep 1.0 must prune "
              "nothing ===")
        if not noop_smoke(args.url, args.model, demo_dir):
            raise SystemExit("no-op smoke FAILED; not running the sweep")

    t0 = time.perf_counter()
    for method in args.methods:
        out_dir = out_root / method
        common = [
            "--url", args.url,
            "--model", args.model,
            "--method", method,
            "--concurrency", str(args.concurrency),
            "--out-dir", str(out_dir),
            "--ratios", *[str(r) for r in args.ratios],
            "--strides", *[str(s) for s in args.strides],
        ]
        # MME runs its full fixed question set; the sampled benchmarks
        # pin their examples through the shared manifest.
        if args.benchmark != "mme":
            common += ["--num-samples", str(args.num_samples),
                       "--seed", str(args.seed),
                       "--manifest", str(manifest)]
        run_eval(script, common, out_dir / f"{args.benchmark}_eval.log")

    elapsed = time.perf_counter() - t0
    print(f"\nsweep complete in {elapsed / 60:.1f} min -> {out_root}")
    print(f"next: python3 {demo_dir / 'plot_retention.py'} "
          f"--root {out_root}")


if __name__ == "__main__":
    main()
