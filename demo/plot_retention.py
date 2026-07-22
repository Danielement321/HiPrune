"""Accuracy vs effective retention across pruning methods.

Consumes the per-method summary JSONs produced by a run_method_sweep.py
sweep (one subdirectory per method under --root) and writes, into
--root:

- accuracy_vs_effective_retention.png — accuracy on y, MEAN EFFECTIVE
  keep ratio on x (actual retained / original visual tokens, from
  token_pruning_metadata), one line per method, baseline as a
  horizontal reference. Effective retention is the honest axis:
  pattern methods (nprune/checkered) cannot hit arbitrary requested
  ratios, and even ratio methods can deviate through rounding or
  protected tokens.
- retention_summary.json / .txt — merged table: method, requested keep,
  mean effective keep, mean retained tokens, accuracy, accuracy drop
  vs the shared baseline.

Usage:
    python3 plot_retention.py --root benchmarks/pope_methods
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_STYLE = {
    "hiprune": ("#d62728", "o"),
    "hydart": ("#1f77b4", "s"),
    "hiprune_pp": ("#ff7f0e", "^"),
    "dart": ("#2ca02c", "D"),
    "anchorprune": ("#9467bd", "v"),
    "nprune": ("#8c564b", "P"),
    "checkered": ("#7f7f7f", "X"),
}


def find_summaries(root: Path) -> list[dict]:
    summaries = []
    for path in sorted(root.glob("*/*_summary.json")):
        with open(path) as f:
            data = json.load(f)
        if "method" not in data:
            print(f"skipping {path} (no method field; pre-sweep format)")
            continue
        data["_path"] = str(path)
        summaries.append(data)
    if not summaries:
        raise SystemExit(f"no */*_summary.json with a method field "
                         f"under {root}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="sweep output root (one subdir per method)")
    args = parser.parse_args()
    root = Path(args.root)
    summaries = find_summaries(root)

    benchmark = summaries[0].get("benchmark", "benchmark")
    model = summaries[0]["model"]

    # Flatten to rows: one per (method, config).
    rows = []
    for s in summaries:
        for r in s["results"]:
            rows.append({
                "method": s["method"],
                "label": r.get("label", f"{r['ratio']:.2f}"),
                "requested_keep": r.get("requested_keep", r["ratio"]),
                "mean_effective_keep": r.get("mean_effective_keep"),
                "mean_retained_tokens": r.get("mean_retained_tokens"),
                "mean_original_tokens": r.get("mean_original_tokens"),
                "stride": r.get("stride"),
                "accuracy": r["accuracy"],
            })

    baseline_rows = [r for r in rows if r["method"] == "baseline"
                     or r["label"] == "baseline"]
    baseline = baseline_rows[0] if baseline_rows else None
    if baseline is None:
        print("warning: no baseline run found; accuracy drop left blank")

    for r in rows:
        r["accuracy_drop"] = (r["accuracy"] - baseline["accuracy"]
                              if baseline else None)

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted({r["method"] for r in rows if r["method"] != "baseline"})
    for method in methods:
        pts = [r for r in rows
               if r["method"] == method and r["label"] != "baseline"
               and r["mean_effective_keep"] is not None]
        pts.sort(key=lambda r: r["mean_effective_keep"])
        if not pts:
            continue
        color, marker = METHOD_STYLE.get(method, ("#17becf", "o"))
        ax.plot([p["mean_effective_keep"] for p in pts],
                [p["accuracy"] for p in pts],
                marker=marker, ls="-" if len(pts) > 1 else "",
                color=color, lw=1.8, ms=7, label=method)
    if baseline:
        ax.axhline(baseline["accuracy"], ls="--", color="black", lw=1,
                   label=f"baseline ({baseline['accuracy']:.3f})")
        ax.plot([1.0], [baseline["accuracy"]], marker="*", ms=14,
                color="black")
    ax.set_xlabel("mean effective keep ratio (retained / original "
                  "visual tokens)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{benchmark.upper()}: accuracy vs effective retention\n"
                 f"{model}")
    ax.set_xlim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot_path = root / "accuracy_vs_effective_retention.png"
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")

    # ---------------- merged table ----------------
    def sort_key(r):
        return (r["method"] != "baseline", r["method"],
                -(r["mean_effective_keep"] or 1.0))

    rows.sort(key=sort_key)

    def fmt(v, spec=".3f", default="—"):
        return format(v, spec) if v is not None else default

    header = (f"{'method':>12} {'config':>10} {'req keep':>9} "
              f"{'eff keep':>9} {'orig tok':>9} {'retained':>9} "
              f"{'accuracy':>9} {'drop':>8}")
    lines = [
        f"{benchmark.upper()} method sweep | {model}",
        "(keep ratio = fraction of visual tokens RETAINED; effective keep "
        "from token_pruning_metadata)",
        "", header, "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{r['method']:>12} {r['label']:>10} "
            f"{r['requested_keep']:>9.2f} "
            f"{fmt(r['mean_effective_keep']):>9} "
            f"{fmt(r['mean_original_tokens'], '.1f'):>9} "
            f"{fmt(r['mean_retained_tokens'], '.1f'):>9} "
            f"{r['accuracy']:>9.3f} "
            f"{fmt(r['accuracy_drop'], '+.3f'):>8}")
    table = "\n".join(lines)

    with open(root / "retention_summary.txt", "w") as f:
        f.write(table + "\n")
    with open(root / "retention_summary.json", "w") as f:
        json.dump({
            "benchmark": benchmark,
            "model": model,
            "baseline_accuracy": baseline["accuracy"] if baseline else None,
            "rows": [{k: v for k, v in r.items()} for r in rows],
        }, f, indent=2)
    print(f"wrote {root / 'retention_summary.txt'}, "
          f"{root / 'retention_summary.json'}")
    print("\n" + table)


if __name__ == "__main__":
    main()
