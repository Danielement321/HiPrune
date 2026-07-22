"""Plot the POPE accuracy-vs-speed results produced by pope_eval.py.

Reads pope_summary.json and writes two charts next to it:

- pope_accuracy.png — accuracy and F1 vs retention ratio, with the
  unpruned baseline as a reference line and per-category accuracy as
  light dashed lines
- pope_tradeoff.png — accuracy vs mean TTFT, one point per ratio: the
  "what accuracy do you give up for the speed" chart

Usage:
    python3 plot_pope.py <pope_summary.json>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    summary_path = Path(sys.argv[1])
    with open(summary_path) as f:
        data = json.load(f)
    out_dir = summary_path.parent
    results = sorted(data["results"], key=lambda s: s["ratio"])
    ratios = [s["ratio"] for s in results]
    labels = ["baseline" if r >= 1.0 else f"{r:.2f}" for r in ratios]
    accuracy = [s["accuracy"] for s in results]
    f1 = [s["f1"] for s in results]
    ttft_ms = [s["mean_ttft_s"] * 1e3 for s in results]
    baseline = next(s for s in results if s["ratio"] >= 1.0)

    # --- accuracy vs retention -------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ratios, accuracy, "o-", color="#1a70c4", lw=2, label="accuracy")
    ax.plot(ratios, f1, "s--", color="#7a4dc4", lw=1.5, label="F1")
    ax.axhline(baseline["accuracy"], color="#888888", ls=":", lw=1,
               label=f"baseline accuracy ({baseline['accuracy']:.3f})")
    categories = sorted(baseline["per_category_accuracy"])
    for cat in categories:
        ax.plot(ratios, [s["per_category_accuracy"][cat] for s in results],
                "--", lw=0.8, alpha=0.5, label=f"{cat} accuracy")
    ax.set_xlabel("token retention ratio (1.0 = no pruning)")
    ax.set_ylabel("score")
    ax.set_title(f"POPE accuracy vs HiPrune retention\n"
                 f"{data['model']}, {data['num_samples']} samples")
    ax.set_xticks(ratios)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    acc_path = out_dir / "pope_accuracy.png"
    fig.savefig(acc_path, dpi=150)
    print(f"wrote {acc_path}")

    # --- accuracy vs TTFT -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ttft_ms, accuracy, "-", color="#cccccc", lw=1, zorder=1)
    ax.scatter(ttft_ms, accuracy, s=70, color="#1a70c4", zorder=2)
    for x, y, label in zip(ttft_ms, accuracy, labels):
        ax.annotate(f"  {label}", (x, y), fontsize=9, va="bottom")
    ax.set_xlabel("mean TTFT (ms, serial streaming, cache-busted)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"POPE accuracy vs prefill latency\n"
                 f"{data['model']}, {data['num_samples']} samples")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    trade_path = out_dir / "pope_tradeoff.png"
    fig.savefig(trade_path, dpi=150)
    print(f"wrote {trade_path}")


if __name__ == "__main__":
    main()
