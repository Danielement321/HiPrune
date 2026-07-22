"""Comparison plots + paired stats for MME and TextVQA (HiPrune vs HyDART).

Consumes the output dirs of mme_eval.py / textvqa_eval.py for both
methods and produces, into --out-dir:

- mme_methods.png       — accuracy / perception / cognition vs retention
- textvqa_methods.png   — VQA soft accuracy vs retention
- mme_categories.png    — per-category MME score deltas (HyDART - HiPrune)
- benchmark_summary.txt — tables + McNemar / paired stats

McNemar (exact binomial on discordant pairs) is used for MME's yes/no
questions; a paired bootstrap on per-question soft accuracy is used for
TextVQA.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_records(path: Path) -> dict[tuple[float, str], dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["ratio"], r["id"], r.get("question", ""))] = r
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided binomial tail, p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(diffs: list[float], iters: int = 10000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Returns (mean diff, 2.5th pct, 97.5th pct)."""
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    return (sum(diffs) / n, means[int(0.025 * iters)],
            means[int(0.975 * iters)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mme-hydart", required=True)
    parser.add_argument("--mme-hiprune", required=True)
    parser.add_argument("--textvqa-hydart", required=True)
    parser.add_argument("--textvqa-hiprune", required=True)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mme_hy = load_summary(Path(args.mme_hydart) / "mme_summary.json")
    mme_hp = load_summary(Path(args.mme_hiprune) / "mme_summary.json")
    tv_hy = load_summary(Path(args.textvqa_hydart) / "textvqa_summary.json")
    tv_hp = load_summary(Path(args.textvqa_hiprune) / "textvqa_summary.json")

    def by_ratio(summary):
        return {round(s["ratio"], 4): s for s in summary["results"]}

    mme_hy_r, mme_hp_r = by_ratio(mme_hy), by_ratio(mme_hp)
    tv_hy_r, tv_hp_r = by_ratio(tv_hy), by_ratio(tv_hp)

    baseline_mme = mme_hy_r.get(1.0) or mme_hp_r.get(1.0)
    baseline_tv = tv_hy_r.get(1.0) or tv_hp_r.get(1.0)
    ratios = sorted(r for r in mme_hy_r if r < 1.0)

    hy_color, hp_color = "#1f77b4", "#d62728"

    # ---------------- MME figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("accuracy", "Accuracy", 1.0),
        ("perception_score", "Perception score (max 2000)", 2000),
        ("cognition_score", "Cognition score (max 800)", 800),
    ]
    for ax, (key, title, _) in zip(axes, panels):
        ax.plot(ratios, [mme_hp_r[r][key] for r in ratios], "o-",
                color=hp_color, label="HiPrune")
        ax.plot(ratios, [mme_hy_r[r][key] for r in ratios], "o-",
                color=hy_color, label="HyDART")
        ax.axhline(baseline_mme[key], ls="--", color="gray",
                   label=f"baseline ({baseline_mme[key]:.3g})")
        ax.set_xlabel("token retention ratio")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"MME (full, 2374 questions) — {mme_hy['model']}")
    fig.tight_layout()
    fig.savefig(out / "mme_methods.png", dpi=150)
    plt.close(fig)

    # ---------------- TextVQA figure ----------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ratios, [tv_hp_r[r]["vqa_accuracy"] for r in ratios], "o-",
            color=hp_color, label="HiPrune")
    ax.plot(ratios, [tv_hy_r[r]["vqa_accuracy"] for r in ratios], "o-",
            color=hy_color, label="HyDART")
    ax.axhline(baseline_tv["vqa_accuracy"], ls="--", color="gray",
               label=f"baseline ({baseline_tv['vqa_accuracy']:.3f})")
    ax.set_xlabel("token retention ratio")
    ax.set_ylabel("VQA soft accuracy")
    ax.set_title(f"TextVQA ({tv_hy['num_samples']} val samples) — "
                 f"{tv_hy['model']}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "textvqa_methods.png", dpi=150)
    plt.close(fig)

    # ---------------- MME per-category deltas ----------------
    fig, axes = plt.subplots(1, len(ratios), figsize=(5.2 * len(ratios), 5),
                             sharey=True)
    if len(ratios) == 1:
        axes = [axes]
    for ax, r in zip(axes, ratios):
        cats = sorted(mme_hy_r[r]["per_category"])
        deltas = [mme_hy_r[r]["per_category"][c]["score"]
                  - mme_hp_r[r]["per_category"][c]["score"] for c in cats]
        colors = [hy_color if d >= 0 else hp_color for d in deltas]
        ax.barh(cats, deltas, color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"retention {r:.2f}")
        ax.set_xlabel("score delta (HyDART − HiPrune)")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("MME per-category score: HyDART − HiPrune "
                 "(blue = HyDART better)")
    fig.tight_layout()
    fig.savefig(out / "mme_categories.png", dpi=150)
    plt.close(fig)

    # ---------------- Paired stats ----------------
    mme_hy_recs = load_records(Path(args.mme_hydart) / "mme_results.jsonl")
    mme_hp_recs = load_records(Path(args.mme_hiprune) / "mme_results.jsonl")
    tv_hy_recs = load_records(Path(args.textvqa_hydart)
                              / "textvqa_results.jsonl")
    tv_hp_recs = load_records(Path(args.textvqa_hiprune)
                              / "textvqa_results.jsonl")

    lines = []
    lines.append(f"MME (full, {mme_hy['num_samples']} questions) | "
                 f"{mme_hy['model']}")
    lines.append("")
    header = (f"{'retention':>10} | {'HiPrune acc':>11} {'HyDART acc':>11} | "
              f"{'HiPrune P':>10} {'HyDART P':>9} | "
              f"{'HiPrune C':>10} {'HyDART C':>9} | "
              f"{'McNemar p':>10}")
    lines += [header, "-" * len(header)]
    base = baseline_mme
    lines.append(f"{'baseline':>10} | {base['accuracy']:>11.3f} "
                 f"{'—':>11} | {base['perception_score']:>10.1f} {'—':>9} | "
                 f"{base['cognition_score']:>10.1f} {'—':>9} | {'—':>10}")
    for r in ratios:
        keys = [k for k in mme_hy_recs if k[0] == r]
        b = c = 0
        for k in keys:
            if k not in mme_hp_recs:
                continue
            hy_ok = mme_hy_recs[k]["correct"]
            hp_ok = mme_hp_recs[k]["correct"]
            if hp_ok and not hy_ok:
                b += 1
            elif hy_ok and not hp_ok:
                c += 1
        p = mcnemar_exact(b, c)
        hy, hp = mme_hy_r[r], mme_hp_r[r]
        lines.append(
            f"{r:>10.2f} | {hp['accuracy']:>11.3f} {hy['accuracy']:>11.3f} | "
            f"{hp['perception_score']:>10.1f} {hy['perception_score']:>9.1f} | "
            f"{hp['cognition_score']:>10.1f} {hy['cognition_score']:>9.1f} | "
            f"{p:>10.4f}")
        lines.append(f"{'':>10}   discordant: HiPrune-only correct {b}, "
                     f"HyDART-only correct {c}")

    lines += ["", "", f"TextVQA ({tv_hy['num_samples']} val samples, "
              f"seed {tv_hy['seed']}) | {tv_hy['model']}", ""]
    header = (f"{'retention':>10} | {'HiPrune':>8} {'HyDART':>8} | "
              f"{'mean diff':>10} {'95% CI (bootstrap)':>20}")
    lines += [header, "-" * len(header)]
    lines.append(f"{'baseline':>10} | {baseline_tv['vqa_accuracy']:>8.3f} "
                 f"{'—':>8} | {'—':>10} {'—':>20}")
    for r in ratios:
        keys = [k for k in tv_hy_recs if k[0] == r]
        diffs = [tv_hy_recs[k]["vqa_acc"] - tv_hp_recs[k]["vqa_acc"]
                 for k in keys if k in tv_hp_recs]
        mean, lo, hi = paired_bootstrap(diffs)
        hy, hp = tv_hy_r[r], tv_hp_r[r]
        lines.append(
            f"{r:>10.2f} | {hp['vqa_accuracy']:>8.3f} "
            f"{hy['vqa_accuracy']:>8.3f} | {mean:>+10.4f} "
            f"[{lo:>+.4f}, {hi:>+.4f}]")

    text = "\n".join(lines)
    with open(out / "benchmark_summary.txt", "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote plots + benchmark_summary.txt to {out}/")


if __name__ == "__main__":
    main()
