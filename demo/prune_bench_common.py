"""Shared helpers for the pruning method-sweep benchmarks.

Used by pope_eval.py / mme_eval.py / textvqa_eval.py so every benchmark
speaks the same method/ratio/token-metadata language:

- build_configs(): expands a --method into concrete run configs
  (requested keep ratio + request-body fields). Ratio-controllable
  methods sweep `token_pruning=<keep>`; NPrune sweeps strides (its keep
  count is ceil(H/s)*ceil(W/s), not round(N*ratio)); Checkered has one
  fixed ~50% pattern; baseline sends no pruning fields at all.
- parse_token_metadata(): robust extraction of original / pruned /
  retained visual-token counts from `token_pruning_metadata`.
- token_aggregates(): per-config mean/median retained-token stats.
- apply_manifest(): pin the exact sample IDs shared by every method in
  a sweep (written once, reused by all runs).

Terminology: `token_pruning` is a KEEP (retention) ratio -- 0.14 keeps
14% of the visual tokens.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RATIO_METHODS = ("hiprune", "hydart", "hiprune_pp", "dart", "anchorprune")
PATTERN_METHODS = ("nprune", "checkered")
ALL_METHODS = ("baseline",) + RATIO_METHODS + PATTERN_METHODS

DEFAULT_RATIOS = [0.75, 0.50, 0.25, 0.14]
# Achievable lattice strides. The deployed vLLM fork currently validates
# stride ∈ {1, 2} (stride 1 = no-op; stride 2 ≈ 25% keep). Wider strides
# (3, 4) are rejected with HTTP 400 until the fork's allow-list expands.
DEFAULT_STRIDES = [2]


def build_configs(method: str, ratios: list[float],
                  strides: list[int]) -> list[dict]:
    """Expand one method into run configs.

    Each config: {label, requested_keep, stride, body_fields} where
    body_fields are merged into the chat-completion request body.
    """
    if method == "baseline":
        return [{"label": "baseline", "requested_keep": 1.0,
                 "stride": None, "body_fields": {}}]

    if method == "nprune":
        # Stride is the real knob; the ratio field only activates the
        # pipeline, so send the nominal 1/s^2 keep for bookkeeping.
        return [{
            "label": f"stride{s}",
            "requested_keep": round(1.0 / (s * s), 4),
            "stride": s,
            "body_fields": {
                "token_pruning": 1.0 / (s * s),
                "token_pruning_method": "nprune",
                "token_pruning_params": {"stride": s},
            },
        } for s in strides]

    if method == "checkered":
        return [{
            "label": "checkered",
            "requested_keep": 0.5,
            "stride": None,
            "body_fields": {
                "token_pruning": 0.5,
                "token_pruning_method": "checkered",
            },
        }]

    if method not in RATIO_METHODS:
        raise ValueError(f"unknown method {method!r}")

    configs = []
    for r in ratios:
        if r >= 1.0:
            # A 1.0 entry in --ratios degrades to a no-pruning row.
            configs.append({"label": "baseline", "requested_keep": 1.0,
                            "stride": None, "body_fields": {}})
        else:
            configs.append({
                "label": f"{r:.2f}",
                "requested_keep": r,
                "stride": None,
                "body_fields": {
                    "token_pruning": r,
                    "token_pruning_method": method,
                },
            })
    return configs


_EMPTY_MD = {
    "original_tokens": None,
    "pruned_tokens": None,
    "retained_tokens": None,
    "effective_keep": None,
    "metadata_missing": True,
}


def parse_token_metadata(resp: dict) -> dict:
    """Extract visual-token counts from `token_pruning_metadata`.

    Null-safe; sums across entries for multi-image requests (these
    benchmarks send one image, so it is normally a single entry).
    Invalid metadata (pruned count exceeding num_tokens, duplicate
    indices are deduplicated) is treated as missing rather than
    silently miscounted.
    """
    entries = [m for m in (resp.get("token_pruning_metadata") or []) if m]
    if not entries:
        return dict(_EMPTY_MD)

    original = 0
    pruned = 0
    for md in entries:
        n = md.get("num_tokens")
        pruned_idx = md.get("pruned")
        if not isinstance(n, int) or n <= 0 or not isinstance(pruned_idx, list):
            return dict(_EMPTY_MD)
        p = len(set(pruned_idx))
        if p > n:
            return dict(_EMPTY_MD)
        original += n
        pruned += p

    retained = original - pruned
    return {
        "original_tokens": original,
        "pruned_tokens": pruned,
        "retained_tokens": retained,
        "effective_keep": retained / original,
        "metadata_missing": False,
    }


def token_aggregates(records: list[dict], is_baseline: bool) -> dict:
    """Aggregate per-request token counts into summary stats."""
    retained = [r["retained_tokens"] for r in records
                if r.get("retained_tokens") is not None]
    original = [r["original_tokens"] for r in records
                if r.get("original_tokens") is not None]
    effective = [r["effective_keep"] for r in records
                 if r.get("effective_keep") is not None]
    missing = sum(bool(r.get("metadata_missing")) for r in records)

    agg = {
        "mean_original_tokens": statistics.mean(original) if original else None,
        "mean_retained_tokens": statistics.mean(retained) if retained else None,
        "median_retained_tokens": (statistics.median(retained)
                                   if retained else None),
        "min_retained_tokens": min(retained) if retained else None,
        "max_retained_tokens": max(retained) if retained else None,
        "mean_effective_keep": statistics.mean(effective) if effective else None,
        "metadata_missing": missing,
    }
    if is_baseline and agg["mean_effective_keep"] is None:
        # Baseline sends no pruning fields, so there is no metadata;
        # by definition every visual token is retained.
        agg["mean_effective_keep"] = 1.0
    return agg


def apply_manifest(samples: list[dict], manifest_path: str | Path,
                   benchmark: str, seed: int) -> list[dict]:
    """Pin the sample set to a shared manifest.

    If the manifest exists, reorder/filter `samples` to exactly its
    `sample_ids` (fails loudly on missing IDs so two methods can never
    silently see different data). Otherwise write the manifest from the
    freshly sampled IDs so subsequent runs reuse it.
    """
    path = Path(manifest_path)
    if path.exists():
        with open(path) as f:
            manifest = json.load(f)
        by_id = {s["id"]: s for s in samples}
        missing = [i for i in manifest["sample_ids"] if i not in by_id]
        if missing:
            raise SystemExit(
                f"manifest {path} has {len(missing)} sample IDs not present "
                f"in the freshly loaded dataset subset (first: {missing[:3]}). "
                "Re-run with the same --num-samples/--seed used to create it, "
                "or delete the manifest.")
        print(f"using sample manifest {path} "
              f"({len(manifest['sample_ids'])} samples)")
        return [by_id[i] for i in manifest["sample_ids"]]

    manifest: dict = {
        "benchmark": benchmark,
        "seed": seed,
        "num_samples": len(samples),
        "sample_ids": [s["id"] for s in samples],
    }
    subsets: dict[str, list[str]] = {}
    for s in samples:
        if "category" in s:
            subsets.setdefault(s["category"], []).append(s["id"])
    if subsets:
        manifest["subsets"] = subsets
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote sample manifest {path} ({len(samples)} samples)")
    return samples


def config_row_fields(method: str, cfg: dict) -> dict:
    """The identity fields shared by every jsonl record of one config."""
    fields = {
        "method": method,
        "requested_keep": cfg["requested_keep"],
        # legacy key kept so older tooling keyed on "ratio" still works
        "ratio": cfg["requested_keep"],
    }
    if cfg["stride"] is not None:
        fields["stride"] = cfg["stride"]
    return fields
