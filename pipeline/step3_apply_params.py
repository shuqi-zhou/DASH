"""
step3_apply_params.py
---------------------
Apply HMM parameters from the first row of each postprocess z-score CSV.

Reads (n_mix, nstates, global_threshold, local_threshold) from
postprocess/<dataset>/step3_HMM_postprocess_<dataset>_zscore.csv, re-runs
predict + merge, and outputs:
  1. Final prediction PKL (same format as step3)
  2. Summary PNG (FRET histogram per state + overall)
  3. Transition rate matrix CSV (segment-aware, post-processed)
  4. Emission matrix CSV (per-state FRET mean/std/occupancy)
  5. Kinetics PKL (full parameter bundle)

Segment source:
  - label: build segments from rec[2] in each *_fret.pkl; pred_seg_dir is not used.
  - step2: load segments from Step 2 prediction PKL; pred_seg_dir is required.
If --seg_source is omitted, hmm.seg_source from config.yaml is used.

Usage
-----
# Batch mode for unlabeled data (step0/step2 group directories are inferred)
python pipeline/step3_apply_params.py \
    --base_dir output/extra_test_unlabeled/step3 \
    --seg_source step2

# Labeled data: use ground-truth labels; pred_seg_dir is unnecessary
python pipeline/step3_apply_params.py \
    --output_dir output/train_labeled/step3 \
    --fret_dir output/train_labeled/step0 \
    --seg_source label

# Single unlabeled group: use Step 2 predictions
python pipeline/step3_apply_params.py \
    --output_dir output/extra_test_unlabeled/step3/b2AR_Gs \
    --pred_seg_dir output/extra_test_unlabeled/step2/b2AR_Gs \
    --seg_source step2
"""

import os
import sys
import argparse
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils import _merge_close_states_by_estimated_mean
from utils.utils_fret_analyse import StateAnalyzer_Test
from step3_postprocess import (
    extract_fret_segments, load_cached_models, run_merge,
    _build_analyzer_kwargs, _find_pred_seg_file, STATE_COLORS,
)
from step3_HMM_test_metric import seg_info_from_labels

# ---------------------------------------------------------------------------
#  Segment-aware transition counting
# ---------------------------------------------------------------------------

def get_transition_counts_segmented(predictions, length_info):
    """Count transitions within each segment, avoiding cross-segment boundaries."""
    unique_states = np.unique(predictions)
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n = len(unique_states)
    counts = np.zeros((n, n), dtype=int)

    start = 0
    for length in length_info:
        seg = predictions[start:start + length]
        for t in range(len(seg) - 1):
            counts[state_to_idx[seg[t]], state_to_idx[seg[t + 1]]] += 1
        start += length

    return counts, unique_states


# ---------------------------------------------------------------------------
#  Summary histogram (per-state + overall, no Gaussian fit)
# ---------------------------------------------------------------------------

def plot_fret_summary(update_predictions_all, dataset_name, save_dir):
    """Plot per-state and overall FRET histograms with transparency."""
    all_fret = np.concatenate([d["fret_data"] for d in update_predictions_all])
    all_pred = np.concatenate([d["predictions"] for d in update_predictions_all])
    states = np.unique(all_pred)
    n_states = len(states)

    fig, axes = plt.subplots(1, n_states + 1,
                             figsize=(3 * (n_states + 1), 3),
                             squeeze=False)
    axes = axes[0]
    bins = np.linspace(-0.1, 1.1, 60)
    bin_width = bins[1] - bins[0]
    n_total = len(all_fret)

    # Overall
    ax = axes[0]
    ax.hist(all_fret, bins=bins, color="lightgray", alpha=0.6, density=True)
    for si, state in enumerate(states):
        color = STATE_COLORS[si % len(STATE_COLORS)]
        mask = all_pred == state
        data = all_fret[mask]
        w = np.ones(len(data)) / (n_total * bin_width)
        ax.hist(data, bins=bins, weights=w, color=color, alpha=0.35,
                label=f"State {si}")
    ax.set_title("Overall", fontsize=9)
    ax.set_xlabel("FRET")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6)

    # Per-state
    for si, state in enumerate(states):
        ax = axes[si + 1]
        color = STATE_COLORS[si % len(STATE_COLORS)]
        mask = all_pred == state
        data = all_fret[mask]
        ax.hist(all_fret, bins=bins, color="lightgray", alpha=0.4, density=True)
        w = np.ones(len(data)) / (n_total * bin_width)
        ax.hist(data, bins=bins, weights=w, color=color, alpha=0.6)
        mu, std = np.mean(data), np.std(data)
        frac = np.sum(mask) / len(all_pred)
        ax.set_title(f"State {si}: μ={mu:.3f} σ={std:.3f}\n"
                     f"({frac:.1%}, {np.sum(mask)} frames)", fontsize=8)
        ax.set_xlabel("FRET")

    fig.suptitle(f"{dataset_name} — FRET Distribution by State", fontsize=11,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(save_dir) / f"step3_HMM_summary_{dataset_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
#  Z-score CSV scanning and parameter extraction
# ---------------------------------------------------------------------------

def scan_zscore_csvs(output_dir):
    """Read the first row of each dataset's postprocess z-score CSV."""
    output_dir = Path(output_dir)
    postprocess_dir = output_dir / "postprocess"
    if not postprocess_dir.exists():
        return [], []

    tasks = []
    errors = []
    param_cols = ["n_mix", "nstates", "global_threshold", "local_threshold"]

    for dataset_dir in sorted(p for p in postprocess_dir.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        csv_path = dataset_dir / f"step3_HMM_postprocess_{dataset_name}.csv"
        if not csv_path.exists():
            errors.append((dataset_name, f"file not found: {csv_path}"))
            continue

        df = pd.read_csv(csv_path)
        missing = [col for col in param_cols if col not in df.columns]
        if missing:
            errors.append((dataset_name, f"missing columns {missing}: {csv_path}"))
            continue
        if df.empty:
            errors.append((dataset_name, f"empty CSV: {csv_path}"))
            continue

        first_row = df.iloc[0]
        params = first_row[param_cols].to_dict()
        params["n_mix"] = int(params["n_mix"])
        params["nstates"] = int(params["nstates"])
        params["final_score"] = (
            float(first_row["final_score"])
            if "final_score" in df.columns and pd.notna(first_row["final_score"])
            else None
        )
        tasks.append((dataset_name, params, csv_path))

    return tasks, errors


def resolve_output_dirs(args):
    """Determine list of output_dir paths from --base_dir or --output_dir."""
    if args.output_dir:
        return [Path(args.output_dir).resolve()]

    base = Path(args.base_dir).resolve()
    # base_dir itself has postprocess/?
    if (base / "postprocess").exists():
        return [base]

    # Scan subdirectories
    dirs = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir() and (sub / "postprocess").exists():
            dirs.append(sub)
    return dirs


def resolve_fret_pred_dirs(output_dir, args, seg_source):
    """Resolve fret_dir and pred_seg_dir for a given output_dir.

    If CLI args specify them, use those directly.
    Otherwise try to infer from parallel step0/step2 directories,
    then fall back to config.yaml.
    """
    group_name = output_dir.name  # e.g. "ribo"

    # CLI explicit
    if args.fret_dir:
        fret_dir = Path(args.fret_dir).resolve()
    else:
        # Try parallel: step3/<group> → step0/<group>
        step3_parent = output_dir.parent  # e.g. .../step3
        inferred = step3_parent.parent / "step0" / group_name
        if inferred.exists():
            fret_dir = inferred
        else:
            fret_dir = _config_fret_dir()

    if seg_source == "step2":
        if args.pred_seg_dir:
            pred_seg_dir = Path(args.pred_seg_dir).resolve()
        else:
            step3_parent = output_dir.parent
            inferred = step3_parent.parent / "step2" / group_name
            if inferred.exists():
                pred_seg_dir = inferred
            else:
                pred_seg_dir = _config_pred_seg_dir()
    else:
        pred_seg_dir = None

    return fret_dir, pred_seg_dir


def _config_fret_dir():
    from config_loader import load_config
    cfg = load_config()
    return Path(cfg.get("hmm", {}).get("fret_dir", ""))


def _config_pred_seg_dir():
    from config_loader import load_config
    cfg = load_config()
    return Path(cfg.get("hmm", {}).get("pred_seg_dir", ""))


def _config_seg_source():
    from config_loader import load_config
    cfg = load_config()
    return str(cfg.get("hmm", {}).get("seg_source", "step2")).lower()


# ---------------------------------------------------------------------------
#  Core: process one dataset
# ---------------------------------------------------------------------------

def process_one_dataset(output_dir, dataset_name, params, fret_dir, pred_seg_dir,
                        seg_source="step2"):
    """Process a single dataset with selected parameters.

    Returns (success: bool, summary_row: dict or None).
    """
    output_dir = Path(output_dir)
    n_mix = params["n_mix"]
    nstates = params["nstates"]
    g_thresh = params["global_threshold"]
    l_thresh = params["local_threshold"]

    # --- Load FRET data ---
    fret_path = Path(fret_dir) / f"{dataset_name}_fret.pkl"
    if not fret_path.exists():
        print(f"  ERROR: FRET file not found: {fret_path}")
        return False, None

    with open(fret_path, "rb") as f:
        data_all = pickle.load(f)
    if isinstance(data_all, tuple):
        data_all = data_all[0]

    if seg_source == "label":
        segs_preds = [seg_info_from_labels(rec[2]) for rec in data_all]
    elif seg_source == "step2":
        pred_seg_path = _find_pred_seg_file(pred_seg_dir, dataset_name)
        if pred_seg_path is None:
            print(f"  ERROR: pred_seg file not found for {dataset_name} in {pred_seg_dir}")
            return False, None
        with open(pred_seg_path, "rb") as f:
            segs_preds = pickle.load(f)
    else:
        raise ValueError(f"Unsupported seg_source: {seg_source}")

    fret_np, length_info, segment_boundaries, raw_all, segment_metadata = extract_fret_segments(
        data_all, segs_preds, return_metadata=True
    )
    if len(fret_np) == 0:
        print(f"  ERROR: No valid FRET segments for {dataset_name}")
        return False, None

    print(f"  Segment source: {seg_source}")
    print(f"  {len(length_info)} segments | {len(fret_np)} frames")

    # --- Load cached HMM model ---
    models = load_cached_models(output_dir, dataset_name,
                                filter_n_mix=n_mix, filter_nstates=nstates)
    if (n_mix, nstates) not in models:
        print(f"  ERROR: HMM model mix{n_mix}_states{nstates} not found for {dataset_name}")
        return False, None

    hmm_model = models[(n_mix, nstates)]
    analyzer_kwargs = _build_analyzer_kwargs(output_dir)

    # --- Predict + merge ---
    final_pred, merged_means, raw_hidden = run_merge(
        hmm_model, fret_np, length_info, segment_boundaries,
        g_thresh, l_thresh, analyzer_kwargs
    )

    if len(final_pred) == 0:
        print(f"  WARNING: run_merge returned empty predictions for {dataset_name}")
        return False, None

    unique_states = np.unique(final_pred)
    n_states_final = len(unique_states)
    print(f"  Merged → {n_states_final} state(s)")

    # --- Build update_predictions_all (same format as step3) ---
    full_mean = np.zeros(int(max(unique_states)) + 1)
    for s in unique_states:
        full_mean[int(s)] = np.mean(fret_np[final_pred == s])

    update_predictions_all = []
    for idx, (seg_start, seg_end) in enumerate(segment_boundaries):
        seg_fret = fret_np[seg_start:seg_end]
        seg_pred = final_pred[seg_start:seg_end]
        update_predictions_all.append({
            "predictions": seg_pred,
            "trace_label": dataset_name,
            "fret_data": seg_fret,
            "full_mean": full_mean,
            "raw_data": raw_all[idx],
            **segment_metadata[idx],
        })

    # --- Output directory ---
    save_dir = output_dir / "applied_params" / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Save final predictions PKL ---
    pred_pkl_path = save_dir / f"step3_HMM_final_predictions_{dataset_name}.pkl"
    with open(pred_pkl_path, "wb") as f:
        pickle.dump(update_predictions_all, f)
    print(f"    Saved: {pred_pkl_path.name}")

    # --- 2. Summary PNG ---
    png_path = plot_fret_summary(update_predictions_all, dataset_name, save_dir)
    print(f"    Saved: {png_path.name}")

    # --- 3. Transition matrix CSV (segment-aware) ---
    trans_counts, trans_unique = get_transition_counts_segmented(final_pred, length_info)
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_prob = trans_counts / np.maximum(row_sums, 1)

    state_labels = [f"state_{i}" for i in range(len(trans_unique))]
    trans_df = pd.DataFrame(trans_prob, index=state_labels, columns=state_labels)
    trans_csv_path = save_dir / f"step3_HMM_transmat_{dataset_name}.csv"
    trans_df.to_csv(trans_csv_path)
    print(f"    Saved: {trans_csv_path.name}")

    # --- 4. Emission CSV ---
    emission_rows = []
    for si, state in enumerate(unique_states):
        mask = final_pred == state
        emission_rows.append({
            "state": si,
            "fret_mean": float(np.mean(fret_np[mask])),
            "fret_std": float(np.std(fret_np[mask])),
            "occupancy": float(np.sum(mask) / len(final_pred)),
            "n_frames": int(np.sum(mask)),
        })
    emission_df = pd.DataFrame(emission_rows)
    emission_csv_path = save_dir / f"step3_HMM_emission_{dataset_name}.csv"
    emission_df.to_csv(emission_csv_path, index=False)
    print(f"    Saved: {emission_csv_path.name}")

    # --- 5. Kinetics PKL ---
    kinetics = {
        "dataset_name": dataset_name,
        "combo_params": {
            "n_mix": n_mix,
            "nstates": nstates,
            "global_threshold": g_thresh,
            "local_threshold": l_thresh,
            "final_score": params.get("final_score"),
            "num_states_after_merging": n_states_final,
        },
        "post_trans_counts": trans_counts,
        "post_trans_prob": trans_prob,
        "post_unique_states": trans_unique,
        "emission": emission_rows,
        "raw_hmm_transmat": hmm_model.transmat_,
        "raw_hmm_means": hmm_model.means_,
        "raw_hmm_covars": hmm_model.covars_,
        "raw_hmm_weights": hmm_model.weights_,
        "raw_hmm_startprob": hmm_model.startprob_,
    }
    kinetics_pkl_path = save_dir / f"step3_HMM_kinetics_{dataset_name}.pkl"
    with open(kinetics_pkl_path, "wb") as f:
        pickle.dump(kinetics, f)
    print(f"    Saved: {kinetics_pkl_path.name}")

    # --- Build summary row ---
    state_means_str = ", ".join(
        f"{np.mean(fret_np[final_pred == s]):.4f}" for s in unique_states
    )
    summary_row = {
        "output_dir": str(output_dir),
        "dataset": dataset_name,
        "n_mix": n_mix,
        "nstates": nstates,
        "global_threshold": g_thresh,
        "local_threshold": l_thresh,
        "n_states_final": n_states_final,
        "n_segments": len(length_info),
        "n_frames": len(fret_np),
        "state_means": state_means_str,
        "pkl_path": str(pred_pkl_path),
    }

    return True, summary_row


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply HMM parameters from the first row of postprocess z-score CSVs"
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--base_dir", type=str,
                            help="Scan root containing multiple group subdirectories")
    mode_group.add_argument("--output_dir", type=str,
                            help="Single group step3 output directory")

    parser.add_argument("--fret_dir", type=str, default=None,
                        help="FRET data directory (auto-inferred in batch mode)")
    parser.add_argument("--pred_seg_dir", type=str, default=None,
                        help="Step2 prediction directory (auto-inferred in batch mode)")
    parser.add_argument("--seg_source", choices=["label", "step2"], default=None,
                        help="Segment source (default: config hmm.seg_source)")

    args = parser.parse_args()
    seg_source = args.seg_source or _config_seg_source()
    if seg_source not in ("label", "step2"):
        print(f"ERROR: unsupported seg_source: {seg_source}")
        sys.exit(1)
    print(f"Segment source: {seg_source}")

    # --- Resolve output_dir list ---
    output_dirs = resolve_output_dirs(args)
    if not output_dirs:
        print("ERROR: No valid output directories found.")
        sys.exit(1)

    # --- Phase 1: Scan all postprocess z-score CSVs ---
    all_tasks = []  # (output_dir, dataset_name, params)
    all_errors = []

    for output_dir in output_dirs:
        tasks, errors = scan_zscore_csvs(output_dir)
        for dataset_name, params, csv_path in tasks:
            all_tasks.append((output_dir, dataset_name, params))
        all_errors.extend(
            (output_dir, dataset_name, reason)
            for dataset_name, reason in errors
        )

    # Print discovery summary
    if all_tasks:
        print(f"\nFound {len(all_tasks)} dataset(s) with selected params:")
        for output_dir, dataset_name, params in all_tasks:
            group = output_dir.name
            print(f"  {group}/{dataset_name}: "
                  f"mix{params['n_mix']}_st{params['nstates']} "
                  f"g={params['global_threshold']} l={params['local_threshold']}")
    else:
        print("\nNo valid z-score parameter CSV files found.")

    if all_errors:
        print(f"\nParameter CSV errors ({len(all_errors)}):")
        for output_dir, dataset_name, reason in all_errors:
            print(f"  {output_dir.name}/{dataset_name}: {reason}")

    if not all_tasks:
        sys.exit(1)

    # --- Phase 2: Process each dataset ---
    summary_rows = []
    any_failed = bool(all_errors)

    for output_dir, dataset_name, params in all_tasks:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {output_dir.name}/{dataset_name}")
        print(f"  Params: mix{params['n_mix']}_st{params['nstates']} "
              f"g={params['global_threshold']} l={params['local_threshold']}")
        print("=" * 60)

        fret_dir, pred_seg_dir = resolve_fret_pred_dirs(
            output_dir, args, seg_source
        )
        if not fret_dir.exists():
            print(f"  ERROR: fret_dir not found: {fret_dir}")
            any_failed = True
            continue
        if seg_source == "step2" and not pred_seg_dir.exists():
            print(f"  ERROR: pred_seg_dir not found: {pred_seg_dir}")
            any_failed = True
            continue

        ok, row = process_one_dataset(
            output_dir, dataset_name, params, fret_dir, pred_seg_dir,
            seg_source=seg_source,
        )
        if ok and row:
            summary_rows.append(row)
        elif not ok:
            any_failed = True

    # --- Phase 3: Write summary.csv ---
    if summary_rows:
        # Write per-group summary.csv under each output_dir's applied_params/
        rows_by_dir = {}
        for row in summary_rows:
            d = row["output_dir"]
            rows_by_dir.setdefault(d, []).append(row)

        for d, rows in rows_by_dir.items():
            summary_path = Path(d) / "applied_params" / "summary.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print(f"\nSummary: {summary_path}")

    if any_failed:
        print("\nDone with errors.")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
