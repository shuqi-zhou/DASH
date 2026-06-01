"""
step3_HMM_test_metric.py
------------------------
HMM fitting and best-parameter selection for DASH FRET data.

Reads root / processed_dir paths from config.yaml.
HMM grid-search parameters are fixed below (edit directly if needed).

Usage
-----
# List available fret pkl files:
    python step3_HMM_test_metric.py --list

# Process a single file by name:
    python step3_HMM_test_metric.py --file fret_test

# Process ALL fret pkl files (batch mode):
    python step3_HMM_test_metric.py --batch

# Default (no flags): processes all data as single file
    python step3_HMM_test_metric.py

Saves per-file:
  - HMM checkpoints  → <root>/output/ckpt_hmm/saved_hmm_models/<dataset_name>/
  - Metrics CSV       → <root>/output/result/step3_HMM_<dataset_name>.csv
  - Predictions pkl   → <root>/output/result/step3_HMM_final_predictions_<dataset_name>.pkl
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
# Modified 2026-05-31: set BLAS thread limits before NumPy/SciPy import to avoid OpenBLAS crashes.
import sys
import argparse
import pickle
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score

import torch

from config_loader import load_config
from utils.utils import preprocess_fret_cross, _merge_close_states_by_estimated_mean
from utils.utils_fret_analyse import StateAnalyzer_Test
from utils.utils_hmm import fit_hmm
from utils.utils_full_plot import get_abbreviation, pattern_to_df, count_unique_states, count_num_transition, plot_hmm_summary

# ---------------------------------------------------------------------------
#  GPU device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Logging Setup
# ---------------------------------------------------------------------------
def setup_logging(root_path, log_file=None):
    """Setup logging to both file and console."""
    if log_file is None:
        log_dir = Path(root_path) / "output" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"step3_hmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Force re-configuration by removing existing handlers from the root logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(str(log_file), encoding='utf-8')
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(fh)
    root_logger.addHandler(sh)

    _logger = logging.getLogger(__name__)
    _logger.info(f"Log file: {log_file}")
    return _logger

logger = None  # Will be initialized in main()

# ---------------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------------
cfg      = load_config()
root     = cfg["root"]
raw_dir  = cfg["preprocess"]["raw_dir"]
proc_dir = cfg["preprocess"]["processed_dir"]
# Modified 2026-05-31: allow Step 3 comparison outputs to stay under /user/project/DASH/0531compare.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "DASH_data"

# HMM grid-search parameters — edit directly to change the search space.
# n_mix : number of Gaussian mixture components per hidden state (emission model).
#         1  → standard Gaussian HMM (fast, less flexible)
#         2  → default: captures mild asymmetry in each FRET state
#         3+ → richer emission model but slower and more prone to overfitting
N_MIX_VALUES            = [2]          # extend to [1, 2] or [2, 3] to grid-search
# Modified 2026-05-31: match ver1 step3 notebook search space.
NSTATES_VALUES          = [2, 3, 4, 5] # number of FRET states to try
GLOBAL_THRESHOLD_VALUES = [0.3, 0.2, 0.1]
LOCAL_THRESHOLD_VALUES  = [0.15, 0.12]


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def viterbi_path_consistency(hidden_states, length_info):
    """Measures how consistent the Viterbi path is (fewer state changes is better)."""
    start = 0
    scores = []
    for length in length_info:
        end = start + length
        trace = hidden_states[start:end]
        transitions = np.sum(np.diff(trace) != 0)
        scores.append(1.0 - (transitions / (length - 1)) if length > 1 else 1.0)
        start = end
    return np.mean(scores)


def calculate_merged_bic(fret_values, merged_states):
    unique_states = np.unique(merged_states)
    n_states      = len(unique_states)
    n_samples     = len(fret_values)

    log_likelihood = 0
    for state in unique_states:
        state_data = fret_values[merged_states == state]
        if len(state_data) > 1:
            mu, std = np.mean(state_data), np.std(state_data)
            if std == 0:
                std = 1e-6
            log_likelihood += np.sum(stats.norm.logpdf(state_data, mu, std))

    k   = 2 * n_states + n_states * (n_states - 1)
    bic = -2 * log_likelihood + k * np.log(n_samples)
    return max(0, 1 - (bic / 1_000_000))


def calculate_silhouette(fret_values, merged_states):
    try:
        if len(np.unique(merged_states)) > 1:
            # Fix: ensure fret_values is 2D for silhouette_score
            X = fret_values.reshape(-1, 1) if fret_values.ndim == 1 else fret_values
            sil = silhouette_score(X, merged_states,
                                   sample_size=10000, random_state=42)
        else:
            sil = 0.0
        return (sil + 1) / 2
    except Exception as e:
        print(f"Error calculating silhouette: {e}")
        return 0.5


def state_separation_clarity(fret_values, hidden_states):
    unique_states = np.unique(hidden_states)
    state_means = [np.mean(fret_values[hidden_states == state]) for state in unique_states]
    state_stds  = [np.std(fret_values[hidden_states == state]) for state in unique_states]

    if len(state_means) < 2:
        return 0.0

    sorted_idx   = np.argsort(state_means)
    sorted_means = np.array(state_means)[sorted_idx]
    sorted_stds  = np.array(state_stds)[sorted_idx]
    separations  = []
    for i in range(len(sorted_means) - 1):
        mean_diff    = sorted_means[i + 1] - sorted_means[i]
        combined_std = np.sqrt(sorted_stds[i] ** 2 + sorted_stds[i + 1] ** 2)
        separations.append(mean_diff / combined_std if combined_std > 0 else 0.0)
    return float(np.mean(separations))


def dwell_time_score(hidden_states, length_info):
    start = 0
    dwell_times = []
    for length in length_info:
        end   = start + length
        trace = hidden_states[start:end]
        runs  = np.diff(np.where(np.diff(np.append([-1], trace)) != 0)[0])
        dwell_times.extend(runs)
        start = end
    if len(dwell_times) < 5:
        return 0.0
    dwell_times = np.array(dwell_times)
    cv = np.std(dwell_times) / np.mean(dwell_times) if np.mean(dwell_times) > 0 else 0
    return 1.0 - min(abs(cv - 1.0), 1.0)


def physical_realism_score(hidden_states, fret_values):
    unique_states = np.unique(hidden_states)
    state_means = [np.mean(fret_values[hidden_states == state]) for state in unique_states]

    if len(state_means) < 2:
        return 0.5

    sorted_means     = sorted(state_means)
    coverage         = (max(sorted_means) - min(sorted_means)) / 1.4
    min_separation   = min(np.diff(sorted_means)) if len(sorted_means) > 1 else 1.0
    separation_score = min(min_separation / 0.1, 1.0)
    edge_count       = sum(1 for m in state_means if m < -0.1 or m > 1.1)
    edge_ratio       = edge_count / len(state_means)
    edge_penalty     = 1.0 - min(edge_ratio, 0.5) * 2
    return coverage * 0.4 + separation_score * 0.4 + edge_penalty * 0.2


def transition_matrix_entropy(hmm_model, normalized=True):
    trans_mat = hmm_model.transmat_
    row_entropies = []
    for row in trans_mat:
        row = row + 1e-10
        row = row / np.sum(row)
        entropy = -np.sum(row * np.log2(row))
        max_entropy = np.log2(len(row))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        row_entropies.append(normalized_entropy if normalized else entropy)

    mean_entropy = np.mean(row_entropies)
    if normalized:
        if mean_entropy < 0.3:
            score = mean_entropy / 0.3
        elif mean_entropy > 0.7:
            score = 1.0 - (mean_entropy - 0.7) / 0.3
        else:
            score = 1.0
    else:
        score = 1.0 - abs(mean_entropy - 1.0)

    return max(0, score)


# ---------------------------------------------------------------------------
#  Core fitting
# ---------------------------------------------------------------------------

def test_evaluation_metrics(fret_np, raw_all, length_info,
                            load_saved_models=False, save_models=True,
                            dataset_name=None, models_base_dir=None,
                            data_dir=None):
    """Fit HMM models across parameter grid, score each, return (df, results_all)."""
    models_dir = os.path.join(str(models_base_dir), dataset_name)
    data_path  = data_dir if data_dir else str(proc_dir / dataset_name) + "/"
    if save_models:
        os.makedirs(models_dir, exist_ok=True)

    saved_models = {}
    if load_saved_models:
        try:
            for n_mix in N_MIX_VALUES:
                for nstates in NSTATES_VALUES:
                    model_file = os.path.join(models_dir,
                                              f"hmm_model_mix{n_mix}_states{nstates}.pkl")
                    if os.path.exists(model_file):
                        with open(model_file, "rb") as f:
                            saved_models[(n_mix, nstates)] = pickle.load(f)
        except Exception as e:
            print(f"Warning: could not load saved models: {e}")
            saved_models = {}

    if saved_models:
        print(f"  Loaded {len(saved_models)} cached HMM model(s).")

    # Pre-compute segment boundaries
    segment_boundaries = []
    start = 0
    for length in length_info:
        segment_boundaries.append((start, start + length))
        start += length

    analyzer = StateAnalyzer_Test(
        root_dir=str(root) + "/",
        data_dir=data_path,
        save_dir="result/",
        verbose=False,
        assign_state="kmeans",
        hmm_init_method="curve_fit",
        max_states=6,
        handle_short_dwells=False,
    )

    fret_reshaped = fret_np.reshape(-1, 1)
    results = []

    total_combos = len(N_MIX_VALUES) * len(NSTATES_VALUES)
    fitted = 0
    for n_mix in N_MIX_VALUES:
        for nstates in NSTATES_VALUES:
            fitted += 1
            if (n_mix, nstates) in saved_models:
                hmm_vit = saved_models[(n_mix, nstates)]
            else:
                print(f"  Fitting HMM [{fitted}/{total_combos}]: "
                      f"n_mix={n_mix}, nstates={nstates} ...", end=" ", flush=True)
                hmm_vit = fit_hmm(fret_np, n_mix=n_mix, nstates=nstates,
                                  algorithm="viterbi", covariance_type="diag",
                                  use_length=False)
                print("done")
                if save_models:
                    model_file = os.path.join(models_dir,
                                              f"hmm_model_mix{n_mix}_states{nstates}.pkl")
                    with open(model_file, "wb") as f:
                        pickle.dump(hmm_vit, f)

            transition_score = transition_matrix_entropy(hmm_vit)
            hidden_states    = hmm_vit.predict(fret_reshaped, lengths=length_info)

            for global_threshold in GLOBAL_THRESHOLD_VALUES:
                merged_predictions, merged_means, _ = analyzer.merge_states(
                    hidden_states, fret_np, threshold=global_threshold
                )

                for local_threshold in LOCAL_THRESHOLD_VALUES:
                    final_predictions_all = []
                    for idx, (seg_start, seg_end) in enumerate(segment_boundaries):
                        fret_segment = fret_np[seg_start:seg_end]
                        pred         = merged_predictions[seg_start:seg_end].copy()

                        # Fix: build per-segment estimated means for ALL states, not just those present
                        pred_update_estimated_mean = merged_means.copy()
                        for state in range(len(merged_means)):
                            mask = (pred == state)
                            if np.any(mask):
                                pred_update_estimated_mean[state] = np.mean(fret_segment[mask])

                        pred = analyzer._handle_short_dwells(
                            pred, fret_segment, merged_means, min_last_frame=2)

                        if len(pred) == 0:
                            continue

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            # Fix: pass pred_update_estimated_mean (per-segment means) as the
                            # estimated_means argument, not merged_means (global means)
                            seg_merge_map, _ = _merge_close_states_by_estimated_mean(
                                pred, merged_means, pred_update_estimated_mean,
                                fret_segment, threshold=local_threshold
                            )
                        final_predictions_all.extend([seg_merge_map[p] for p in pred])

                    if len(final_predictions_all) == 0:
                        continue

                    final_predictions_all = np.array(final_predictions_all)

                    consistency_score = viterbi_path_consistency(final_predictions_all, length_info)
                    separation_score  = state_separation_clarity(fret_np, final_predictions_all)
                    dwell_score       = dwell_time_score(final_predictions_all, length_info)
                    realism_score     = physical_realism_score(final_predictions_all, fret_np)
                    bic_score         = calculate_merged_bic(fret_np, final_predictions_all)
                    sil               = calculate_silhouette(fret_reshaped, final_predictions_all)

                    final_score = (
                        1 / 6 * bic_score +
                        1 / 6 * sil +
                        1 / 6 * consistency_score +
                        1 / 6 * separation_score +
                        1 / 6 * realism_score +
                        1 / 6 * transition_score
                    )

                    results.append({
                        "n_mix": n_mix, "nstates": nstates,
                        "global_threshold": global_threshold,
                        "local_threshold": local_threshold,
                        "final_score": final_score,
                        "bic_score": bic_score, "silhouette": sil,
                        "consistency": consistency_score,
                        "separation": separation_score,
                        "dwell_time": dwell_score,
                        "realism": realism_score,
                        "transition_entropy": transition_score,
                        "num_states_after_merging": len(np.unique(final_predictions_all)),
                        "hmm_model": hmm_vit,
                        "merged_predictions": final_predictions_all,
                        "merged_means": merged_means,
                    })

    df = pd.DataFrame([
        {k: v for k, v in m.items()
         if k not in ("hmm_model", "merged_predictions", "merged_means")}
        for m in results
    ])
    return df, results


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_raw_files():
    """Return (all_traces, cumulative_file_lengths) from raw .npz files."""
    def _load_npz_traces(npz_dir):
        """Load traces from .npz files."""
        npz_files = sorted([f for f in os.listdir(str(npz_dir)) if f.endswith('.npz')])
        all_traces = []
        for npz_file in npz_files:
            npz_path = os.path.join(str(npz_dir), npz_file)
            data = np.load(npz_path, allow_pickle=True)
            traces = data['traces']  # shape: (N, T, 2)
            all_traces.extend(traces)
        return all_traces

    npz_files = [f for f in os.listdir(str(raw_dir)) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {raw_dir}. MAT input is no longer supported."
        )

    all_traces = _load_npz_traces(raw_dir)
    # Each .npz file is treated as one "file" for file_split
    file_lengths = [0]
    for npz_file in sorted(npz_files):
        npz_path = os.path.join(str(raw_dir), npz_file)
        data = np.load(npz_path, allow_pickle=True)
        n_traces = data['traces'].shape[0]
        file_lengths.append(n_traces)
    return all_traces, np.cumsum(file_lengths)


# ---------------------------------------------------------------------------
#  Per-file pipeline
# ---------------------------------------------------------------------------

def _process_one_file(file_idx, data_type, segs_preds, data_all,
                      file_split, result_dir, ckpt_hmm_dir, fret_dir,
                      fret_file_name=None, use_cached_models=False):
    """Run the full HMM pipeline for a single file index and save outputs."""

    if fret_file_name:
        data_set_name = fret_file_name.replace('_fret.pkl', '')
    else:
        data_set_name = f"{data_type}_{file_idx}"

    n_traces = file_split[file_idx][1] - file_split[file_idx][0]
    print(f"\n{'='*60}")
    print(f"Dataset : {data_set_name}  ({n_traces} traces)")
    print('='*60)

    # ── Build FRET / raw arrays ───────────────────────────────────────────────
    fret_all = []
    raw_all  = []

    for i in range(file_split[file_idx][0], file_split[file_idx][1]):
        trace      = data_all[i][0]
        begin_dict = segs_preds[i][0]
        end_dict   = segs_preds[i][1]

        labeled_segments = []
        if begin_dict.get("1", []) and end_dict.get("3", []):
            for b, e in zip(begin_dict["1"], end_dict["3"]):
                labeled_segments.append((b, e, 1))
        if begin_dict.get("4", []) and end_dict.get("6", []):
            for b, e in zip(begin_dict["4"], end_dict["6"]):
                labeled_segments.append((b, e, 2))

        for begin, end, dash_label in labeled_segments:
            if end - begin < 3:
                continue
            segment = trace[begin:end]
            fret_data, raw_data = preprocess_fret_cross(segment)
            if len(fret_data) > 0:
                fret_all.append(fret_data)
                raw_all.append(raw_data)

    if not fret_all:
        print(f"  No valid FRET segments found, skipping.")
        return

    length_info = [len(f) for f in fret_all]
    fret_np     = np.concatenate(fret_all)
    print(f"  {len(fret_all)} FRET segments  |  {len(fret_np)} frames total")

    segment_boundaries = []
    start = 0
    for length in length_info:
        segment_boundaries.append((start, start + length))
        start += length

    # HMM grid search
    print(f"  Running HMM grid search ...")
    df, results_all = test_evaluation_metrics(
        fret_np, raw_all, length_info,
        load_saved_models=use_cached_models, save_models=True,
        dataset_name=data_set_name,
        models_base_dir=ckpt_hmm_dir,
        data_dir=str(fret_dir) + "/",
    )

    csv_path = result_dir / f"step3_HMM_{data_set_name}.csv"
    df.sort_values("final_score", ascending=False).to_csv(str(csv_path), index=False)

    best_result        = max(results_all, key=lambda x: x["final_score"])
    merged_predictions = best_result["merged_predictions"]
    merged_means       = best_result["merged_means"]

    print(f"  Best → n_mix={best_result['n_mix']}, nstates={best_result['nstates']}, "
          f"g_thresh={best_result['global_threshold']}, "
          f"l_thresh={best_result['local_threshold']}, "
          f"score={best_result['final_score']:.4f}, "
          f"states={best_result['num_states_after_merging']}")

    # Build pattern record and prediction list
    pattern_record, pattern_seg = defaultdict(int), defaultdict(list)
    update_predictions_all = []

    for idx, (seg_start, seg_end) in enumerate(segment_boundaries):
        fret = fret_np[seg_start:seg_end]
        raw  = raw_all[idx]
        pred = merged_predictions[seg_start:seg_end]

        abbreviation = get_abbreviation(pred)
        if len(abbreviation) > 1:
            transition_tuple = tuple(frozenset(zip(abbreviation[:-1], abbreviation[1:])))
        else:
            transition_tuple = (abbreviation[0],)

        pattern_record[transition_tuple] += 1
        pattern_seg[transition_tuple].append(idx)
        update_predictions_all.append({
            "predictions": pred,
            "trace_label": data_set_name,
            "fret_data":   fret,
            "full_mean":   merged_means,
            "raw_data":    raw,
        })

    pred_pkl = result_dir / f"step3_HMM_final_predictions_{data_set_name}.pkl"
    with open(str(pred_pkl), "wb") as f:
        pickle.dump(update_predictions_all, f)

    print(f"  Saved → {csv_path.name}  |  {pred_pkl.name}")

    # Top transition patterns
    df_pattern = pattern_to_df(pattern_record)
    df_pattern["num_state"]     = df_pattern["pattern"].map(count_unique_states)
    df_pattern["num_transition"] = df_pattern["pattern"].map(count_num_transition)
    df_pattern = df_pattern[df_pattern["count"] > 1].sort_values("count", ascending=False)
    if not df_pattern.empty:
        print("\n  Top transition patterns:")
        print(df_pattern.head(10).to_string(index=False))

    # Visualize
    plot_hmm_summary(
        update_predictions_all, merged_means,
        data_set_name, result_dir,
        max_traces=6,
    )


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def list_fret_files(fret_dir):
    """List available *_fret.pkl files with trace counts."""
    fret_files = sorted([f for f in os.listdir(str(fret_dir)) if f.endswith('_fret.pkl')])
    if not fret_files:
        data_type = os.path.basename(str(fret_dir).rstrip('/\\'))
        old_path = fret_dir / data_type / "data_fret.pkl"
        if old_path.exists():
            fret_files = [old_path.name]

    if not fret_files:
        print(f"No *_fret.pkl files found in {fret_dir}")
        return []

    print(f"\nAvailable *_fret.pkl files in {fret_dir}:")
    print("-" * 60)
    file_info = []
    for idx, fret_file in enumerate(fret_files):
        fret_path = fret_dir / fret_file
        try:
            with open(str(fret_path), 'rb') as f:
                data = pickle.load(f)
            # Fix: remove the erroneous second pickle.load; just unwrap tuple if needed
            if isinstance(data, tuple):
                data = data[0]
            n_traces = len(data)
            print(f"  [{idx}] {fret_file} ({n_traces} traces)")
            file_info.append((idx, fret_file, n_traces))
        except Exception as e:
            print(f"  [{idx}] {fret_file} (error: {e})")
            file_info.append((idx, fret_file, 0))
    print("-" * 60)
    return file_info


def get_fret_file_index(fret_dir, name):
    """Get file index by name (partial or full match)."""
    fret_files = sorted([f for f in os.listdir(str(fret_dir)) if f.endswith('_fret.pkl')])
    for idx, fret_file in enumerate(fret_files):
        if fret_file == name or fret_file == f"{name}.pkl":
            return idx

    # Try partial match (e.g., "fret_test" matches "fret_test.pkl")
    for idx, fret_file in enumerate(fret_files):
        base_name = fret_file.replace('_fret.pkl', '')
        if base_name == name or fret_file.startswith(name):
            return idx

    return None


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    global logger
    parser = argparse.ArgumentParser(
        description="DASH Step 3: HMM fitting and parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list              # List available fret pkl files
  %(prog)s --file fret_test   # Process a single file by name
  %(prog)s --batch            # Process all fret pkl files
        """
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list",  action="store_true",
                       help="List available *_fret.pkl files.")
    group.add_argument("--file",  type=str, default=None, metavar="NAME",
                       help="Process a single file by name (e.g., fret_test).")
    group.add_argument("--batch", action="store_true",
                       help="Process ALL fret pkl files in the data directory.")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "0531compare"),
                        help="Directory for Step 3 HMM comparison outputs.")
    parser.add_argument("--fret_dir", type=str, default=None,
                        help="Override directory containing *_fret.pkl files.")
    parser.add_argument("--pred_seg_dir", type=str, default=None,
                        help="Override directory containing Step 2 prediction pkl files.")
    parser.add_argument("--use_cached_models", action="store_true",
                        help="Reuse saved HMM models instead of refitting them.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    # Modified 2026-05-31: save HMM results/checkpoints/logs inside comparison folder.
    result_dir   = output_dir / "hmm_result"
    ckpt_hmm_dir = output_dir / "ckpt_hmm" / "saved_hmm_models"
    log_dir = output_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_hmm_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(root), log_file=log_dir / f"step3_hmm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f"Step 3 HMM started  (device={DEVICE})")

    raw_dir_str = str(cfg["preprocess"]["raw_dir"])
    data_type   = os.path.basename(raw_dir_str.rstrip('/\\'))

    fret_dir     = Path(args.fret_dir) if args.fret_dir else Path(cfg.get("hmm", {}).get("fret_dir", ""))
    pred_seg_dir = Path(args.pred_seg_dir) if args.pred_seg_dir else Path(cfg.get("hmm", {}).get("pred_seg_dir", ""))
    if not fret_dir.exists():
        fallback_fret_dir = LOCAL_DATA_ROOT / "step0_preprocess" / "test"
        if fallback_fret_dir.exists():
            print(f"Configured HMM fret_dir not found: {fret_dir}")
            print(f"Using local fret_dir: {fallback_fret_dir}")
            fret_dir = fallback_fret_dir
    if not pred_seg_dir.exists():
        fallback_pred_seg_dir = output_dir
        if fallback_pred_seg_dir.exists():
            print(f"Configured HMM pred_seg_dir not found: {pred_seg_dir}")
            print(f"Using comparison pred_seg_dir: {fallback_pred_seg_dir}")
            pred_seg_dir = fallback_pred_seg_dir

    if not fret_dir.exists():
        print(f"Error: HMM fret directory not found: {fret_dir}")
        sys.exit(1)
    if not pred_seg_dir.exists():
        print(f"Error: HMM pred_seg directory not found: {pred_seg_dir}")
        sys.exit(1)

    # Discover fret pkl files
    fret_files = sorted([f for f in os.listdir(str(fret_dir)) if f.endswith('_fret.pkl')])
    if not fret_files:
        data_type_dir = os.path.basename(str(fret_dir).rstrip('/\\'))
        old_path = fret_dir / data_type_dir / "data_fret.pkl"
        if old_path.exists():
            fret_files = [old_path.name]
            fret_dir   = fret_dir / data_type_dir

    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Data directory (fret_dir):  {fret_dir}")
    print(f"  Dataset name (data_type):   {data_type}")
    print(f"  Available fret files:       {len(fret_files)}")
    print(f"  Result directory:           {result_dir}")
    print(f"{'='*60}\n")

    # --list mode (early return)
    if args.list:
        list_fret_files(fret_dir)
        print("\nUse --file NAME to process a specific file, or --batch to process all.")
        return

    # ── Determine which fret files to process ────────────────────────────────
    if args.batch:
        selected_files = fret_files
        print(f"Batch mode: processing all {len(selected_files)} fret file(s).")
    elif args.file is not None:
        file_idx_found = get_fret_file_index(fret_dir, args.file)
        if file_idx_found is None:
            print(f"Error: No fret file matching '{args.file}' found.")
            print("Use --list to see available files.")
            sys.exit(1)
        selected_files = [fret_files[file_idx_found]]
        print(f"Processing file: {selected_files[0]} (index {file_idx_found})")
    else:
        # No CLI args: fall back to config
        config_mode      = cfg.get("hmm", {}).get("mode", "batch")
        config_file_name = cfg.get("hmm", {}).get("file_name", "")

        if config_mode == "list":
            list_fret_files(fret_dir)
            print("\nUse --file NAME to process a specific file, or --batch to process all.")
            return
        elif config_mode == "single" and config_file_name:
            file_idx_found = get_fret_file_index(fret_dir, config_file_name)
            if file_idx_found is None:
                print(f"Error: config file_name '{config_file_name}' not found.")
                print("Use --list to see available files.")
                sys.exit(1)
            selected_files = [fret_files[file_idx_found]]
            print(f"Using config: mode={config_mode}, file_name={config_file_name}")
            print(f"Processing file: {selected_files[0]} (index {file_idx_found})")
        else:
            selected_files = fret_files
            print(f"Using config: mode={config_mode}, processing all {len(selected_files)} fret file(s).")

    # ── Match each fret file with its pred_seg file ───────────────────────────
    file_info_list = []
    for fret_file in selected_files:
        base_name     = fret_file.replace('_fret.pkl', '')
        pred_seg_file = None
        for pf in os.listdir(str(pred_seg_dir)):
            if pf.startswith(base_name) and pf.endswith('.pkl'):
                pred_seg_file = pred_seg_dir / pf
                break
        if pred_seg_file is None:
            print(f"Warning: No pred_seg file found for {fret_file}, skipping.")
            continue
        file_info_list.append((fret_file, pred_seg_file))

    if not file_info_list:
        print("Error: No valid file pairs found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Files to process: {len(file_info_list)}")
    for ff, ps in file_info_list:
        print(f"  fret: {ff}  ←→  pred_seg: {ps.name}")
    print(f"{'='*60}\n")

    # ── Process each file ─────────────────────────────────────────────────────
    for idx, (fret_file, pred_seg_file) in enumerate(file_info_list):
        logger.info(f"[{idx+1}/{len(file_info_list)}] Processing: {fret_file}")
        fret_path = fret_dir / fret_file
        with open(str(fret_path), 'rb') as f:
            data_all = pickle.load(f)
            if isinstance(data_all, tuple):
                data_all = data_all[0]

        with open(str(pred_seg_file), 'rb') as f:
            segs_preds = pickle.load(f)

        file_split = [(0, len(data_all))]

        print(f"Processing file {idx+1}/{len(file_info_list)}: {fret_file}")
        _process_one_file(
            0,
            data_type, segs_preds, data_all,
            file_split, result_dir, ckpt_hmm_dir, fret_dir,
            fret_file_name=fret_file,
            use_cached_models=args.use_cached_models,
        )
        logger.info(f"[{idx+1}/{len(file_info_list)}] Done: {fret_file}")

    logger.info("All files processed successfully.")
    print("\nAll done.")


if __name__ == "__main__":
    main()
