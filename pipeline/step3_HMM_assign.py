"""
step3_HMM_assign.py
---------------------
HMM fitting, best-parameter selection, and final Step 3 outputs for DASH FRET data.

Reads root / processed_dir paths from config.yaml.
HMM grid-search parameters are fixed below (edit directly if needed).

Usage
-----
# List available fret pkl files:
    python step3_HMM_assign.py --list

# Process a single file by name:
    python step3_HMM_assign.py --file fret_test

# Process ALL fret pkl files (batch mode):
    python step3_HMM_assign.py --batch

# Default (no flags): processes all data as single file
    python step3_HMM_assign.py

Saves outputs:
  - HMM checkpoints  → <output_dir>/ckpt_hmm/saved_hmm_models/<dataset_name>/
  - Final outputs    → <output_dir>/hmm_result/<dataset_name>/
  - Summary CSV      → <output_dir>/hmm_result/summary.csv
"""

import os
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "4"

import sys
import argparse
import pickle
import logging
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from utils.utils_full_plot import get_abbreviation
from step3_apply_params import get_transition_counts_segmented, plot_fret_summary

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "DASH_data"

# HMM grid-search parameters — edit directly to change the search space.
# n_mix : number of Gaussian mixture components per hidden state (emission model).
#         1  → standard Gaussian HMM (fast, less flexible)
#         2  → default: captures mild asymmetry in each FRET state
#         3+ → richer emission model but slower and more prone to overfitting
N_MIX_VALUES            = [1]
# Modified 2026-05-31: match ver1 step3 notebook search space.
NSTATES_VALUES          = [2,3,4,5]
GLOBAL_THRESHOLD_VALUES = [0.05,0.08,0.1,0.12,0.15]
LOCAL_THRESHOLD_VALUES  = [0.05,0.08,0.1,0.12,0.15]
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Token ids: 1/4 = donor/acceptor bleach begin, 3/6 = donor/acceptor bleach end.
SEG_START_TOKENS = (1, 4)
SEG_END_TOKENS   = (3, 6)


def seg_info_from_labels(label_seq):
    """Build (start_dic, end_dic) from a per-frame token sequence.

    Mirrors step2's get_seq_info_batch for one sequence, so ground-truth labels
    can substitute for step2 predictions when hmm.seg_source == "label".
    start_dic keys: '1', '4';  end_dic keys: '3', '6'.
    """
    start_dic = {str(t): [] for t in SEG_START_TOKENS}
    end_dic   = {str(t): [] for t in SEG_END_TOKENS}
    for i, val in enumerate(label_seq):
        v = int(val)
        if v in SEG_START_TOKENS:
            start_dic[str(v)].append(i)
        if v in SEG_END_TOKENS:
            end_dic[str(v)].append(i)
    return start_dic, end_dic


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

def _score_hmm_combo(n_mix, nstates, hmm_vit, fret_np, length_info,
                     segment_boundaries, analyzer_kwargs,
                     global_thresholds, local_thresholds):
    """Score one fitted HMM across all threshold combinations."""
    analyzer = StateAnalyzer_Test(**analyzer_kwargs)
    fret_reshaped = fret_np.reshape(-1, 1)
    transition_score = transition_matrix_entropy(hmm_vit)
    hidden_states = hmm_vit.predict(fret_reshaped, lengths=length_info)
    results = []

    # Raw score on direct HMM predictions (no threshold merging)
    raw_consistency = viterbi_path_consistency(hidden_states, length_info)
    raw_separation = state_separation_clarity(fret_np, hidden_states)
    raw_realism = physical_realism_score(hidden_states, fret_np)
    raw_bic = calculate_merged_bic(fret_np, hidden_states)
    raw_sil = calculate_silhouette(fret_reshaped, hidden_states)
    raw_final_score = (
        raw_bic + raw_sil + raw_consistency +
        raw_separation + raw_realism + transition_score
    ) / 6
    raw_result = {
        "n_mix": n_mix, "nstates": nstates,
        "final_score": raw_final_score,
        "bic_score": raw_bic, "silhouette": raw_sil,
        "consistency": raw_consistency,
        "separation": raw_separation,
        "realism": raw_realism,
        "transition_entropy": transition_score,
        "num_states": len(np.unique(hidden_states)),
    }

    for global_threshold in global_thresholds:
        merged_predictions, merged_means, _ = analyzer.merge_states(
            hidden_states, fret_np, threshold=global_threshold
        )

        for local_threshold in local_thresholds:
            final_predictions_all = []
            for seg_start, seg_end in segment_boundaries:
                fret_segment = fret_np[seg_start:seg_end]
                pred = merged_predictions[seg_start:seg_end].copy()

                # Build per-segment means for all states, including absent states.
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
                    seg_merge_map, _ = _merge_close_states_by_estimated_mean(
                        pred, merged_means, pred_update_estimated_mean,
                        fret_segment, threshold=local_threshold
                    )
                final_predictions_all.extend([seg_merge_map[p] for p in pred])

            if len(final_predictions_all) == 0:
                continue

            final_predictions_all = np.array(final_predictions_all)

            consistency_score = viterbi_path_consistency(final_predictions_all, length_info)
            separation_score = state_separation_clarity(fret_np, final_predictions_all)
            realism_score = physical_realism_score(final_predictions_all, fret_np)
            bic_score = calculate_merged_bic(fret_np, final_predictions_all)
            sil = calculate_silhouette(fret_reshaped, final_predictions_all)

            final_score = (
                bic_score + sil + consistency_score +
                separation_score + realism_score + transition_score
            ) / 6

            results.append({
                "n_mix": n_mix, "nstates": nstates,
                "global_threshold": global_threshold,
                "local_threshold": local_threshold,
                "final_score": final_score,
                "bic_score": bic_score, "silhouette": sil,
                "consistency": consistency_score,
                "separation": separation_score,
                "realism": realism_score,
                "transition_entropy": transition_score,
                "num_states_after_merging": len(np.unique(final_predictions_all)),
                "hmm_model": hmm_vit,
                "merged_predictions": final_predictions_all,
                "merged_means": merged_means,
            })

    return results, raw_result


def _fit_and_score_one_combo(args):
    """Fit one (n_mix, nstates) combo and score all threshold combinations."""
    (n_mix, nstates, fret_np, length_info,
     segment_boundaries, analyzer_kwargs,
     global_thresholds, local_thresholds,
     models_dir, save_models) = args

    hmm_vit = fit_hmm(fret_np, n_mix=n_mix, nstates=nstates,
                      algorithm="viterbi", covariance_type="diag",
                      use_length=False, length_info=length_info)

    if save_models:
        os.makedirs(models_dir, exist_ok=True)
        model_file = os.path.join(models_dir,
                                  f"hmm_model_mix{n_mix}_states{nstates}.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(hmm_vit, f)

    results, raw_result = _score_hmm_combo(
        n_mix, nstates, hmm_vit, fret_np, length_info,
        segment_boundaries, analyzer_kwargs,
        global_thresholds, local_thresholds,
    )
    return n_mix, nstates, results, raw_result


def test_evaluation_metrics(fret_np, raw_all, length_info,
                            load_saved_models=False, save_models=True,
                            dataset_name=None, models_base_dir=None,
                            data_dir=None,
                            max_workers=None, blas_threads=4,
                            no_parallel=False):
    """Fit HMM models across parameter grid, score each, return (df, results_all)."""
    if blas_threads < 1:
        raise ValueError("blas_threads must be >= 1")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1")

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

    analyzer_kwargs = dict(
        root_dir=str(root) + "/",
        data_dir=data_path,
        save_dir="result/",
        verbose=False,
        assign_state="kmeans",
        hmm_init_method="curve_fit",
        max_states=6,
        handle_short_dwells=False,
    )

    results = []
    raw_results = []
    all_combos = [(n_mix, nstates)
                  for n_mix in N_MIX_VALUES
                  for nstates in NSTATES_VALUES]
    combos_to_fit = [combo for combo in all_combos if combo not in saved_models]
    combos_cached = [combo for combo in all_combos if combo in saved_models]

    worker_count = max_workers
    if worker_count is None:
        n_cpu = os.cpu_count() or 4
        worker_count = min(len(combos_to_fit), max(1, n_cpu // blas_threads))
    else:
        worker_count = min(len(combos_to_fit), worker_count)

    use_parallel = (not no_parallel and len(combos_to_fit) > 1 and worker_count > 1)
    task_args_list = [
        (
            n_mix, nstates, fret_np, length_info,
            segment_boundaries, analyzer_kwargs,
            GLOBAL_THRESHOLD_VALUES, LOCAL_THRESHOLD_VALUES,
            models_dir, save_models,
        )
        for n_mix, nstates in combos_to_fit
    ]

    if use_parallel:
        saved_env = {var: os.environ.get(var) for var in _THREAD_VARS}
        for var in _THREAD_VARS:
            os.environ[var] = str(blas_threads)
        try:
            print(f"  Parallel HMM fitting: {len(combos_to_fit)} combos, "
                  f"{worker_count} workers, {blas_threads} BLAS threads/worker")
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_fit_and_score_one_combo, args): args[:2]
                    for args in task_args_list
                }
                for future in as_completed(futures):
                    n_mix, nstates = futures[future]
                    _, _, combo_results, combo_raw = future.result()
                    results.extend(combo_results)
                    raw_results.append(combo_raw)
                    print(f"    Done: n_mix={n_mix}, nstates={nstates} "
                          f"({len(combo_results)} scores)")
        finally:
            for var in _THREAD_VARS:
                if saved_env[var] is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = saved_env[var]
    else:
        total_combos = len(all_combos)
        combo_positions = {combo: idx + 1 for idx, combo in enumerate(all_combos)}
        for args in task_args_list:
            n_mix, nstates = args[:2]
            print(f"  Fitting HMM [{combo_positions[(n_mix, nstates)]}/{total_combos}]: "
                  f"n_mix={n_mix}, nstates={nstates} ...", end=" ", flush=True)
            _, _, combo_results, combo_raw = _fit_and_score_one_combo(args)
            results.extend(combo_results)
            raw_results.append(combo_raw)
            print("done")

    for n_mix, nstates in combos_cached:
        hmm_vit = saved_models[(n_mix, nstates)]
        combo_results, combo_raw = _score_hmm_combo(
            n_mix, nstates, hmm_vit, fret_np, length_info,
            segment_boundaries, analyzer_kwargs,
            GLOBAL_THRESHOLD_VALUES, LOCAL_THRESHOLD_VALUES,
        )
        results.extend(combo_results)
        raw_results.append(combo_raw)

    order_mix = {value: idx for idx, value in enumerate(N_MIX_VALUES)}
    order_states = {value: idx for idx, value in enumerate(NSTATES_VALUES)}
    order_global = {value: idx for idx, value in enumerate(GLOBAL_THRESHOLD_VALUES)}
    order_local = {value: idx for idx, value in enumerate(LOCAL_THRESHOLD_VALUES)}
    results.sort(key=lambda m: (
        order_mix[m["n_mix"]],
        order_states[m["nstates"]],
        order_global[m["global_threshold"]],
        order_local[m["local_threshold"]],
    ))

    df = pd.DataFrame([
        {k: v for k, v in m.items()
         if k not in ("hmm_model", "merged_predictions", "merged_means")}
        for m in results
    ])
    df_raw = pd.DataFrame(raw_results).sort_values("final_score", ascending=False)
    return df, df_raw, results


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
                      fret_file_name=None, use_cached_models=False,
                      max_workers=None, blas_threads=4, no_parallel=False):
    """Run HMM fitting for one file and save apply_params-style outputs."""

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
    segment_metadata = []

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
                segment_metadata.append({
                    "trace_index": int(i),
                    "original_start": int(begin),
                    "original_end": int(end),
                    "segment_label": int(dash_label),
                })

    if not fret_all:
        print(f"  No valid FRET segments found, skipping.")
        return None

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
    _df, _df_raw, results_all = test_evaluation_metrics(
        fret_np, raw_all, length_info,
        load_saved_models=use_cached_models, save_models=True,
        dataset_name=data_set_name,
        models_base_dir=ckpt_hmm_dir,
        data_dir=str(fret_dir) + "/",
        max_workers=max_workers,
        blas_threads=blas_threads,
        no_parallel=no_parallel,
    )
    if not results_all:
        print(f"  No HMM scoring results found, skipping.")
        return None

    best_result        = max(results_all, key=lambda x: x["final_score"])
    merged_predictions = best_result["merged_predictions"]
    hmm_model          = best_result["hmm_model"]

    print(f"  Best → n_mix={best_result['n_mix']}, nstates={best_result['nstates']}, "
          f"g_thresh={best_result['global_threshold']}, "
          f"l_thresh={best_result['local_threshold']}, "
          f"score={best_result['final_score']:.4f}, "
          f"states={best_result['num_states_after_merging']}")

    unique_states = np.unique(merged_predictions)
    n_states_final = len(unique_states)
    full_mean = np.zeros(int(max(unique_states)) + 1)
    for state in unique_states:
        full_mean[int(state)] = np.mean(fret_np[merged_predictions == state])

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

        update_predictions_all.append({
            "predictions": pred,
            "trace_label": data_set_name,
            "fret_data":   fret,
            "full_mean":   full_mean,
            "raw_data":    raw,
            **segment_metadata[idx],
        })

    save_dir = result_dir / data_set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_pkl = save_dir / f"step3_HMM_final_predictions_{data_set_name}.pkl"
    with open(pred_pkl, "wb") as f:
        pickle.dump(update_predictions_all, f)
    print(f"    Saved: {pred_pkl.name}")

    png_path = plot_fret_summary(update_predictions_all, data_set_name, save_dir)
    print(f"    Saved: {png_path.name}")

    trans_counts, trans_unique = get_transition_counts_segmented(
        merged_predictions, length_info
    )
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_prob = trans_counts / np.maximum(row_sums, 1)

    state_labels = [f"state_{i}" for i in range(len(trans_unique))]
    trans_df = pd.DataFrame(trans_prob, index=state_labels, columns=state_labels)
    trans_csv = save_dir / f"step3_HMM_transmat_{data_set_name}.csv"
    trans_df.to_csv(trans_csv)
    print(f"    Saved: {trans_csv.name}")

    emission_rows = []
    for si, state in enumerate(unique_states):
        mask = merged_predictions == state
        emission_rows.append({
            "state": si,
            "fret_mean": float(np.mean(fret_np[mask])),
            "fret_std": float(np.std(fret_np[mask])),
            "occupancy": float(np.sum(mask) / len(merged_predictions)),
            "n_frames": int(np.sum(mask)),
        })
    emission_csv = save_dir / f"step3_HMM_emission_{data_set_name}.csv"
    pd.DataFrame(emission_rows).to_csv(emission_csv, index=False)
    print(f"    Saved: {emission_csv.name}")

    kinetics = {
        "dataset_name": data_set_name,
        "combo_params": {
            "n_mix": best_result["n_mix"],
            "nstates": best_result["nstates"],
            "global_threshold": best_result["global_threshold"],
            "local_threshold": best_result["local_threshold"],
            "final_score": best_result["final_score"],
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
    kinetics_pkl = save_dir / f"step3_HMM_kinetics_{data_set_name}.pkl"
    with open(kinetics_pkl, "wb") as f:
        pickle.dump(kinetics, f)
    print(f"    Saved: {kinetics_pkl.name}")

    state_means_str = ", ".join(
        f"{np.mean(fret_np[merged_predictions == s]):.4f}" for s in unique_states
    )
    return {
        "output_dir": str(result_dir),
        "dataset": data_set_name,
        "n_mix": best_result["n_mix"],
        "nstates": best_result["nstates"],
        "global_threshold": best_result["global_threshold"],
        "local_threshold": best_result["local_threshold"],
        "n_states_final": n_states_final,
        "n_segments": len(length_info),
        "n_frames": len(fret_np),
        "state_means": state_means_str,
        "pkl_path": str(pred_pkl),
    }

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
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "step3"),
                        help="Directory for Step 3 HMM comparison outputs.")
    parser.add_argument("--fret_dir", type=str, default=None,
                        help="Override directory containing *_fret.pkl files.")
    parser.add_argument("--pred_seg_dir", type=str, default=None,
                        help="Override directory containing Step 2 prediction pkl files.")
    parser.add_argument("--seg_source", choices=["label", "step2"], default=None,
                        help="Segment source (default: config hmm.seg_source).")
    parser.add_argument("--use_cached_models", action="store_true",
                        help="Reuse saved HMM models instead of refitting them.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max parallel HMM fitting workers (default: auto).")
    parser.add_argument("--blas_threads", type=int, default=4,
                        help="BLAS threads per worker (default: 4).")
    parser.add_argument("--no_parallel", action="store_true",
                        help="Disable parallel fitting (serial mode).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
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

    # seg_source: "step2" → read step2 pred_seg files; "label" → use ground-truth
    # labels embedded in the fret pkl (labeled/train data; pred_seg_dir ignored).
    seg_source = (args.seg_source or str(
        cfg.get("hmm", {}).get("seg_source", "step2")
    )).lower()

    fret_dir     = Path(args.fret_dir) if args.fret_dir else Path(cfg.get("hmm", {}).get("fret_dir", ""))
    pred_seg_dir = Path(args.pred_seg_dir) if args.pred_seg_dir else Path(cfg.get("hmm", {}).get("pred_seg_dir", ""))
    if not fret_dir.exists():
        fallback_fret_dir = LOCAL_DATA_ROOT / "step0_preprocess" / "test"
        if fallback_fret_dir.exists():
            print(f"Configured HMM fret_dir not found: {fret_dir}")
            print(f"Using local fret_dir: {fallback_fret_dir}")
            fret_dir = fallback_fret_dir
    if seg_source != "label" and not pred_seg_dir.exists():
        fallback_pred_seg_dir = output_dir
        if fallback_pred_seg_dir.exists():
            print(f"Configured HMM pred_seg_dir not found: {pred_seg_dir}")
            print(f"Using comparison pred_seg_dir: {fallback_pred_seg_dir}")
            pred_seg_dir = fallback_pred_seg_dir

    if not fret_dir.exists():
        print(f"Error: HMM fret directory not found: {fret_dir}")
        sys.exit(1)
    if seg_source != "label" and not pred_seg_dir.exists():
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
    print(f"  Segment source:             {seg_source}")
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

    # ── Match each fret file with its segment source ──────────────────────────
    file_info_list = []
    if seg_source == "label":
        # Train mode: segments come from ground-truth labels inside the fret pkl.
        file_info_list = [(fret_file, None) for fret_file in selected_files]
    else:
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
        src = ps.name if ps is not None else "ground-truth labels"
        print(f"  fret: {ff}  ←→  segments: {src}")
    print(f"{'='*60}\n")

    # ── Process each file ─────────────────────────────────────────────────────
    summary_rows = []
    for idx, (fret_file, pred_seg_file) in enumerate(file_info_list):
        logger.info(f"[{idx+1}/{len(file_info_list)}] Processing: {fret_file}")
        fret_path = fret_dir / fret_file
        with open(str(fret_path), 'rb') as f:
            data_all = pickle.load(f)
            if isinstance(data_all, tuple):
                data_all = data_all[0]

        if seg_source == "label":
            segs_preds = [seg_info_from_labels(rec[2]) for rec in data_all]
        else:
            with open(str(pred_seg_file), 'rb') as f:
                segs_preds = pickle.load(f)

        file_split = [(0, len(data_all))]

        print(f"Processing file {idx+1}/{len(file_info_list)}: {fret_file}")
        summary_row = _process_one_file(
            0,
            data_type, segs_preds, data_all,
            file_split, result_dir, ckpt_hmm_dir, fret_dir,
            fret_file_name=fret_file,
            use_cached_models=args.use_cached_models,
            max_workers=args.max_workers,
            blas_threads=args.blas_threads,
            no_parallel=args.no_parallel,
        )
        if summary_row:
            summary_rows.append(summary_row)
        logger.info(f"[{idx+1}/{len(file_info_list)}] Done: {fret_file}")

    if summary_rows:
        summary_path = result_dir / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary: {summary_path}")

    logger.info("All files processed successfully.")
    print("\nAll done.")


if __name__ == "__main__":
    main()
