import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import argparse
import pickle
import json
import glob
import random
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
cfg_early = load_config()
_log_dir = Path(str(cfg_early.get("preprocess", {}).get("log_dir", "output/logs")))
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / f"step0_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Sequential label map: 0=background, 1/2/3=donor bleach B/I/E, 4/5/6=acceptor bleach B/I/E
LABEL_MAP = {
    "5":   0,
    "1_B": 1, "1_I": 2, "1_E": 3,
    "2_B": 4, "2_I": 5, "2_E": 6,
}
VALID_SEG_LABELS = {1, 2}   # only donor-bleach(1) and acceptor-bleach(2) are meaningful
RARE_CLASS_WEIGHT = 1.0     # numerator for inverse-frequency weighting of rare classes


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_sep(input_data):
    """Normalise each trace by its top-5 total intensity (donor + acceptor).

    Traces whose normalisation factor is near zero are skipped with a warning.

    Args:
        input_data: list of (raw_trace, label, meta)  —  raw_trace shape (T, 2)

    Returns:
        data_scaled : list of (trace_norm, label, meta)
        data_fret   : list of (trace_norm, fret_efficiency, label, meta)
    """
    data_scaled, data_fret = [], []
    skipped = 0

    for raw_trace, label, meta in input_data:
        column_sum = raw_trace[:, 0] + raw_trace[:, 1]
        average = float(np.mean(np.sort(column_sum)[-5:]))

        if average <= 1e-10:
            logger.warning(f"Trace '{meta}': normalisation factor ≈ 0 ({average:.2e}), skipping.")
            skipped += 1
            continue

        trace_norm = raw_trace / average

        # Calculate FRET without filtering so trace, FRET, and labels stay aligned.
        # Step 3 applies its FRET range filter after extracting each segment.
        donor = trace_norm[:, 0]
        acceptor = trace_norm[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fret_eff = acceptor / (acceptor + donor)

        data_scaled.append((trace_norm, label, meta))
        data_fret.append((trace_norm, fret_eff, label, meta))

    if skipped:
        logger.warning(f"normalize_sep: skipped {skipped} traces due to near-zero intensity.")
    return data_scaled, data_fret


# ---------------------------------------------------------------------------
# Segment parsing
# ---------------------------------------------------------------------------
def _parse_seg_6col(seg):
    """Parse one row from 6-column annotation format.

    Column layout: [_, _, label, start_1based, end_type2_1based, end_type1_1based]

    Returns (label, seg_start, seg_end) in 0-based indices, or None on failure.
    """
    label     = int(seg[2])
    seg_start = max(0, int(seg[3]) - 1)

    if label == 1:
        raw_end = int(seg[5])
    elif label == 2:
        raw_end = int(seg[4])
    else:
        return None   # unknown label type

    if raw_end <= 0:
        logger.warning(f"6-col segment label={label}: end value is 0, skipping.")
        return None

    seg_end = raw_end - 1
    return label, seg_start, seg_end


def _parse_seg_3col(seg):
    """Parse one row from 3-column annotation format: [label, start, end] (0-based).

    Returns (label, seg_start, seg_end) or None on failure.
    """
    label     = int(seg[0])
    seg_start = max(0, int(seg[1]))
    seg_end   = int(seg[2])
    return label, seg_start, seg_end


def process_trace_seg(seg_info):
    """Extract segment boundaries and labels from a segment annotation array.

    Handles both (N, 6) and (N, 3) formats as well as transposed (6, N).
    Silently skips invalid / overlapping segments and logs warnings.

    Args:
        seg_info: np.ndarray — segment annotation

    Returns:
        y_seg       : list of (start, end) int tuples   (0-based, inclusive)
        y_seg_label : list of int labels  (1 or 2)
    """
    y_seg, y_seg_label = [], []

    if seg_info is None:
        return y_seg, y_seg_label

    seg_info = np.atleast_2d(seg_info)

    # Detect (6, N) transposed format: ncols==6 is ambiguous when nrows==6 too,
    # so we rely on the known column width from _load_npz_traces (always 6 rows).
    if seg_info.shape[0] == 6 and seg_info.shape[1] != 6:
        seg_info = seg_info.T

    if seg_info.shape[0] == 0:
        return y_seg, y_seg_label

    n_cols = seg_info.shape[1]
    prev_end = -1   # track previous segment end to detect overlaps

    for seg in seg_info:
        # --- parse ---
        if n_cols == 6:
            parsed = _parse_seg_6col(seg)
        elif n_cols >= 3:
            parsed = _parse_seg_3col(seg)
        else:
            logger.warning(f"Segment row has only {n_cols} columns, skipping: {seg}")
            continue

        if parsed is None:
            continue

        label, seg_start, seg_end = parsed

        # --- validate label ---
        if label not in VALID_SEG_LABELS:
            logger.debug(f"Skipping segment with label={label} (not in {VALID_SEG_LABELS}).")
            continue

        # --- validate geometry ---
        if seg_end <= seg_start:
            logger.warning(f"Degenerate segment label={label}: start={seg_start} >= end={seg_end}, skipping.")
            continue

        if seg_start <= prev_end:
            logger.warning(
                f"Overlapping segment label={label}: start={seg_start} <= prev_end={prev_end}. "
                f"Clamping start to {prev_end + 1}."
            )
            seg_start = prev_end + 1
            if seg_end <= seg_start:
                continue   # nothing left after clamping

        y_seg_label.append(label)
        y_seg.append((seg_start, seg_end))
        prev_end = seg_end

    return y_seg, y_seg_label


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------
def get_seg_label(length, y_seg, y_seg_label):
    """Convert segment boundaries to per-frame sequential labels.

    Args:
        length      : int — total number of frames
        y_seg       : list of (start, end) tuples (0-based, inclusive)
        y_seg_label : list of segment labels (1 or 2)

    Returns:
        label : list[int] of length `length`, values in LABEL_MAP
    """
    label = [LABEL_MAP["5"]] * length

    if not y_seg:
        return label

    for (seg_start, seg_end), seg_label in zip(y_seg, y_seg_label):
        seg_start, seg_end = int(seg_start), int(seg_end)

        # Clamp to valid range
        if seg_end >= length:
            logger.warning(
                f"seg_end={seg_end} >= trace length={length}; clipping to {length - 1}."
            )
            seg_end = length - 1

        if seg_start >= length or seg_end < seg_start:
            logger.warning(f"Skipping out-of-range segment: [{seg_start}, {seg_end}], length={length}.")
            continue

        label[seg_start] = LABEL_MAP[f"{seg_label}_B"]
        if seg_end > seg_start + 1:
            label[seg_start + 1:seg_end] = [LABEL_MAP[f"{seg_label}_I"]] * (seg_end - seg_start - 1)
        label[seg_end] = LABEL_MAP[f"{seg_label}_E"]

    return label


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------
def process_weight(length, label):
    """Compute per-frame loss weights: rare classes get weight = 1/count."""
    label  = np.array(label)
    weight = np.ones(length)
    for key in ("5", "2_I", "1_I"):
        mask = label == LABEL_MAP[key]
        if mask.sum() > 0:
            weight[mask] = RARE_CLASS_WEIGHT / mask.sum()
    return weight


# ---------------------------------------------------------------------------
# NPZ loading  (single unified function used by both train and test modes)
# ---------------------------------------------------------------------------
def _parse_metadata(raw_meta):
    """Robustly parse metadata from various numpy storage formats."""
    if hasattr(raw_meta, "item"):
        raw_meta = raw_meta.item()
    if isinstance(raw_meta, (bytes, str)):
        return json.loads(raw_meta)
    if isinstance(raw_meta, dict):
        return raw_meta
    raise ValueError(f"Unrecognised metadata type: {type(raw_meta)}")


def load_npz_file(npz_path, contain_label, data_filt=False):
    """Load one .npz file and return a list of (trace, label, meta) tuples.

    Args:
        npz_path      : path to a single .npz file
        contain_label : whether to read segment annotations
        data_filt     : if True, skip traces with no segments or only short segs (<=30 frames)

    Returns:
        records : list of (trace np.ndarray (T,2), label list[int], meta str)
    """
    data     = np.load(npz_path, allow_pickle=True)
    traces   = data["traces"]
    n_traces = traces.shape[0]
    metadata = _parse_metadata(data["metadata"])

    segments = list(data["segments"]) if contain_label and "segments" in data else [None] * n_traces
    base     = os.path.basename(npz_path)

    records  = []
    filtered = 0

    for i in range(n_traces):
        trace_meta   = metadata["trace_metadata"][i]
        trace_length = trace_meta["trace_length"]
        trace        = traces[i, :trace_length, :]

        seg_array = None
        if contain_label and i < len(segments):
            seg_info = segments[i]
            if seg_info is not None and len(seg_info) > 0:
                n_segs    = len(seg_info)
                seg_array = np.zeros((6, n_segs))
                for j, seg in enumerate(seg_info):
                    seg_class = seg[0]
                    if seg_class not in VALID_SEG_LABELS:
                        logger.warning(f"{base} trace {i} seg {j}: unknown class {seg_class}, skipping seg.")
                        continue
                    seg_array[2, j] = seg_class
                    seg_array[3, j] = seg[1] + 1
                    if seg_class == 1:
                        seg_array[5, j] = seg[2] + 1
                    else:
                        seg_array[4, j] = seg[2] + 1

        if seg_array is not None:
            y_seg, y_seg_label = process_trace_seg(seg_array)
        else:
            y_seg, y_seg_label = [], []

        # Filter here using y_seg so logic is identical to original
        if data_filt and not _passes_filter(y_seg):
            filtered += 1
            continue

        label = get_seg_label(trace_length, y_seg, y_seg_label)
        meta  = f"{base}_trace{i}"
        records.append((trace, label, meta))

    if filtered:
        logger.info(f"{base}: filtered out {filtered}/{n_traces} traces (data_filt).")

    return records


def _passes_filter(y_seg):
    """Return True if trace has at least one segment with length > 30 frames.

    Mirrors the original logic exactly: length = end - start (exclusive end).
    """
    return bool(y_seg) and any(e - s > 30 for s, e in y_seg)


def load_all_files(data_path, contain_label, data_filt=False):
    """Load all .npz files from *data_path* and return raw (un-normalised) records.

    Args:
        data_path     : directory containing .npz files
        contain_label : whether segment annotations are expected
        data_filt     : if True, drop traces with no segments or only short segs (≤30 frames)

    Returns:
        records : list of (trace, label, meta)
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Directory not found: {data_path}")

    npz_files = sorted(f for f in os.listdir(data_path) if f.endswith(".npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_path}")

    logger.info(f"Found {len(npz_files)} .npz file(s) in {data_path}")
    records = []

    for npz_file in tqdm(npz_files, desc="Loading .npz files"):
        npz_path = os.path.join(data_path, npz_file)
        try:
            file_records = load_npz_file(npz_path, contain_label, data_filt=data_filt)
        except Exception as exc:
            logger.error(f"Failed to load {npz_file}: {exc}")
            continue

        records.extend(file_records)
        logger.info(f"{npz_file}: loaded {len(file_records)} traces (total so far: {len(records)})")

    return records


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved {len(obj)} records → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg            = load_config()
    preprocess_cfg = cfg.get("preprocess", {})

    parser = argparse.ArgumentParser(description="Preprocess FRET traces for training or evaluation")
    parser.add_argument("--data_mode", type=str,
                        default=preprocess_cfg.get("data_mode", "test"),
                        choices=["train", "test"],
                        help="train: merge all data; test: process each file separately")
    parser.add_argument("--contain_label", default=preprocess_cfg.get("contain_label", False),
                        action="store_true",
                        help="Whether data contains ground-truth labels (required for train mode)")
    args = parser.parse_args()

    if args.data_mode == "train" and not args.contain_label:
        logger.error("train mode requires --contain_label. Exiting.")
        sys.exit(1)

    data_path         = str(cfg["preprocess"]["raw_dir"])
    processed_dir_str = str(cfg["preprocess"]["processed_dir"])

    if "{data_mode}" in processed_dir_str:
        processed_dir_str = processed_dir_str.replace("{data_mode}", args.data_mode)

    os.makedirs(processed_dir_str, exist_ok=True)
    logger.info(f"Output directory: {processed_dir_str}")
    logger.info(f"Mode: {args.data_mode} | contain_label: {args.contain_label}")

    # ------------------------------------------------------------------
    if args.data_mode == "train":
        logger.info("=== Train Mode: loading and merging all data ===")
        records = load_all_files(data_path, contain_label=args.contain_label)
        logger.info(f"Total traces loaded (raw): {len(records)}")

        data_scaled, data_fret = normalize_sep(records)
        logger.info(f"After normalisation: {len(data_scaled)} traces retained.")

        if args.contain_label:
            random.seed(42)
            indices   = list(range(len(data_scaled)))
            random.shuffle(indices)
            split_idx = int(len(indices) * 0.8)

            train_scale = [data_scaled[i] for i in indices[:split_idx]]
            val_scale   = [data_scaled[i] for i in indices[split_idx:]]

            _save_pkl(train_scale, os.path.join(processed_dir_str, "data_train_labeled.pkl"))
            _save_pkl(val_scale,   os.path.join(processed_dir_str, "data_val_labeled.pkl"))

            # Split merged fret by source file (parsed from meta) so Step 3 can
            # process each file individually, matching test-mode <base>_fret.pkl naming.
            fret_by_file = {}
            for rec in data_fret:
                base_name = os.path.splitext(rec[3].rsplit("_trace", 1)[0])[0]
                fret_by_file.setdefault(base_name, []).append(rec)
            for base_name, recs in fret_by_file.items():
                _save_pkl(recs, os.path.join(processed_dir_str, f"{base_name}_fret.pkl"))
        else:
            _save_pkl(data_scaled, os.path.join(processed_dir_str, "data_unlabeled.pkl"))
            _save_pkl(data_fret,   os.path.join(processed_dir_str, "data_fret.pkl"))

    # ------------------------------------------------------------------
    else:
        logger.info("=== Test Mode: processing each .npz file individually ===")
        npz_files = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        if not npz_files:
            logger.error(f"No .npz files found in {data_path}. Exiting.")
            sys.exit(1)

        for npz_file in tqdm(npz_files, desc="Processing files"):
            try:
                records = load_npz_file(npz_file, contain_label=False)
            except Exception as exc:
                logger.error(f"Skipping {npz_file}: {exc}")
                continue

            data_scaled, data_fret_single = normalize_sep(records)
            base_name = os.path.splitext(os.path.basename(npz_file))[0]

            _save_pkl(data_scaled,     os.path.join(processed_dir_str, f"{base_name}_unlabeled.pkl"))
            _save_pkl(data_fret_single, os.path.join(processed_dir_str, f"{base_name}_fret.pkl"))

        logger.info(f"Processed {len(npz_files)} file(s), saved to: {processed_dir_str}")
