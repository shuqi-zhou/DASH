import os
import json
import pickle
import re
import sys
import glob
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_root_str, load_config

root = get_root_str()
sys.path.append(root)

import torch
import argparse
from tqdm import tqdm
from build_model.seq_dataset import SequenceDataset, collate_fn, DataLoader
from build_model.seq_model import LSTM_CRF
from build_model.utils import set_seed, pad_crf_output

_cfg = load_config()["evaluate"]

# Token values for segment boundary detection
START_TOKENS = [1, 4]
END_TOKENS   = [3, 6]
CLASS_NAMES  = {
    0: 'background',
    1: 'seg1_start', 2: 'seg1_mid', 3: 'seg1_end',
    4: 'seg2_start', 5: 'seg2_mid', 6: 'seg2_end',
}


def parse_ckpt_name(ckpt):
    """Extract model hyperparameters from checkpoint directory name."""
    hidden_dim = int(re.search(r'hidden(\d+)', ckpt).group(1))
    num_layers = int(re.search(r'layers(\d+)', ckpt).group(1))
    dropout    = float(ckpt.split('dropout')[1].split('_')[0])
    m          = re.search(r'batch_size(\d+)', ckpt)
    batch_size = int(m.group(1)) if m else None
    return hidden_dim, num_layers, dropout, batch_size


def get_seq_info_batch(preds):
    """Extract start/end token positions for every sequence in a batch.

    Args:
        preds: 2-D tensor (batch, seq_len) or 1-D tensor (seq_len,) on any device.

    Returns:
        List of (start_dic, end_dic) per sequence.
        start_dic keys: '1', '4';  end_dic keys: '3', '6'.
    """
    results  = []
    preds_cpu = preds.cpu()

    seqs = preds_cpu.unsqueeze(0) if preds_cpu.dim() == 1 else preds_cpu

    for seq in seqs:
        start_dic = {str(t): [] for t in START_TOKENS}
        end_dic   = {str(t): [] for t in END_TOKENS}
        for i, val in enumerate(seq.tolist()):
            if val in START_TOKENS:
                start_dic[str(val)].append(i)
            if val in END_TOKENS:
                end_dic[str(val)].append(i)
        results.append((start_dic, end_dic))

    return results


def statistics_on_seg(pred_info, label_info, preds_seq, start='1'):
    """Segment-level evaluation: count, boundary shift, and class-label checks.

    Args:
        pred_info  : (start_dic, end_dic) from get_seq_info_batch for one predicted sequence.
        label_info : (start_dic, end_dic) from get_seq_info_batch for the corresponding labels.
        preds_seq  : 1-D tensor of predicted token ids for this sequence.
        start      : '1' (donor-bleach) or '4' (acceptor-bleach).

    Returns:
        (wrong_samples, acc, wrong_flag)
        wrong_samples : 1 if this trace has a mismatch, else 0.
        acc           : number of segments with correct class label.
        wrong_flag    : True if any structural or positional error was found.
    """
    wrong_samples = 0
    acc = 0
    wrong_flag = False
    end = str(int(start) + 2)

    seg_pred_num  = len(pred_info[0][start])
    seg_label_num = len(label_info[0][start])

    # start/end count must be consistent within predictions
    if seg_pred_num != len(pred_info[1][end]):
        return 1, acc, True

    # predicted segment count must match labeled count
    if seg_label_num != seg_pred_num:
        return 1, acc, True

    # no segments in either → nothing to evaluate
    if seg_pred_num == 0 or seg_label_num == 0:
        return wrong_samples, acc, wrong_flag

    for i in range(seg_pred_num):
        pred_start = pred_info[0][start][i]
        pred_end   = pred_info[1][end][i]
        label_start = label_info[0][start][i]
        label_end   = label_info[1][end][i]

        # class label accuracy: dominant token in the predicted segment
        seg_tokens = preds_seq[pred_start:pred_end]
        if len(seg_tokens) > 0:
            seg_pred_class = int(torch.mode(seg_tokens)[0].cpu().item())
            expected_class = int(start) + 1
            acc += int(seg_pred_class == expected_class)

        # boundary shift tolerance (allow ±2 frames per endpoint)
        shift = abs(pred_start - label_start) + abs(pred_end - label_end) - 4
        shift = max(0, shift)
        if shift > 0:
            wrong_samples += 1
            wrong_flag = True

    return wrong_samples, acc, wrong_flag


def strip_suffix(name, suffixes=('_labeled', '_unlabeled')):
    """Remove a known data-type suffix from a base filename."""
    for s in suffixes:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def print_and_save_stats(pred_seg, class_counts, total_tokens, data_file, stat_path):
    """Print class-distribution stats and write them to *stat_path*."""
    traces_with_seg1 = sum(1 for sd, _ in pred_seg if sd.get('1'))
    traces_with_seg4 = sum(1 for sd, _ in pred_seg if sd.get('4'))
    n = len(pred_seg)

    lines = [
        f"File: {os.path.basename(data_file)}",
        "=" * 60,
        f"Prediction class distribution (total tokens: {total_tokens})",
        "=" * 60,
    ]
    for c, cnt in class_counts.items():
        pct = 100.0 * cnt / total_tokens if total_tokens > 0 else 0.0
        lines.append(f"  class {c} ({CLASS_NAMES[c]:>12s}): {cnt:>8d}  ({pct:5.2f}%)")
    lines += [
        "-" * 60,
        (f"Traces with seg-type1 (start=1): {traces_with_seg1}/{n}"
         f"  ({100*traces_with_seg1/n:.1f}%)") if n else "No traces",
        (f"Traces with seg-type2 (start=4): {traces_with_seg4}/{n}"
         f"  ({100*traces_with_seg4/n:.1f}%)") if n else "No traces",
        "=" * 60,
    ]

    stat_str = "\n".join(lines)
    print(stat_str)
    with open(stat_path, 'w', encoding='utf-8') as f:
        f.write(stat_str + "\n")

    return traces_with_seg1, traces_with_seg4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM-CRF model")
    parser.add_argument('--ckpt',       type=str, default=_cfg['ckpt'])
    parser.add_argument('--load_ckpt',  type=str, default=_cfg['load_ckpt'])
    parser.add_argument('--device',     type=str, default=None,
                        help='cuda:0 / cuda:1 / cpu  (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--verbose',    action='store_true',
                        help='Segment-level accuracy evaluation (requires labeled data)')
    args = parser.parse_args()

    set_seed(7)
    torch.cuda.empty_cache()
    torch.set_num_threads(10)

    # ── Hyperparameters from checkpoint name ──────────────────────────────────
    ckpt = args.ckpt
    hidden_dim, num_layers, dropout, ckpt_batch = parse_ckpt_name(ckpt)
    batch_size    = ckpt_batch if ckpt_batch else args.batch_size
    load_ckpt     = args.load_ckpt
    bidirectional = True
    output_dim    = 7
    input_dim     = 2

    # ── Device ────────────────────────────────────────────────────────────────
    device  = torch.device(args.device) if args.device else \
              torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'  # Fix: check device type, not global availability
    if use_amp:
        print("Using AMP for faster inference")

    # ── Data files ────────────────────────────────────────────────────────────
    test_data_path = _cfg.get('test_data_path', '')
    if not test_data_path:
        print("Error: Please set test_data_path in config.yaml")
        sys.exit(1)  # Fix: use sys.exit instead of exit

    labeled_files   = glob.glob(os.path.join(test_data_path, '*_labeled.pkl'))
    unlabeled_files = glob.glob(os.path.join(test_data_path, '*_unlabeled.pkl'))

    if labeled_files:
        data_files = labeled_files
        print(f"Found {len(labeled_files)} labeled file(s)")
    elif unlabeled_files:
        data_files = unlabeled_files
        print(f"Found {len(unlabeled_files)} unlabeled file(s)")
    else:
        print(f"Error: No data files found at {test_data_path}")
        sys.exit(1)  # Fix: use sys.exit instead of exit

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(root, 'output', 'ckpt_CRF', ckpt, load_ckpt)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        sys.exit(1)  # Fix: use sys.exit instead of exit

    model = LSTM_CRF(input_dim, hidden_dim, output_dim, num_layers, dropout, bidirectional)
    model.to(device)

    # Load checkpoint (supports both new dict format and legacy state_dict format)
    raw_ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(raw_ckpt, dict) and "model_state" in raw_ckpt:
        state_dict = raw_ckpt["model_state"]
        print(f"Checkpoint format: new (saved at epoch {raw_ckpt.get('epoch', '?')})")
    else:
        state_dict = raw_ckpt
        print("Checkpoint format: legacy (state_dict only)")

    load_result   = model.load_state_dict(state_dict, strict=False)
    missing_keys  = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)
    print(f"Checkpoint loaded: {ckpt_path}")
    print(f"Model on device:   {device}")

    # ── Output directory ──────────────────────────────────────────────────────
    eval_output_dir = os.path.join(root, 'output', 'eval')
    os.makedirs(eval_output_dir, exist_ok=True)
    safe_ckpt = os.path.basename(ckpt).replace('/', '_').replace('\\', '_')

    # ── Per-file evaluation ───────────────────────────────────────────────────
    all_stats = []

    for file_idx, data_file in enumerate(data_files):
        print(f"\n{'='*60}")
        print(f"Processing file {file_idx+1}/{len(data_files)}: {os.path.basename(data_file)}")
        print('='*60)

        with open(data_file, 'rb') as pf:
            data_val = pickle.load(pf)
        print(f"Loaded {len(data_val)} traces")

        val_loader = DataLoader(
            SequenceDataset(data_val),
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        pred_seg     = []
        class_counts = {i: 0 for i in range(output_dim)}
        total_tokens = 0
        do_seg_eval  = args.verbose and bool(labeled_files)

        # Verbose segment-level accumulators (only used when do_seg_eval=True)
        seg_wrong1 = seg_acc1 = seg_wrong4 = seg_acc4 = seg_total = 0

        model.eval()
        with torch.no_grad():
            for sequences, labels, weights in tqdm(val_loader, desc="Evaluating"):
                sequences  = sequences.to(device)
                labels     = labels.to(device)
                label_mask = (labels != -1).bool()

                # Fix: use torch.amp.autocast (torch.cuda.amp.autocast is deprecated in PyTorch 2.x)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(sequences, pad_mask=label_mask)

                preds = pad_crf_output(outputs).to(device)
                # Fix: more explicit shape check with clear semantics
                if preds.dim() == 2 and preds.shape[0] != labels.shape[0]:
                    preds = preds.T

                for c in range(output_dim):
                    class_counts[c] += (preds == c).sum().item()
                total_tokens += label_mask.sum().item()

                batch_pred_infos = get_seq_info_batch(preds)
                pred_seg.extend(batch_pred_infos)

                if do_seg_eval:
                    batch_label_infos = get_seq_info_batch(labels)
                    for seq_i, (p_info, l_info) in enumerate(zip(batch_pred_infos, batch_label_infos)):
                        preds_seq = preds[seq_i]
                        ws1, a1, _ = statistics_on_seg(p_info, l_info, preds_seq, start='1')
                        ws4, a4, _ = statistics_on_seg(p_info, l_info, preds_seq, start='4')
                        seg_wrong1 += ws1;  seg_acc1 += a1
                        seg_wrong4 += ws4;  seg_acc4 += a4
                        seg_total  += 1

        # ── Output filenames ──────────────────────────────────────────────────
        base_name    = strip_suffix(os.path.splitext(os.path.basename(data_file))[0])
        pred_seg_path = os.path.join(eval_output_dir, f'{base_name}_pred_{safe_ckpt}.pkl')
        stat_path    = os.path.join(eval_output_dir, f'{base_name}_stats_{safe_ckpt}.txt')

        with open(pred_seg_path, 'wb') as f:
            pickle.dump(pred_seg, f)
        print(f"Predictions saved: {pred_seg_path}  ({len(pred_seg)} traces)")

        seg1, seg4 = print_and_save_stats(pred_seg, class_counts, total_tokens, data_file, stat_path)
        print(f"Stats saved:       {stat_path}")

        if do_seg_eval and seg_total > 0:
            print(f"\n── Segment-level evaluation ({'labeled'}) ──")
            print(f"  Seg-type1 (donor):    wrong={seg_wrong1}/{seg_total}  "
                  f"class_acc={seg_acc1}/{seg_total}")
            print(f"  Seg-type2 (acceptor): wrong={seg_wrong4}/{seg_total}  "
                  f"class_acc={seg_acc4}/{seg_total}")

        all_stats.append({
            "file": os.path.basename(data_file),
            "base_name": base_name,
            "total_traces": len(pred_seg),
            "total_tokens": total_tokens,
            "class_counts": class_counts.copy(),
            "traces_with_seg1": seg1,
            "traces_with_seg4": seg4,
            "output_files": {"predictions": pred_seg_path, "statistics": stat_path},
        })

    # ── Summary JSON ──────────────────────────────────────────────────────────
    eval_config = {
        "evaluation_info": {
            "timestamp":             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "device":                str(device),
            "total_files_processed": len(data_files),
            "total_traces":          sum(s["total_traces"] for s in all_stats),
            "total_tokens":          sum(s["total_tokens"] for s in all_stats),
        },
        "model_config": {
            "checkpoint":    ckpt,
            "load_ckpt":     load_ckpt,
            "checkpoint_path": ckpt_path,
            "input_dim":     input_dim,
            "hidden_dim":    hidden_dim,
            "output_dim":    output_dim,
            "num_layers":    num_layers,
            "dropout":       dropout,
            "bidirectional": bidirectional,
            "batch_size":    batch_size,
        },
        "data_config": {
            "test_data_path":  test_data_path,
            "labeled_files":   len(labeled_files),
            "unlabeled_files": len(unlabeled_files),
            "processed_files": [s["file"] for s in all_stats],
        },
        "checkpoint_loading": {
            "missing_keys":   missing_keys,
            "unexpected_keys": unexpected_keys,
            "load_status":    "success" if not missing_keys else "partial",
        },
        "file_results": [
            {**s, "class_counts": {str(k): v for k, v in s["class_counts"].items()}}
            for s in all_stats
        ],
    }

    config_path = os.path.join(eval_output_dir, f'eval_config_{safe_ckpt}.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(eval_config, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Evaluation complete")
    print(f"  Files processed : {len(data_files)}")
    print(f"  Total traces    : {sum(s['total_traces'] for s in all_stats)}")
    print(f"  Total tokens    : {sum(s['total_tokens'] for s in all_stats)}")
    print(f"  Output dir      : {eval_output_dir}")
    print(f"  Summary JSON    : {config_path}")
    print('='*60)