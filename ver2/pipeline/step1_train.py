import os
import pickle
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_root_str, load_config

root = get_root_str()
sys.path.append(root)

import torch
import torch.optim as optim
from tqdm import tqdm
from build_model.seq_dataset import SequenceDataset, BucketBatchSampler, collate_fn, DataLoader
from build_model.seq_model import LSTM_CRF
from build_model.utils import set_seed, pad_crf_output


def setup_logger(log_path):
    """Set up logger to write to both console and file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def cal_metric(preds, labels, weights, positive_value):
    """Calculate per-batch accuracy, recall, precision and F1 for begin/end tokens."""
    begin_end_mask = (weights == 1)
    non_be_mask    = (weights < 1) & (weights > 0)

    b_e_all    = begin_end_mask.sum().item()
    non_be_all = non_be_mask.sum().item()

    b_e_preds    = (preds == labels)[begin_end_mask].sum().item()
    non_be_preds = (preds == labels)[non_be_mask].sum().item()
    all_acc      = (b_e_preds + non_be_preds) / (b_e_all + non_be_all + 1e-8)

    b_e_recall    = b_e_preds / (b_e_all + 1e-8)
    fp_tp         = torch.any(preds.unsqueeze(-1) == positive_value, dim=-1).sum().item()
    b_e_precision = b_e_preds / (fp_tp + 1e-8)
    f1            = 2 * b_e_precision * b_e_recall / (b_e_precision + b_e_recall + 1e-8)
    non_be_recall = non_be_preds / (non_be_all + 1e-8)

    return all_acc, b_e_recall, non_be_recall, b_e_precision, f1


def get_loss_mask(labels, additional_mask, abandon_ratio, device):
    """Create loss mask with optional random abandonment of non-critical tokens.

    Args:
        labels: label tensor
        additional_mask: mask for non-padding tokens
        abandon_ratio: ratio of tokens to randomly drop (0 = disabled)
        device: torch device

    Returns:
        loss_mask: boolean tensor for loss calculation
    """
    loss_mask = (labels != -1).bool()
    combined_mask = additional_mask & loss_mask

    if abandon_ratio > 0:
        abandon_ratio_tensor = torch.full(combined_mask.size(), abandon_ratio, device=device)
        mask_to_zero = torch.bernoulli(abandon_ratio_tensor).bool()
        loss_mask[combined_mask] &= ~mask_to_zero[combined_mask]

    # Ensure first element is non-zero to avoid CRF errors
    loss_mask[:, 0] = 1
    return loss_mask


def run_epoch(model, dataloader, device, positive_value, optimizer=None, abandon_ratio=0):
    """Run one train or eval epoch. Pass optimizer=None for eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    totals = dict(loss=0, acc=0, be_recall=0, non_be_recall=0, be_pre=0, be_f1=0)
    desc   = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for sequences, labels, weights in tqdm(dataloader, desc=desc, leave=False):
            sequences  = sequences.to(device)
            labels     = labels.to(device)
            weights    = weights.to(device)
            label_mask = (labels != -1).bool()
            abandon_mask = (weights < 1) & (weights > 0)

            if is_train:
                optimizer.zero_grad()

            # Use dynamic loss mask if abandon_ratio > 0
            loss_mask = get_loss_mask(labels, abandon_mask, abandon_ratio, device) if abandon_ratio > 0 else label_mask

            loss, outputs = model(
                sequences, loss_mask=loss_mask, pad_mask=label_mask, labels=labels
            )
            preds = pad_crf_output(outputs).to(device)
            if preds.shape[0] != labels.shape[0]:
                preds = preds.T

            if is_train:
                loss.backward()
                optimizer.step()

            acc, be_r, non_be_r, be_p, f1 = cal_metric(preds, labels, weights, positive_value)
            totals["loss"]          += loss.item()
            totals["acc"]           += acc
            totals["be_recall"]     += be_r
            totals["non_be_recall"] += non_be_r
            totals["be_pre"]        += be_p
            totals["be_f1"]         += f1

    n      = len(dataloader)
    prefix = "train" if is_train else "val"
    return {f"{prefix}_{k}": v / n for k, v in totals.items()}


def save_checkpoint(model, optimizer, epoch, path):
    """Save model + optimizer state for full resume support."""
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint. Supports both legacy (state_dict only) and new (dict) formats.
    Returns the saved epoch number, or None for legacy format.
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt["epoch"]
    else:
        # Legacy format: raw state_dict (optimizer state not restored)
        model.load_state_dict(ckpt)
        return None


if __name__ == "__main__":
    _cfg = load_config()["train"]

    parser = argparse.ArgumentParser(description="Train LSTM-CRF sequence labeling model")
    parser.add_argument("--lr",           type=float, default=_cfg["lr"])
    parser.add_argument("--hidden_dim",   type=int,   default=_cfg["hidden_dim"])
    parser.add_argument("--num_layers",   type=int,   default=_cfg["num_layers"])
    parser.add_argument("--dropout",      type=float, default=_cfg["dropout"])
    parser.add_argument("--batch_size",   type=int,   default=_cfg["batch_size"])
    parser.add_argument("--num_epochs",   type=int,   default=_cfg["num_epochs"])
    parser.add_argument("--resume_start", type=int,   default=_cfg["resume_start"])
    parser.add_argument("--patience",     type=int,   default=10,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--abandon_ratio", type=float, default=0,
                        help="Ratio of non-critical tokens to randomly drop from loss calculation")
    args = parser.parse_args()

    set_seed(7)
    torch.cuda.empty_cache()
    torch.set_num_threads(10)

    # ── Paths ─────────────────────────────────────────────────────────────────
    model_name = (
        f"lr{args.lr}_hidden{args.hidden_dim}"
        f"_num_layers{args.num_layers}"
        f"_dropout{args.dropout}"
        f"_batch_size{args.batch_size}"
    )
    model_save_path = os.path.join(root, "output", "ckpt_CRF", model_name)
    log_dir         = os.path.join(root, "output", "logs")
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir,         f"train_{model_name}_{timestamp}.log")
    best_path = os.path.join(model_save_path, "best.pt")

    logger = setup_logger(log_path)
    logger.info(f"Model:          {model_name}")
    logger.info(f"Checkpoint dir: {model_save_path}")
    logger.info(f"Log file:       {log_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_data_path = _cfg.get("train_data_path", "")
    if not train_data_path:
        logger.error("Please set train_data_path in config.yaml")
        exit(1)

    train_file = os.path.join(train_data_path, "data_train_labeled.pkl")
    val_file   = os.path.join(train_data_path, "data_val_labeled.pkl")

    for fpath in (train_file, val_file):
        if not os.path.exists(fpath):
            logger.error(f"Data file not found: {fpath}")
            exit(1)

    with open(train_file, "rb") as f: data_train = pickle.load(f)
    with open(val_file,   "rb") as f: data_val   = pickle.load(f)
    logger.info(f"Data loaded: train={len(data_train)}, val={len(data_val)}")

    bucket_boundaries = [400, 800, 1200, 1600, 1800]
    train_dataset     = SequenceDataset(data_train)
    val_dataset       = SequenceDataset(data_val)
    train_loader      = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(train_dataset, args.batch_size, bucket_boundaries),
        collate_fn=collate_fn,
    )
    val_loader        = DataLoader(
        val_dataset,
        batch_sampler=BucketBatchSampler(val_dataset, args.batch_size, bucket_boundaries),
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device_str = _cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    logger.info(f"Device: {device}")

    model          = LSTM_CRF(2, args.hidden_dim, 7, args.num_layers, args.dropout, bidirectional=True)
    model.to(device)
    optimizer      = optim.Adam(model.parameters(), lr=args.lr)
    positive_value = torch.tensor([1, 3, 4, 6]).to(device)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = args.resume_start
    if start_epoch > 0:
        ckpt_path = os.path.join(model_save_path, f"epoch_{start_epoch}.pt")
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            exit(1)
        saved_epoch = load_checkpoint(model, optimizer, ckpt_path, device)
        logger.info(f"Resumed from: {ckpt_path}  (saved_epoch={saved_epoch})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1      = 0.0
    patience_counter = 0

    logger.info(f"Training: epochs {start_epoch}→{args.num_epochs},  patience={args.patience}")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        train_res = run_epoch(model, train_loader, device, positive_value, optimizer=optimizer, abandon_ratio=args.abandon_ratio)
        val_res   = run_epoch(model, val_loader,   device, positive_value, optimizer=None, abandon_ratio=args.abandon_ratio)

        val_f1  = val_res["val_be_f1"]
        is_best = val_f1 > best_val_f1

        # ── Per-epoch log ─────────────────────────────────────────────────────
        logger.info(
            f"  Train │ loss={train_res['train_loss']:.4f}  acc={train_res['train_acc']:.4f}"
            f"  BE_F1={train_res['train_be_f1']:.4f}"
            f"  recall={train_res['train_be_recall']:.4f}  pre={train_res['train_be_pre']:.4f}"
        )
        logger.info(
            f"  Val   │ loss={val_res['val_loss']:.4f}  acc={val_res['val_acc']:.4f}"
            f"  BE_F1={val_f1:.4f}"
            f"  recall={val_res['val_be_recall']:.4f}  pre={val_res['val_be_pre']:.4f}"
            + ("  ← best" if is_best else "")
        )

        # ── Save epoch checkpoint ─────────────────────────────────────────────
        epoch_ckpt = os.path.join(model_save_path, f"epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, epoch_ckpt)

        # ── Save best model ───────────────────────────────────────────────────
        if is_best:
            best_val_f1      = val_f1
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_path)
            logger.info(f"  ✓ Best model saved  (val_be_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(f"Early stopping triggered: no improvement for {args.patience} epochs.")
            break

    logger.info("=" * 80)
    logger.info(f"Training complete.  Best val_be_f1 = {best_val_f1:.4f}")
    logger.info(f"Best model : {best_path}")
    logger.info(f"Log file   : {log_path}")