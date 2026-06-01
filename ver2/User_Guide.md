# DASH User Guide
### Architecture, data flow, and technical details

---

## 0. Prerequisites

### 0.1 Install Docker

| Platform | Install |
|----------|---------|
| Windows / macOS | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Linux | [Docker Engine](https://docs.docker.com/engine/install/) |

### 0.2 GPU support (required)

DASH requires an NVIDIA GPU. Install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
so Docker can access your GPU.

Verify the installation:

```bash
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed in the output.

### 0.3 Start Docker before running

- **Windows / macOS:** Launch Docker Desktop and wait for the engine status to show **"Running"**.
- **Linux:** `sudo systemctl start docker`

> **Common error:** `failed to connect to the docker API` means the Docker
> engine is not running. Start it as described above.

### 0.4 Quick start

Run the full demo pipeline with a single command:

```bash
docker run --gpus all -v ./output:/workspace/output shuqizhou/dash:v1 demo
```

Results are written to the `./output` directory on the host.

---

## 1. Project overview

DASH solves two sequential problems:

1. **Segmentation** — given a raw fluorescence trace of arbitrary length,
   find the time windows that contain informative FRET signal (not
   photobleaching, blinking, or background).
2. **State assignment** — within each valid window, determine how many
   discrete molecular conformations exist and label every frame accordingly.

These are handled by two separate modules that share a common data format.

---

## 2. Overall architecture

```
Raw .npz files
      │
      ▼
┌─────────────────┐
│  step0          │  Normalisation, segmentation labelling,
│  preprocess.py  │  train/test split → .pkl files
└────────┬────────┘
         │  *_labeled.pkl  /  *_unlabeled.pkl
         ▼
┌─────────────────┐
│  step1          │  LSTM-CRF trained on labelled traces
│  train.py       │  to classify each frame as one of 7 token types
└────────┬────────┘
         │  epoch_N.pt  /  best.pt
         ▼
┌─────────────────┐
│  step2          │  Model inference → per-trace dictionaries of
│  evaluate.py    │  segment start/end frame indices
└────────┬────────┘
         │  *_pred_*.pkl
         ▼
┌─────────────────┐
│  step3          │  Grid-searched Gaussian-mixture HMM
│  HMM_metric.py  │  scores each parameter combination → best model
└────────┬────────┘
         │
         ▼
   State assignments  +  Summary figure  +  Metrics CSV
```

---

## 3. Data formats

### 3.1 Raw input (`.npz`)
Each `.npz` file contains an array `traces` of shape `(N, T, 2)`:
- `N` — number of molecules
- `T` — number of time frames
- `2` — donor channel (index 0) and acceptor channel (index 1)

### 3.2 Preprocessed PKL (`*_labeled.pkl`, `*_unlabeled.pkl`)
A Python list of length N. Each element is a tuple `(trace, label_seq)`:
- `trace` — `(T, 2)` float array, mean-normalised
- `label_seq` — `(T,)` int array with token values 0–6 (labeled) or all-zero (unlabeled)

Token vocabulary:

| Value | Meaning |
|-------|---------|
| 0 | background / non-FRET |
| 1 | segment-type-1 start |
| 2 | segment-type-1 middle |
| 3 | segment-type-1 end |
| 4 | segment-type-2 start |
| 5 | segment-type-2 middle |
| 6 | segment-type-2 end |

### 3.3 Segment prediction PKL (`*_pred_*.pkl`)
A list of N tuples `(start_dic, end_dic)`:
- `start_dic` — `{"1": [f1, f2, ...], "4": [f3, ...]}`  frame indices of start tokens
- `end_dic`   — `{"3": [f1, f2, ...], "6": [f3, ...]}` frame indices of end tokens

### 3.4 FRET data PKL (`*_fret.pkl`)
Produced by step0 alongside the labeled/unlabeled PKL.  
A list of N elements. Each element is a tuple `(trace_array, metadata)` where
`trace_array` has shape `(T, 2)` (donor, acceptor).

### 3.5 HMM predictions PKL (`step3_HMM_final_predictions_*.pkl`)
A list of dicts, one per FRET segment:
```python
{
    "predictions": np.ndarray,   # (L,) integer state label per frame
    "fret_data":   np.ndarray,   # (L,) FRET efficiency per frame
    "raw_data":    np.ndarray,   # (L, 2) donor/acceptor intensities
    "full_mean":   list,         # global mean of each state
    "trace_label": str,          # dataset name
}
```

---

## 4. Step-by-step technical description

### Step 0 — Preprocessing (`step0_preprocess.py`)

**What it does:**
- Loads each `.npz` file from `raw_dir`.
- Applies `mean_scale()` to normalise donor + acceptor to zero mean.
- Calls `process_trace_seg()` to parse manually annotated bleaching events
  into start/end frame pairs.
- Computes FRET efficiency: `E = acceptor / (donor + acceptor)`.
- Writes one `*_labeled.pkl` (or `*_unlabeled.pkl`) per input file.
- In `data_mode = "train"`, also creates a merged `data_train_labeled.pkl`
  and `data_val_labeled.pkl` (80/20 split).

**Key config keys:**
```yaml
preprocess:
  raw_dir: "Demo/Test_dataset"
  data_mode: "test"        # "train" merges everything; "test" keeps files separate
  contain_label: true      # false → write unlabeled PKL (no token labels)
```

---

### Step 1 — LSTM-CRF training (`step1_train.py`)

**Model architecture:**

```
Input (T, 2)
    │
    ▼
Bidirectional LSTM  ×  num_layers
    │  hidden state (T, 2·hidden_dim)
    ▼
Linear projection  →  (T, 7)  emission scores
    │
    ▼
CRF layer  →  Viterbi-decoded token sequence (T,)
```
The CRF layer enforces valid transition constraints (e.g. a `seg1_start`
token must eventually be followed by `seg1_end`) and is trained with
negative log-likelihood.

**Loss weighting:**  
Segment boundary tokens (1, 3, 4, 6) receive weight 1.0;
middle tokens (2, 5) receive a lower weight to focus training on
boundary detection accuracy (measured as `BE_F1`).

**Checkpoint naming convention:**
```
ckpt_CRF/lr{lr}_hidden{hidden_dim}_num_layers{num_layers}_dropout{dropout}_batch_size{batch_size}/
    epoch_0.pt   …   epoch_N.pt   best.pt   train_log.csv
```
Each `.pt` file stores `{"epoch": int, "model_state": ..., "optimizer_state": ...}`.

**Key config keys:**
```yaml
train:
  lr: 0.005
  hidden_dim: 32
  num_layers: 3
  dropout: 0.2
  batch_size: 128
  num_epochs: 200
  device: "cuda:0"
  train_data_path: "…/output/step0_preprocess/train"
```

---

### Step 2 — Segment detection (`step2_evaluate_CRF.py`)

**What it does:**
1. Loads the checkpoint specified by `ckpt` + `load_ckpt`.
2. Runs the model in inference mode (no gradient, optional AMP) on all
   test PKL files found in `test_data_path`.
3. Decodes the Viterbi path for each sequence.
4. For each trace, records the frame indices of tokens 1, 4 (starts) and
   3, 6 (ends) into `start_dic` / `end_dic`.
5. Saves one `*_pred_*.pkl` per input file.

**AMP (automatic mixed precision):** enabled automatically when `device` is
CUDA, using `torch.amp.autocast`.

**Output filename pattern:**
```
eval/{base_name}_pred_{ckpt_folder_name}.pkl
```
where `base_name` is the input filename with `_labeled` / `_unlabeled` stripped.

---

### Step 3 — HMM state assignment (`step3_HMM_test_metric.py`)

#### 3a. FRET extraction

For each trace, step3 uses the `start_dic` / `end_dic` from step2 to
extract FRET segments:
- Segment type 1: frames `start_dic["1"][i]` → `end_dic["3"][i]`
- Segment type 2: frames `start_dic["4"][i]` → `end_dic["6"][i]`

`preprocess_fret_cross()` computes FRET efficiency and removes outlier frames.

#### 3b. HMM grid search

Parameters searched:

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `n_mix` | [2] | Gaussian mixture components per hidden state |
| `nstates` | [2, 3, 4, 5] | Number of hidden states |
| `global_threshold` | [0.3, 0.2, 0.1] | Min mean-distance to merge states globally |
| `local_threshold` | [0.15, 0.12] | Min mean-distance to merge states locally per segment |

For each `(n_mix, nstates)` combination a **Gaussian Mixture HMM** is
fitted with the Viterbi algorithm (`hmmlearn.hmm.GMMHMM`).  
The fitted model is cached to `ckpt_hmm/saved_hmm_models/<dataset>/` so
subsequent runs skip refitting.

#### 3c. State merging

After Viterbi decoding:
1. **Global merge** (`analyzer.merge_states`): states whose means are
   within `global_threshold` are collapsed.
2. **Short-dwell removal** (`_handle_short_dwells`): isolated 1–2 frame
   dwells are reassigned to the neighbouring state.
3. **Local merge** (`_merge_close_states_by_estimated_mean`): per-segment
   re-estimation of means, then merge if within `local_threshold`.

#### 3d. Scoring metrics

Each parameter combination is scored on six metrics (equal weight 1/6):

| Metric | What it measures |
|--------|-----------------|
| `bic_score` | Normalised Bayesian Information Criterion (lower BIC = better fit) |
| `silhouette` | Cluster separation in FRET space |
| `consistency` | Fraction of frames with no state change (rewards stable dwells) |
| `separation` | Mean Fisher criterion between adjacent state means |
| `realism` | Physical plausibility: coverage of [0,1], min separation, edge penalty |
| `transition_entropy` | Transition matrix entropy (penalises overly random or rigid models) |

GPU (PyTorch) is used for all metric calculations when CUDA is available.

#### 3e. Output files

| File | Content |
|------|---------|
| `step3_HMM_<name>.csv` | All `(n_mix, nstates, g_thresh, l_thresh)` combinations with all six scores and the composite `final_score` |
| `step3_HMM_final_predictions_<name>.pkl` | List of per-segment dicts (see §3.5) |
| `step3_HMM_summary_<name>.png` | Three-panel summary figure |

---

## 5. Summary figure anatomy

```
┌──────────────────────────┬──────────────────┐
│  A: FRET Histogram       │  B: Occupancy    │
│  · grey bars = all data  │  · bar per state │
│  · coloured curves =     │  · % label on    │
│    weighted Gaussians    │    each bar      │
│  · dashed = state mean   │                  │
└──────────────────────────┴──────────────────┘
│  C: Representative traces (most transitions first)          │
│  · grey scatter = raw FRET                                  │
│  · coloured hlines = HMM state assignment                   │
│  · dotted hlines = global state means                       │
│  · title shows trace index, length, transition count        │
└─────────────────────────────────────────────────────────────┘
```

Traces are selected by sorting all segments by number of state transitions
(descending) so the most dynamic molecules appear first.

---

## 6. Re-running and caching behaviour

| What changed | Restart from |
|-------------|-------------|
| Raw `.npz` data | Step 0 |
| Training hyperparameters | Step 1 |
| Checkpoint selection | Step 2 |
| HMM thresholds only | Step 3 (HMM models reused from cache) |
| Force HMM re-fit | Delete `output/ckpt_hmm/saved_hmm_models/<name>/` |

---

## 7. Extending the pipeline

- **Add more HMM states:** edit `NSTATES_VALUES` in `step3_HMM_test_metric.py`.
- **Change scoring weights:** edit the `final_score` formula in
  `test_evaluation_metrics()`.
