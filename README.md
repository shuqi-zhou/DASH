# DASH — Dynamic Analysis of Single-molecule FRET data with Hidden Markov Models

## Quick Start

```bash
# Full demo — no setup required (GPU)
docker run --gpus all --rm -v $(pwd)/output:/workspace/output shuqizhou/dash:v1

# CPU-only (no GPU)
docker run --rm -v $(pwd)/output:/workspace/output shuqizhou/dash:v1

# Run a specific step
docker run --gpus all --rm -v $(pwd)/output:/workspace/output shuqizhou/dash:v1 step3 --batch

# Interactive shell
docker run --gpus all --rm -it -v $(pwd):/workspace shuqizhou/dash:v1 bash
```

Results are written to `./output/hmm_result/` on your host machine.

---

## Running with Docker — Step-by-Step Guide

This section is written for users with **no programming experience**. Follow it from top to bottom the first time.

---

### Prerequisites checklist

Complete each item before moving on.

| # | Requirement | How to verify |
|---|-------------|---------------|
| 1 | **Docker Desktop installed** | Open a terminal and run `docker --version` → should print `Docker version 29.x.x` or similar |
| 2 | **Docker Desktop is running** | The whale icon appears in the system tray (Windows) or menu bar (Mac) and shows "Docker Desktop is running" |
| 3 | **NVIDIA GPU driver** *(GPU users only)* | Run `nvidia-smi` → your GPU model and driver version are listed |

#### How to install Docker Desktop

1. Visit [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) and download the installer for your OS.
2. Run the installer and follow the on-screen steps (accept defaults).
3. **Restart your computer** after installation.
4. Open Docker Desktop from the Start Menu (Windows) or Applications (Mac).
5. Wait until the status bar at the bottom reads **"Docker Desktop is running"**.

#### Verify Docker works

Open **PowerShell** (Windows) or **Terminal** (Mac / Linux) and run:

```bash
docker run hello-world
```

Expected output (first line of the response):
```
Hello from Docker!
```

If you see that, Docker is ready. If you see an error, see [Troubleshooting](#troubleshooting-docker) below.

---

### Step 1 — Pull the DASH image

Download the pre-built image (contains all software dependencies, ~5 GB, one-time download):

```bash
docker pull shuqizhou/dash:v1
```

Wait until the terminal shows:
```
Status: Downloaded newer image for shuqizhou/dash:v1
```

**Check it worked:**
```bash
docker images
```
You should see a row with `shuqizhou/dash` in the REPOSITORY column and `v1` in the TAG column.

---

### Step 2 — Create an output folder

Navigate to the folder where you want results saved, then create an `output` subfolder.

**Windows (PowerShell):**
```powershell
cd C:\Users\YourName\my_experiment    # change this to your folder
mkdir output
```

**Mac / Linux:**
```bash
cd /Users/YourName/my_experiment      # change this to your folder
mkdir -p output
```

---

### Step 3 — Run the demo

The image ships with built-in demo data so you can test the full pipeline immediately.

**Windows (PowerShell) — with GPU:**
```powershell
docker run --gpus all --rm -v "${PWD}/output:/workspace/output" shuqizhou/dash:v1
```

**Windows (PowerShell) — CPU only (no GPU / GPU errors):**
```powershell
docker run --rm -v "${PWD}/output:/workspace/output" shuqizhou/dash:v1
```

**Mac / Linux — with GPU:**
```bash
docker run --gpus all --rm -v "$(pwd)/output:/workspace/output" shuqizhou/dash:v1
```

**Mac / Linux — CPU only:**
```bash
docker run --rm -v "$(pwd)/output:/workspace/output" shuqizhou/dash:v1
```

#### What you will see

```
[DASH] Checkpoint restored.
[DASH] Running full demo: step0 → step2 → step3 --batch
INFO - Processed 5 file(s), saved to: /workspace/output/step0_preprocess/test
...
Evaluating: 100%|██████████| 15/15
...
INFO - Saved results → /workspace/output/hmm_result/step3_HMM_as0101_0000.csv
```

The full run takes about **2–5 min** with GPU, or **15–30 min** on CPU.

#### Check the run completed successfully

**Windows (PowerShell):**
```powershell
ls output\hmm_result\
```

**Mac / Linux:**
```bash
ls output/hmm_result/
```

You should see files like:
```
step3_HMM_as0101_0000.csv
step3_HMM_final_predictions_as0101_0000.pkl
step3_HMM_summary_as0101_0000.png
```

Open the `.png` files — they are the summary figures showing FRET state assignments.
If the folder is empty or missing, see [Troubleshooting](#troubleshooting-docker) below.

---

### Step 4 — Run on your own data

1. **Prepare a data folder** containing your raw `.npz` trace files, e.g. `my_data/`.

2. **Copy `config.yaml`** from the project to your experiment folder and open it in any text editor (Notepad is fine). Update the `raw_dir` line:
   ```yaml
   raw_dir: "/workspace/my_data"
   ```

3. **Run** with your data and config mounted:

   **Windows (PowerShell):**
   ```powershell
   docker run --gpus all --rm `
     -v "${PWD}/my_data:/workspace/my_data" `
     -v "${PWD}/output:/workspace/output" `
     -v "${PWD}/config.yaml:/workspace/config.yaml" `
     shuqizhou/dash:v1
   ```

   **Mac / Linux:**
   ```bash
   docker run --gpus all --rm \
     -v "$(pwd)/my_data:/workspace/my_data" \
     -v "$(pwd)/output:/workspace/output" \
     -v "$(pwd)/config.yaml:/workspace/config.yaml" \
     shuqizhou/dash:v1
   ```

---

### Troubleshooting (Docker) {#troubleshooting-docker}

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `docker: command not found` | Docker Desktop not installed, or terminal opened before Docker was installed | Install Docker Desktop; close and reopen the terminal |
| "Docker Desktop is starting…" hangs forever (Windows) | WSL 2 not installed | Run `wsl --install` in PowerShell **as Administrator**, restart the computer |
| `docker run hello-world` gives a connection error | Docker Desktop is not running | Open Docker Desktop and wait for "Docker Desktop is running" |
| `docker pull` is very slow or times out | Slow connection or Docker Hub rate limit | Wait and retry; check your internet connection |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit not configured | **Windows:** Docker Desktop → Settings → Resources → WSL Integration → enable for your distro; **Linux:** install `nvidia-container-toolkit` and restart Docker |
| `CUDA out of memory` | GPU VRAM too small | Remove `--gpus all` from the command to run on CPU |
| `output/hmm_result/` is empty after the run | Pipeline crashed partway through | Read the error in the terminal; check `output/logs/` for the full log file |
| `${PWD}` gives an error (Windows) | Running in CMD instead of PowerShell | Open **PowerShell** (search "PowerShell" in Start Menu), or replace `${PWD}` with the full path, e.g. `C:/Users/YourName/my_experiment` |
| Files in `output/` cannot be deleted / are read-only (Linux / Mac) | Container wrote files as root | Run `sudo chown -R $USER output/` to reclaim ownership |

---

DASH is an end-to-end pipeline that turns raw single-molecule FRET traces into
labelled state sequences. It combines a deep-learning segment detector (LSTM-CRF)
with a grid-searched HMM to robustly identify conformational states.

---

## Project layout

```
DASH_code/
├── pipeline/
│   ├── step0_preprocess.py        # Raw data → labelled/unlabelled PKL files
│   ├── step1_train.py             # Train LSTM-CRF on labelled PKL files
│   ├── step2_evaluate_CRF.py      # Run trained model → segment predictions
│   └── step3_HMM_test_metric.py   # HMM grid-search → FRET state assignments
│
├── utils/
│   ├── utils.py                   # FRET preprocessing & state-merging utilities
│   ├── utils_fret_analyse.py      # StateAnalyzer class (merge / short-dwell fix)
│   ├── utils_hmm.py               # HMM fitting wrapper (hmmlearn)
│   ├── utils_full_plot.py         # All plotting functions incl. plot_hmm_summary
│   └── utils_states_assign.py     # State assignment helpers
│
├── build_model/               # LSTM-CRF model definition & dataset helpers
├── config_loader.py           # Reads config.yaml
├── config.yaml                # ← All user-facing settings live here
│
├── output/                    # Created automatically during runs
│   ├── step0_preprocess/      # Preprocessed PKL files
│   ├── ckpt_CRF/              # LSTM-CRF checkpoints
│   ├── eval/                  # Step 2 segment predictions
│   ├── hmm_result/            # Step 3 HMM results & figures
│   ├── ckpt_hmm/              # Cached HMM models
│   └── logs/                  # Log files
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── User_Guide.md
```

---

## Pipeline overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `pipeline/step0_preprocess.py` | Raw `.npz` traces | `*_labeled.pkl` / `*_unlabeled.pkl` |
| 1 | `pipeline/step1_train.py` | `data_train_labeled.pkl` | LSTM-CRF checkpoints in `ckpt_CRF/` |
| 2 | `pipeline/step2_evaluate_CRF.py` | Test PKL + checkpoint | Segment predictions in `eval/` |
| 3 | `pipeline/step3_HMM_test_metric.py` | FRET PKL + segment predictions | State assignments, metrics CSV, summary PNG |

---

## Environment setup

### Option A — Docker (recommended, no version conflicts)

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop)
and, for GPU support, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
# Build image (first time only, ~5 min)
docker compose build

# Start an interactive container with GPU access
# UID/GID ensures permission to write to mounted volumes
UID=$(id -u) GID=$(id -g) docker compose run --rm dash

# Verify GPU access (optional)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Inside the container, run any pipeline step:
python pipeline/step0_preprocess.py

# Exit: Ctrl+D or type exit
```

**Useful commands:**
- `docker compose down` — Stop and remove containers
- `docker compose build --no-cache` — Rebuild from scratch

### Option B — Local Python environment (conda recommended)

```bash
# Create and activate a dedicated conda environment
conda create -n dash python=3.10
conda activate dash
pip install -r requirements.txt

# Verify GPU access (optional)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Update config.yaml (see Configuration section below), then run:
python pipeline/step0_preprocess.py
```

> Tested with Python 3.10, PyTorch 2.0.1+cu118, CUDA 11.8.
> PyTorch cu118 wheels are forward-compatible and work on hosts with CUDA 12.x drivers — no downgrade needed.

---

## Quick-start commands

If a pre-trained checkpoint already exists under `output/ckpt_CRF/`, Step 1 (training) can be
skipped. The minimal inference flow is:

```bash
python pipeline/step0_preprocess.py               # preprocess raw data
# python pipeline/step1_train.py                  # skip if checkpoint is already present
python pipeline/step2_evaluate_CRF.py            # segment detection
python pipeline/step3_HMM_test_metric.py         # HMM state assignment (uses config.yaml)
```

Other Step 3 invocation modes:

```bash
python pipeline/step3_HMM_test_metric.py --list               # list available fret files
python pipeline/step3_HMM_test_metric.py --file as0101_0000   # process a single file
python pipeline/step3_HMM_test_metric.py --batch              # process all files
```

---

## Configuration

All paths and hyperparameters live in **`config.yaml`**.

**Before running locally, update the `root` key and every absolute path to match your
environment.** The default values are set for the Docker container (`/workspace`); running
outside Docker requires changing them to the actual project directory.

```yaml
# Docker (default)
root: "/workspace"

# Local — replace with your actual project path, e.g.:
root: "/home/user/DASH_code"          # Linux / macOS
root: "C:/Users/yourname/DASH_code"   # Windows (forward slashes work)
```

The following keys in `config.yaml` contain absolute paths and **must all be updated**
when `root` changes:

| Key | Section |
|-----|---------|
| `root` | top-level |
| `train_data_path` | `train` |
| `test_data_path` | `evaluate` |
| `fret_dir` | `hmm` |
| `pred_seg_dir` | `hmm` |

Full list of configurable settings:

| Section | Key settings |
|---------|-------------|
| `root` | Absolute path to project root |
| `preprocess` | `raw_dir`, `data_mode` (`train`/`test`), `contain_label` |
| `train` | `lr`, `hidden_dim`, `num_layers`, `dropout`, `batch_size`, `device` |
| `evaluate` | `ckpt` (folder name), `load_ckpt` (filename), `test_data_path` |
| `hmm` | `fret_dir`, `pred_seg_dir`, `mode` (`single`/`batch`/`list`), `file_name` |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `Error: HMM fret directory not found` | `fret_dir` path wrong | Check absolute path in config.yaml |
| `Warning: No pred_seg file found for ...` | Step 2 not run yet | Run `step2_evaluate_CRF.py` first |
| `No *_fret.pkl files found` | Naming mismatch | Run `--list` to see available files |
| `CUDA out of memory` | GPU memory full | Reduce `batch_size` or set `device: "cpu"` |
| Log file is empty | Logging already initialised by another module | Fixed in current code (handlers cleared before setup) |
| Docker GPU not found | `nvidia-container-toolkit` missing | Install toolkit or switch to CPU mode |
| Checkpoint not found | Wrong `ckpt` / `load_ckpt` in config | Check `output/ckpt_CRF/` for available folders/files |

---

## Outputs summary

| File | Description |
|------|-------------|
| `hmm_result/step3_HMM_<name>.csv` | Score table for every HMM parameter combination |
| `hmm_result/step3_HMM_final_predictions_<name>.pkl` | Per-segment state assignments |
| `hmm_result/step3_HMM_summary_<name>.png` | Summary figure (histogram + occupancy + traces) |
| `output/logs/step3_hmm_<timestamp>.log` | Full run log |

---

## Data Format

### Input `.npz` files

Each `.npz` file must contain three arrays:

| Key | Type / Shape | Description |
|-----|-------------|-------------|
| `traces` | `float32 (N, T_max, 2)` | N traces; column 0 = donor intensity, column 1 = acceptor intensity; pad shorter traces with zeros |
| `segments` | `object (N,)` | Per-trace segment annotations (set to `None` for unlabelled data) |
| `metadata` | JSON string or dict | Must contain `trace_metadata[i].trace_length` (int) for each trace |

### Segment annotation formats

Two column layouts are accepted:

**6-column format** (1-based indices, used by training data):

| Column | Content |
|--------|---------|
| 0–1 | unused |
| 2 | label: `1` = donor bleach, `2` = acceptor bleach |
| 3 | segment start (1-based) |
| 4 | segment end, used when label = 2 (1-based) |
| 5 | segment end, used when label = 1 (1-based) |

**3-column format** (0-based indices):

| Column | Content |
|--------|---------|
| 0 | label: `1` = donor bleach, `2` = acceptor bleach |
| 1 | segment start (0-based, inclusive) |
| 2 | segment end (0-based, inclusive) |

### Output pickle files

`step3_HMM_final_predictions_<name>.pkl` contains a Python list. Each element is a dict for one trace segment:

| Key | Type | Description |
|-----|------|-------------|
| `predictions` | `np.ndarray (T,)` | Per-frame HMM state index (0-based) |
| `fret_data` | `np.ndarray (T,)` | FRET efficiency values for this segment |
| `full_mean` | `list[float]` | Mean FRET value for each state across the whole dataset |
| `raw_data` | `np.ndarray (T, 2)` | Normalised donor/acceptor intensities |
| `trace_label` | `str` | Dataset name this segment belongs to |
