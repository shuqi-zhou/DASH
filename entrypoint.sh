#!/usr/bin/env bash
set -e

CKPT_SUBDIR="lr0.005_hidden32_num_layers3_dropout0.2_batch_size128"
CKPT_FILE="epoch_89.pt"
CKPT_DST="output/ckpt_CRF/${CKPT_SUBDIR}/${CKPT_FILE}"

# Google Drive file IDs — fill in after uploading
GDRIVE_CKPT_ID="1FVOlCRai3vzPqI5BRwb6Xjkz9i3HcD_4"
GDRIVE_DEMO_ID="1cdrOSCTLxb1xu8fnAGnmFyeGn0RBx7oS"

# ── Download checkpoint if missing ────────────────────────────────────────────
if [ ! -f "${CKPT_DST}" ]; then
    echo "[DASH] Checkpoint not found, downloading from Google Drive..."
    mkdir -p "output/ckpt_CRF/${CKPT_SUBDIR}"
    gdown "${GDRIVE_CKPT_ID}" -O "${CKPT_DST}"
    echo "[DASH] Checkpoint downloaded."
fi

# ── Download demo dataset if missing ──────────────────────────────────────────
if [ ! -d "Demo" ]; then
    echo "[DASH] Demo dataset not found, downloading from Google Drive..."
    gdown "${GDRIVE_DEMO_ID}" -O /tmp/demo.zip
    unzip -q /tmp/demo.zip -d .
    rm /tmp/demo.zip
    echo "[DASH] Demo dataset downloaded."
fi

# ── Ensure output subdirectories exist ────────────────────────────────────────
mkdir -p output/step0_preprocess/test \
         output/step0_preprocess/train \
         output/ckpt_CRF \
         output/eval \
         output/hmm_result \
         output/ckpt_hmm \
         output/logs

CMD="${1:-demo}"

case "${CMD}" in
    demo)
        echo "[DASH] Running full demo: step0 → step2 → step3 --batch"
        python pipeline/step0_preprocess.py
        python pipeline/step2_evaluate_CRF.py
        python pipeline/step3_HMM_test_metric.py --batch
        ;;
    all)
        echo "[DASH] Running all steps including training: step0 → step1 → step2 → step3 --batch"
        python pipeline/step0_preprocess.py
        python pipeline/step1_train.py
        python pipeline/step2_evaluate_CRF.py
        python pipeline/step3_HMM_test_metric.py --batch
        ;;
    step0)
        shift
        echo "[DASH] Running pipeline/step0_preprocess.py $*"
        python pipeline/step0_preprocess.py "$@"
        ;;
    step1)
        shift
        echo "[DASH] Running pipeline/step1_train.py $*"
        python pipeline/step1_train.py "$@"
        ;;
    step2)
        shift
        echo "[DASH] Running pipeline/step2_evaluate_CRF.py $*"
        python pipeline/step2_evaluate_CRF.py "$@"
        ;;
    step3)
        shift
        echo "[DASH] Running pipeline/step3_HMM_test_metric.py $*"
        python pipeline/step3_HMM_test_metric.py "$@"
        ;;
    bash|sh)
        shift
        exec /bin/bash "$@"
        ;;
    *)
        echo "[DASH] Unknown command: ${CMD}"
        echo "Usage: docker run shuqiZhou/dash [demo|all|step0|step1|step2|step3|bash] [args...]"
        exit 1
        ;;
esac
