#!/bin/bash
# Watches the running SEN1-2 training job (scripts/train_baseline.py, launched separately) and
# renders a visual (scripts/visualize_checkpoint.py) every 5 epochs, for both the pix2pix phase
# and the CycleGAN phase that follows it. Deliberately a *separate* process from the training job
# itself -- it only reads checkpoint files training already wrote, never touches the training
# process, so a bug or crash here can't affect the actual run.
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

mkdir -p outputs/progress

render_when_ready() {
    local model=$1 epoch=$2 ckpt_dir=$3
    local ckpt
    ckpt=$(printf "%s/epoch_%04d.pt" "$ckpt_dir" "$epoch")
    while [ ! -f "$ckpt" ]; do
        sleep 30
    done
    local out
    out=$(printf "outputs/progress/%s_epoch%04d.png" "$model" "$epoch")
    echo "[$(date +%T)] rendering $model epoch $epoch..."
    python -m scripts.visualize_checkpoint --model "$model" --checkpoint "$ckpt" \
        --dataset sen1_2 --root data/sen1_2 --random --seed "$epoch" --out "$out" \
        >> outputs/watch_and_render.log 2>&1
}

for epoch in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80; do
    render_when_ready pix2pix "$epoch" outputs/sen1_2_pix2pix
done

for epoch in 5 10 15 20 25; do
    render_when_ready cyclegan "$epoch" outputs/sen1_2_cyclegan
done

echo "[$(date +%T)] watcher done -- all checkpoints rendered." >> outputs/watch_and_render.log
