#!/bin/bash
set -euo pipefail

# ============================================================
# Torthosplatting 统一流水线
# 用法: ./run_pipeline.sh -d pinhole1,pinhole2 -g 0
# ============================================================

# --- 默认配置 ---
GPU_ID="0"
DATA_BASE="/mnt/hdd1/hjh/2d-gaussian-splatting/2dGSdata"
ITERATION="30000"
DATA_DIRS=""

# --- 固定路径 ---
BASE="/mnt/hdd1/hjh"
PROJ="$BASE/Torthosplatting"
CONDA_BASE="/mnt/hdd1/anaconda3"

# --- 解析参数 ---
usage() {
    echo "用法: $0 -d <dir1,dir2,...> [-g <gpu_id>] [--iteration <n>]"
    echo "  -d       数据目录名，逗号分隔 (如 pinhole1,pinhole2)"
    echo "  -g       GPU ID (默认: 0)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d) DATA_DIRS="$2"; shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        --iteration) ITERATION="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

if [ -z "$DATA_DIRS" ]; then
    echo "[ERROR] 必须指定 -d <dir1,dir2,...>"
    usage
fi

# --- 工具函数 ---
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
run_step(){ log ">>> [$1] $2"; }

activate_env() {
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate Torthosplatting
    cd "$PROJ"
    export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
}

run_pipeline_for_dir() {
    local DATA_DIR="$1"
    local DIR_NAME=$(basename "$DATA_DIR")
    local DEPTH_DIR="$DATA_DIR/depthimages"

    log "=============================================="
    log "开始处理: $DIR_NAME"
    log "数据目录: $DATA_DIR"
    log "=============================================="

    activate_env
    cd "$PROJ"

    # Step 0: 生成虚拟正射相机 + 占位图
    run_step "0/5" "gen_virtual_cams ($DIR_NAME)"
    if [ ! -d "$DATA_DIR/sparse/1" ]; then
        cd "$PROJ" && python "$PROJ/utils/gen_virtual_cams.py" --input "$DATA_DIR/sparse/0" --output "$DATA_DIR/sparse/1"
    else
        log "sparse/1 已存在，跳过 gen_virtual_cams"
    fi
    run_step "0/5" "gen_dummy ($DIR_NAME)"
    # 检查是否已有虚拟占位图
    if ls "$DATA_DIR/images/virtual"* 2>/dev/null | head -1 | grep -q .; then
        log "虚拟占位图已存在，跳过 gen_dummy"
    else
        cd "$PROJ" && python "$PROJ/utils/gen_dummy_images.py" --sparse_dir "$DATA_DIR/sparse/1" --images_dir "$DATA_DIR/images"
    fi

    # Step 1: depth_gen
    run_step "1/5" "depth_gen ($DIR_NAME)"
    if [ -d "$DEPTH_DIR" ] && [ "$(ls -A "$DEPTH_DIR" 2>/dev/null)" ]; then
        log "深度图已存在，跳过 depth_gen"
    else
        cd "$PROJ" && python "$PROJ/depth_gen.py" -s "$DATA_DIR" --depth_imagepath "$DEPTH_DIR"
    fi

    # Step 2: calibrate_depth + train
    run_step "2/5" "calibrate_depth ($DIR_NAME)"
    if [ -f "$DATA_DIR/sparse/1/depth_params.json" ]; then
        log "depth_params.json 已存在，跳过 calibrate_depth"
    else
        cd "$PROJ" && python "$PROJ/utils/calibrate_depth.py" --base_dir "$DATA_DIR" --depths_dir "$DEPTH_DIR"
    fi

    run_step "2/5" "train ($DIR_NAME)"
    mkdir -p "$PROJ/output"
    _before=$(ls -1 "$PROJ/output/" 2>/dev/null | sort || true)
    cd "$PROJ" && python "$PROJ/train.py" -s "$DATA_DIR" --optimizer_type sparse_adam --data_device cpu -d "$DEPTH_DIR" --quiet --disable_viewer
    _after=$(ls -1 "$PROJ/output/" 2>/dev/null | sort || true)
    TRAIN_OUTPUT=$(comm -13 <(echo "$_before") <(echo "$_after") | head -1)
    TRAIN_OUTPUT="$PROJ/output/$TRAIN_OUTPUT"
    log "train 输出: $TRAIN_OUTPUT"

    # Step 3: render_ortho
    run_step "3/5" "render_ortho ($DIR_NAME)"
    cd "$PROJ" && python "$PROJ/render_ortho.py" -m "$TRAIN_OUTPUT" -s "$DATA_DIR" --iteration "$ITERATION"

    # Step 4: restore
    run_step "4/5" "restore ($DIR_NAME)"
    RENDER_DIR="$TRAIN_OUTPUT/virtual_views/ours_${ITERATION}/renders"
    mkdir -p "$DATA_DIR/ortho_output"
    cd "$PROJ" && python "$PROJ/restore.py" --input "$RENDER_DIR" --output "$DATA_DIR/ortho_output"

    # Step 5: stitch
    run_step "5/5" "stitch ($DIR_NAME)"
    cd "$PROJ" && python "$PROJ/utils/stitch_ortho.py" --input "$DATA_DIR/ortho_output" --output "$DATA_DIR/ortho_output/ortho_stitched.jpg"

    log "========== $DIR_NAME 处理完成 =========="
}

# --- 主流程 ---
log "检查 GPU $GPU_ID ..."
FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i "$GPU_ID" 2>/dev/null | grep -oP '\d+' || echo "0")
if [ "$FREE_MEM" -lt 1000 ]; then
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | xargs)
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
log "使用 GPU: $GPU_ID"

IFS=',' read -ra DIRS <<< "$DATA_DIRS"
for d in "${DIRS[@]}"; do
    d=$(echo "$d" | xargs)
    if [ -d "$DATA_BASE/$d" ]; then
        FULL_PATH="$DATA_BASE/$d"
    elif [ -d "$d" ]; then
        FULL_PATH="$d"
    else
        log "[ERROR] 目录不存在: $DATA_BASE/$d 或 $d"
        continue
    fi
    run_pipeline_for_dir "$FULL_PATH"
done

log "========== 全部流水线完成 =========="
