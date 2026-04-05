#!/usr/bin/env bash

set -euo pipefail

MODE="both"
STATE_ENCODER_TYPE="legacy"
PYTHON_EXE=""
PLM_TYPE="qwen"
PLM_SIZE="base"
PLM_PATH=""
DEVICE="cuda:0"
EXP_POOL_PATH="artifacts/exp_pools/exp_pool.pkl"
MODEL_DIR=""
RESUME_CHECKPOINT_DIR=""
TRACE_NAME="fcc-test"
TRACE_NUM=100
VIDEO_NAME="video1"
RANK=128
WINDOW=20
GAMMA="1.0"
LEARNING_RATE="0.0001"
WEIGHT_DECAY="0.0001"
WARMUP_STEPS=2000
NUM_EPOCHS=40
EVAL_PER_EPOCH=2
GRAD_ACCUM_STEPS=32
TARGET_RETURN_SCALE="1.0"
SEED=100003
STATE_FEATURE_DIM=256
PATCH_LEN=3
PATCH_STRIDE=1
NUM_PROTOTYPES=64
REPROGRAM_HEADS=4
REPROGRAM_DROPOUT="0.1"
FIXED_ORDER=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: run_qwen.sh [options] [-- extra run_plm args]

Options:
  --mode {adapt|test|both}
  --state-encoder-type {legacy|patch_reprogram|semantic_reprogram}
  --python-exe PATH
  --plm-type NAME
  --plm-size NAME
  --plm-path PATH
  --device DEVICE
  --exp-pool-path PATH
  --model-dir PATH
  --resume-checkpoint-dir PATH
  --trace NAME
  --trace-num N
  --video NAME
  --rank N
  --window N
  --gamma FLOAT
  --learning-rate FLOAT
  --weight-decay FLOAT
  --warmup-steps N
  --num-epochs N
  --eval-per-epoch N
  --grad-accum-steps N
  --target-return-scale FLOAT
  --seed N
  --state-feature-dim N
  --patch-len N
  --patch-stride N
  --num-prototypes N
  --reprogram-heads N
  --reprogram-dropout FLOAT
  --fixed-order
  --dry-run
  -h, --help

Examples:
  ./run_qwen.sh --mode both
  ./run_qwen.sh --mode adapt --state-encoder-type semantic_reprogram
  ./run_qwen.sh --mode test --model-dir data/ft_plms/qwen_base/.../early_stop_-1_best_model
  ./run_qwen.sh --mode adapt --num-epochs 60 --resume-checkpoint-dir data/ft_plms/llama_small/.../early_stop_-1_checkpoint/38
  ./run_qwen.sh --plm-type llama --plm-size small --plm-path ../downloaded_plms/llama3.2/base --mode both
  ./run_qwen.sh --mode adapt -- --which-layer 8
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

realpath_safe() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
  fi
}

resolve_local_path() {
  local base_dir="$1"
  local path_value="$2"
  if [[ -z "$path_value" ]]; then
    printf '%s\n' "$path_value"
    return
  fi
  if [[ "$path_value" = /* ]]; then
    printf '%s\n' "$path_value"
    return
  fi
  realpath_safe "${base_dir}/${path_value}"
}

resolve_python_exe() {
  local script_dir="$1"
  local preferred_python_exe="$2"
  local candidates=()

  if [[ -n "$preferred_python_exe" ]]; then
    candidates+=("$(resolve_local_path "$script_dir" "$preferred_python_exe")")
  fi

  candidates+=(
    "/data3/wangxh/conda-envs/abr_netllm/bin/python"
    "/data3/wangxh/conda-envs/abr_netllm_qwen/bin/python"
    "${HOME}/.conda/envs/abr_netllm/bin/python"
    "${HOME}/.conda/envs/abr_netllm_qwen/bin/python"
    "${HOME}/miniconda3/envs/abr_netllm/bin/python"
    "${HOME}/miniconda3/envs/abr_netllm_qwen/bin/python"
    "${HOME}/anaconda3/envs/abr_netllm/bin/python"
    "${HOME}/anaconda3/envs/abr_netllm_qwen/bin/python"
    "/opt/conda/envs/abr_netllm/bin/python"
    "/opt/conda/envs/abr_netllm_qwen/bin/python"
  )

  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  die "Cannot find a Python executable for the ABR environment. Please pass --python-exe explicitly."
}

resolve_latest_best_model_dir() {
  local script_dir="$1"
  local search_root="${script_dir}/data/ft_plms/${PLM_TYPE}_${PLM_SIZE}"
  [[ -d "$search_root" ]] || die "Cannot find finetune directory: $search_root"

  local best_dir
  best_dir="$(
    find "$search_root" -type d -name 'early_stop_*_best_model' 2>/dev/null \
      | while IFS= read -r dir; do
          if [[ -f "$dir/modules_except_plm.bin" || -f "$dir/model.bin" ]]; then
            printf '%s\t%s\n' "$(stat -c '%Y' "$dir")" "$dir"
          fi
        done \
      | sort -nr \
      | head -n 1 \
      | cut -f2-
  )"

  [[ -n "$best_dir" ]] || die "Cannot find any best-model directory under $search_root. Please finetune first or pass --model-dir."
  printf '%s\n' "$best_dir"
}

invoke_run_plm() {
  local python_path="$1"
  shift
  local args=("$@")

  printf '\n>>'
  printf ' %q' "$python_path" "${args[@]}"
  printf '\n'

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi

  (
    cd "$SCRIPT_DIR"
    "$python_path" "${args[@]}"
  )
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --state-encoder-type)
      STATE_ENCODER_TYPE="$2"
      shift 2
      ;;
    --python-exe)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --plm-type)
      PLM_TYPE="$2"
      shift 2
      ;;
    --plm-size)
      PLM_SIZE="$2"
      shift 2
      ;;
    --plm-path)
      PLM_PATH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --exp-pool-path)
      EXP_POOL_PATH="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --resume-checkpoint-dir)
      RESUME_CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --trace)
      TRACE_NAME="$2"
      shift 2
      ;;
    --trace-num)
      TRACE_NUM="$2"
      shift 2
      ;;
    --video)
      VIDEO_NAME="$2"
      shift 2
      ;;
    --rank)
      RANK="$2"
      shift 2
      ;;
    --window)
      WINDOW="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --weight-decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --eval-per-epoch)
      EVAL_PER_EPOCH="$2"
      shift 2
      ;;
    --grad-accum-steps)
      GRAD_ACCUM_STEPS="$2"
      shift 2
      ;;
    --target-return-scale)
      TARGET_RETURN_SCALE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --state-feature-dim)
      STATE_FEATURE_DIM="$2"
      shift 2
      ;;
    --patch-len)
      PATCH_LEN="$2"
      shift 2
      ;;
    --patch-stride)
      PATCH_STRIDE="$2"
      shift 2
      ;;
    --num-prototypes)
      NUM_PROTOTYPES="$2"
      shift 2
      ;;
    --reprogram-heads)
      REPROGRAM_HEADS="$2"
      shift 2
      ;;
    --reprogram-dropout)
      REPROGRAM_DROPOUT="$2"
      shift 2
      ;;
    --fixed-order)
      FIXED_ORDER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$MODE" in
  adapt|test|both) ;;
  *) die "--mode must be one of: adapt, test, both" ;;
esac

case "$STATE_ENCODER_TYPE" in
  legacy|patch_reprogram|semantic_reprogram) ;;
  *) die "--state-encoder-type must be one of: legacy, patch_reprogram, semantic_reprogram" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_PYTHON_EXE="$(resolve_python_exe "$SCRIPT_DIR" "$PYTHON_EXE")"
RESOLVED_EXP_POOL_PATH="$(resolve_local_path "$SCRIPT_DIR" "$EXP_POOL_PATH")"
RESOLVED_MODEL_DIR=""
RESOLVED_PLM_PATH=""
RESOLVED_RESUME_CHECKPOINT_DIR=""
if [[ -n "$MODEL_DIR" ]]; then
  RESOLVED_MODEL_DIR="$(resolve_local_path "$SCRIPT_DIR" "$MODEL_DIR")"
fi
if [[ -n "$PLM_PATH" ]]; then
  RESOLVED_PLM_PATH="$(resolve_local_path "$SCRIPT_DIR" "$PLM_PATH")"
fi
if [[ -n "$RESUME_CHECKPOINT_DIR" ]]; then
  RESOLVED_RESUME_CHECKPOINT_DIR="$(resolve_local_path "$SCRIPT_DIR" "$RESUME_CHECKPOINT_DIR")"
fi

if [[ "$MODE" == "adapt" || "$MODE" == "both" ]]; then
  [[ -f "$RESOLVED_EXP_POOL_PATH" ]] || die "Experience pool not found: $RESOLVED_EXP_POOL_PATH"
fi
if [[ -n "$RESOLVED_PLM_PATH" ]]; then
  [[ -d "$RESOLVED_PLM_PATH" ]] || die "Foundation model path not found: $RESOLVED_PLM_PATH"
fi
if [[ -n "$RESOLVED_RESUME_CHECKPOINT_DIR" ]]; then
  [[ -d "$RESOLVED_RESUME_CHECKPOINT_DIR" ]] || die "Resume checkpoint directory not found: $RESOLVED_RESUME_CHECKPOINT_DIR"
fi

COMMON_ARGS=(
  "run_plm.py"
  "--plm-type" "$PLM_TYPE"
  "--plm-size" "$PLM_SIZE"
  "--device" "$DEVICE"
  "--rank" "$RANK"
  "--trace" "$TRACE_NAME"
  "--trace-num" "$TRACE_NUM"
  "--video" "$VIDEO_NAME"
  "--w" "$WINDOW"
  "--gamma" "$GAMMA"
  "--lr" "$LEARNING_RATE"
  "--weight-decay" "$WEIGHT_DECAY"
  "--warmup-steps" "$WARMUP_STEPS"
  "--num-epochs" "$NUM_EPOCHS"
  "--target-return-scale" "$TARGET_RETURN_SCALE"
  "--seed" "$SEED"
  "--state-encoder-type" "$STATE_ENCODER_TYPE"
  "--state-feature-dim" "$STATE_FEATURE_DIM"
)

if [[ -n "$RESOLVED_PLM_PATH" ]]; then
  COMMON_ARGS+=("--plm-path" "$RESOLVED_PLM_PATH")
fi

if [[ "$FIXED_ORDER" -eq 1 ]]; then
  COMMON_ARGS+=("--fixed-order")
fi

if [[ "$STATE_ENCODER_TYPE" == "patch_reprogram" ]]; then
  COMMON_ARGS+=(
    "--patch-len" "$PATCH_LEN"
    "--patch-stride" "$PATCH_STRIDE"
    "--num-prototypes" "$NUM_PROTOTYPES"
    "--reprogram-heads" "$REPROGRAM_HEADS"
    "--reprogram-dropout" "$REPROGRAM_DROPOUT"
  )
elif [[ "$STATE_ENCODER_TYPE" == "semantic_reprogram" ]]; then
  COMMON_ARGS+=(
    "--reprogram-heads" "$REPROGRAM_HEADS"
    "--reprogram-dropout" "$REPROGRAM_DROPOUT"
  )
fi

if [[ "$MODE" == "adapt" || "$MODE" == "both" ]]; then
  ADAPT_ARGS=(
    "--adapt"
    "--exp-pool-path" "$RESOLVED_EXP_POOL_PATH"
    "--eval-per-epoch" "$EVAL_PER_EPOCH"
    "--grad-accum-steps" "$GRAD_ACCUM_STEPS"
  )
  if [[ -n "$RESOLVED_RESUME_CHECKPOINT_DIR" ]]; then
    ADAPT_ARGS+=("--resume-checkpoint-dir" "$RESOLVED_RESUME_CHECKPOINT_DIR")
  fi
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    ADAPT_ARGS+=("${EXTRA_ARGS[@]}")
  fi
  invoke_run_plm "$RESOLVED_PYTHON_EXE" "${COMMON_ARGS[@]}" "${ADAPT_ARGS[@]}"
fi

if [[ "$MODE" == "test" || "$MODE" == "both" ]]; then
  if [[ -z "$RESOLVED_MODEL_DIR" ]]; then
    RESOLVED_MODEL_DIR="$(resolve_latest_best_model_dir "$SCRIPT_DIR")"
  fi
  [[ -d "$RESOLVED_MODEL_DIR" ]] || die "Model directory not found: $RESOLVED_MODEL_DIR"

  printf '\nUsing model dir: %s\n' "$RESOLVED_MODEL_DIR"

  TEST_ARGS=(
    "--test"
    "--model-dir" "$RESOLVED_MODEL_DIR"
  )
  if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    TEST_ARGS+=("${EXTRA_ARGS[@]}")
  fi
  invoke_run_plm "$RESOLVED_PYTHON_EXE" "${COMMON_ARGS[@]}" "${TEST_ARGS[@]}"
fi
