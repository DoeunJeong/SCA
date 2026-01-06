#!/usr/bin/env bash
set -euo pipefail

# --- suppress uv hardlink warning ---
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

# --- server.py에 맞춘 고정값 ---
MODEL_NAME="Qwen/Qwen3-Omni-30B-A3B-Instruct"
VLLM_PORT=8000
API_PORT=8080
VLLM_API_KEY="comedy_key"

# --- CUDA 12.8(cu128) 기준 torch 고정 ---
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
TORCH_VER="2.8.0"
TORCHVISION_VER="0.23.0"
TORCHAUDIO_VER="2.8.0"

# --- vLLM/vLLM-Omni ---
VLLM_VER="0.11.0"

# --- project dir ---
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "$LOG_DIR"
START_LOG="${LOG_DIR}/start.log"
VLLM_LOG="${LOG_DIR}/vllm.log"
API_LOG="${LOG_DIR}/api.log"

exec > >(tee -a "$START_LOG") 2>&1

need_cmd() { command -v "$1" >/dev/null 2>&1; }

wait_for_port() {
  local port="$1"; local name="$2"; local timeout="${3:-180}"
  echo "[wait] $name port $port (timeout ${timeout}s)"
  local start_ts; start_ts="$(date +%s)"
  while true; do
    if python3 - <<PY >/dev/null 2>&1
import socket
s=socket.socket(); s.settimeout(0.5)
try: s.connect(("127.0.0.1", int("${port}")))
finally: s.close()
PY
    then
      echo "[ok] $name is listening on :$port"
      return 0
    fi
    local now_ts; now_ts="$(date +%s)"
    if (( now_ts - start_ts > timeout )); then
      echo "[fail] $name did not open port :$port within ${timeout}s"
      return 1
    fi
    sleep 2
  done
}

detect_gpu_count() {
  if need_cmd nvidia-smi; then nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else echo "0"; fi
}

echo "========================================"
echo "[start] project_dir: $PROJECT_DIR"
echo "[start] model:       $MODEL_NAME"
echo "[start] vllm_port:   $VLLM_PORT"
echo "[start] api_port:    $API_PORT"
echo "========================================"

# --- prereq: uv ---
if ! need_cmd uv; then
  python3 -m pip install -U uv
fi

# --- venv ---
PY_BIN="python3.12"
if ! need_cmd "$PY_BIN"; then PY_BIN="python3"; fi

if [[ ! -d ".venv" ]]; then
  uv venv --python "$PY_BIN" --seed
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "[venv] $(python -V)"

uv pip install -U pip setuptools wheel

# --- constraints (핵심: hub가 1.x로 올라가는 걸 원천봉쇄) ---
cat > "${PROJECT_DIR}/constraints.txt" <<'EOF'
huggingface-hub>=0.34.0,<1.0
transformers<5
EOF

# --- torch (cu128) 먼저 설치 ---
uv pip install --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}"

# --- vLLM / vLLM-Omni 설치 (constraints 적용) ---
uv pip install -c "${PROJECT_DIR}/constraints.txt" "vllm==${VLLM_VER}" --torch-backend=auto
uv pip install -c "${PROJECT_DIR}/constraints.txt" -U vllm-omni

# --- API 런타임 (constraints 적용) ---
uv pip install -c "${PROJECT_DIR}/constraints.txt" -U \
  fastapi "uvicorn[standard]" websockets openai soundfile numpy

# --- (안전장치) 마지막에 한 번 더 constraints 강제 ---
uv pip install --upgrade --force-reinstall -c "${PROJECT_DIR}/constraints.txt" \
  "huggingface-hub>=0.34.0,<1.0" "transformers<5"

# --- start vLLM ---
GPU_COUNT="$(detect_gpu_count)"
TP_SIZE=1
if [[ "$GPU_COUNT" -ge 2 ]]; then TP_SIZE=2; fi

: > "$VLLM_LOG"
echo "[vllm] gpu_count=$GPU_COUNT, tensor_parallel_size=$TP_SIZE"
nohup .venv/bin/vllm serve "$MODEL_NAME" \
  --omni --host 0.0.0.0 --port "$VLLM_PORT" \
  --api-key "$VLLM_API_KEY" \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size "$TP_SIZE" \
  > "$VLLM_LOG" 2>&1 &

echo "[vllm] pid=$! log=$VLLM_LOG"

if ! wait_for_port "$VLLM_PORT" "vLLM-Omni" 360; then
  echo "----- tail vllm.log -----"
  tail -n 200 "$VLLM_LOG" || true
  exit 1
fi

# --- start FastAPI ---
: > "$API_LOG"
nohup .venv/bin/python -m uvicorn server:app \
  --host 0.0.0.0 --port "$API_PORT" --log-level info \
  > "$API_LOG" 2>&1 &

echo "[api] pid=$! log=$API_LOG"

if ! wait_for_port "$API_PORT" "FastAPI" 60; then
  echo "----- tail api.log -----"
  tail -n 200 "$API_LOG" || true
  exit 1
fi

echo "========================================"
echo "[done] vLLM: http://127.0.0.1:${VLLM_PORT}/v1 (key=${VLLM_API_KEY})"
echo "[done] API : http://127.0.0.1:${API_PORT}"
echo "[logs] $START_LOG"
echo "[logs] $VLLM_LOG"
echo "[logs] $API_LOG"
echo "========================================"
