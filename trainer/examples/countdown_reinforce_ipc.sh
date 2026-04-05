#!/usr/bin/env bash
# IPC (colocate) mode: the vLLM server and trainer share the same GPU(s).
# This script starts the vLLM server automatically before launching training.
set -euo pipefail

NPROC_PER_NODE=2
MODEL=Qwen/Qwen2.5-0.5B-Instruct
INFERENCE_TP_SIZE=2
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-300}"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
VLLM_LOG_FILE="$(mktemp -t countdown-vllm-log.XXXXXX)"

if [[ "${1-}" == "--debug" ]]; then
  export ENABLE_DEBUGPY=1
  shift
  echo "Debug mode enabled: all ranks will wait for debugger."
  echo "In VS Code: press F5 and attach to each rank."
fi

start_vllm_server() {
  echo "Starting vLLM server (IPC backend) on port ${VLLM_PORT}..." >&2
  echo "vLLM logs: ${VLLM_LOG_FILE}" >&2
  VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
    uv run vllm serve "${MODEL}" \
    --weight-transfer-config '{"backend": "ipc"}' \
    --enforce-eager \
    --load-format dummy \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size "${INFERENCE_TP_SIZE}" \
    --port "${VLLM_PORT}" \
    >"${VLLM_LOG_FILE}" 2>&1 &
  echo $!
}

is_vllm_server_ready() {
  local health_url="http://localhost:${VLLM_PORT}/health"
  local models_url="http://localhost:${VLLM_PORT}/v1/models"
  curl -sf --max-time 5 "${health_url}" > /dev/null \
    || curl -sf --max-time 5 "${models_url}" > /dev/null
}

wait_for_vllm_server() {
  local pid="${1}"
  local start_time
  local elapsed_seconds
  start_time="$(date +%s)"
  echo "Waiting for vLLM server on port ${VLLM_PORT} (timeout=${VLLM_STARTUP_TIMEOUT_SECONDS}s)..."
  until is_vllm_server_ready; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ERROR: vLLM server process (pid=${pid}) died unexpectedly."
      echo "----- vLLM log tail -----"
      tail -n 60 "${VLLM_LOG_FILE}" || true
      exit 1
    fi
    elapsed_seconds="$(( $(date +%s) - start_time ))"
    if (( elapsed_seconds >= VLLM_STARTUP_TIMEOUT_SECONDS )); then
      echo "ERROR: Timed out waiting for vLLM server readiness."
      echo "ERROR: ${elapsed_seconds=}, ${VLLM_STARTUP_TIMEOUT_SECONDS=}, ${VLLM_LOG_FILE=}"
      echo "----- vLLM log tail -----"
      tail -n 60 "${VLLM_LOG_FILE}" || true
      exit 1
    fi
    sleep 2
  done
  echo "vLLM server is ready."
}

if ! command -v curl > /dev/null 2>&1; then
  echo "ERROR: curl is required by this script but was not found in PATH."
  exit 1
fi

VLLM_PID="$(start_vllm_server)"
trap 'echo "Stopping vLLM server (pid=${VLLM_PID})..."; kill "${VLLM_PID}" 2>/dev/null; wait "${VLLM_PID}" 2>/dev/null' EXIT

wait_for_vllm_server "${VLLM_PID}"

export VLLM_HOST_MODEL="http://localhost:${VLLM_PORT}"

ARGS=(
  rollout.topology=colocate
  rollout.train.path=Jiayi-Pan/Countdown-Tasks-3to4
  "+rollout.train.kwargs={split: train}"
  rollout.train.prompts_per_rollout=128
  rollout.train.responses_per_prompt=2
  rollout.train.sampling_params.max_new_tokens=1024
  "rollout.train.sampling_params.stop=['</answer>']"
  rollout.train.apply_chat_template=false
  rollout.env_path=trainer/envs/countdown.py
  rollout.server_args.mem_fraction_static=0.6
  +rollout.server_args.cuda_graph_max_bs=16
  +rollout.server_args.dp_size=1
  rollout.server_args.tp_size="${INFERENCE_TP_SIZE}"
  actor.model_name="${MODEL}"
  actor.max_length_per_device=8192
  trainer.project=Countdown
  trainer.experiment_name=qwen2.5-0.5b_grpo_ipc
  trainer.total_steps=512
  trainer.test_freq=8
  "trainer.save_freq="
  actor.avg_level=sequence
  actor.kl.type=loss
  actor.kl.reward_estimator=k3
  trainer.use_wandb=false
)

uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m trainer.trainer.grpo \
  "${ARGS[@]}"
