#!/usr/bin/env bash
# Run the vLLM inference server first:
#   make vllm-serve-ipc
# Then run this script in a separate terminal.
set -euo pipefail

NPROC_PER_NODE=2
MODEL=Qwen/Qwen2.5-0.5B-Instruct
INFERENCE_TP_SIZE=2

if [[ "${1-}" == "--debug" ]]; then
  export ENABLE_DEBUGPY=1
  shift
  echo "🐛 Debug mode enabled: all ranks will wait for debugger."
  echo "   In VS Code: press F5 and attach to each rank."
fi

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
