#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=2

# Check if --debug flag is passed
if [[ "${1-}" == "--debug" ]]; then
  export ENABLE_DEBUGPY=1
  shift
  echo "🐛 Debug mode enabled: All ranks will wait for debugger"
  echo "   In VS Code: Press F5 and attach to each rank"
fi

# Common arguments (kept in one place for maintainability)
ARGS=(
  data.train.path=openai/gsm8k
  "data.train.train_on_what=['assistant']"
  data.train.apply_chat_template=false
  'data.train.system_prompt=You are a helpful assistant that can answer questions. Let''s think step by step.'
  data.train.prompt_key=question
  data.train.response_key=answer
  "+data.train.kwargs={name: main}"
  "+data.train.kwargs={split: train}"
  "+data.test.kwargs={name: main}"
  "+data.test.kwargs={split: test}"
  data.test_ratio=0.03
  data.train.max_length=16384
  data.train.batch_size=32
  actor.model_name=Qwen/Qwen2.5-1.5B-Instruct
  actor.cp_size=1
  actor.ddp_size=2
  actor.tp_size=1
  actor.max_length_per_device=4096
  trainer.project=GSM8K
  trainer.experiment_name=qwen2-5-1b-inst
  trainer.n_epochs=1
  trainer.use_wandb=false
)

# Use torchrun for both debug and non-debug modes
# Debug is controlled via ENABLE_DEBUGPY environment variable
uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m trainer.trainer.sft \
  "${ARGS[@]}"
