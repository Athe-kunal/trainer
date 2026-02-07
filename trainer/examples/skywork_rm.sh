#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=2

# Check if --debug flag is passed
if [[ "${1-}" == "--debug" ]]; then
  export ENABLE_DEBUGPY=1
  shift
  echo "🐛 Debug mode enabled: All ranks will wait for debugger"
  echo "   Rank 0 → port 5678"
  echo "   Rank 1 → port 5679"
  echo "   In VS Code: Press F5 and attach to each rank"
fi

# Common arguments (kept in one place for maintainability)
ARGS=(
  data.train.path=Chenmien/SkyworkRM
  data.train.max_length=2048
  critic.model_name=Qwen/Qwen2.5-1.5B-Instruct
  "data.train.system_prompt=You are a helpful assistant."
  data.train.train_on_what=[assistant]
  data.train.chosen_key=chosen
  data.train.rejected_key=rejected
  data.train.apply_chat_template=false
  data.train.prompt_key=instruction
  critic.max_length_per_device=8192
  trainer.project=SkyworkRM
  trainer.experiment_name=qwen2.5-1.5b-inst-rm
)

# Use torchrun for both debug and non-debug modes
# Debug is controlled via ENABLE_DEBUGPY environment variable
uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m trainer.trainer.rm \
  "${ARGS[@]}"
