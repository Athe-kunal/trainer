#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=1
ALGO="dpo"  # default

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      export ENABLE_DEBUGPY=1
      echo "🐛 Debug mode enabled: All ranks will wait for debugger"
      echo "   In VS Code: Press F5 and attach to each rank"
      shift
      ;;
    --algorithm|-a)
      ALGO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--debug] [--algorithm dpo|orpo|simpo|apo_zero|apo_down|kto]"
      exit 1
      ;;
  esac
done

# Validate algorithm
case "$ALGO" in
  dpo|orpo|simpo|apo_zero|apo_down|kto)
    ;;
  *)
    echo "Error: Unsupported algorithm '$ALGO'"
    echo "Supported algorithms: dpo, orpo, simpo, apo_zero, apo_down, kto"
    exit 1
    ;;
esac

# Common arguments (kept in one place for maintainability)
ARGS=(
  data.train.path=argilla/dpo-mix-7k
  "+data.train.kwargs.split=train"
  data.test.path=argilla/dpo-mix-7k
  "+data.test.kwargs.split=test"
  data.train.max_length=1024
  data.train.apply_chat_template=false
  "data.train.system_prompt=You are a helpful assistant."
  data.train.apply_chat_template=true
  data.train.train_on_what=[assistant]
  "+data.train.chosen_messages_key=chosen"
  "+data.train.rejected_messages_key=rejected"
  actor.model_name=Qwen/Qwen2.5-1.5B-Instruct
  actor.max_length_per_device=4096
  actor.max_inference_length_per_device=4096
  trainer.project=ArgillaMix7k
  "trainer.experiment_name=qwen2-5-1b-inst-${ALGO}"
  trainer.n_epochs=1
  trainer.use_wandb=false
)

# Use torchrun for both debug and non-debug modes
# Debug is controlled via ENABLE_DEBUGPY environment variable
uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m "trainer.trainer.${ALGO}" \
  "${ARGS[@]}"
