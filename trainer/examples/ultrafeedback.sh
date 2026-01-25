#!/usr/bin/env bash
set -euo pipefail

ALGO="dpo"   # default

# parse only --algorithm
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algorithm|-a)
      ALGO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--algorithm dpo|orpo|simpo]"
      exit 1
      ;;
  esac
done

# Validate algorithm
case "$ALGO" in
  dpo|orpo|simpo)
    ;;
  *)
    echo "Error: Unsupported algorithm '$ALGO'"
    echo "Supported algorithms: dpo, orpo, simpo"
    exit 1
    ;;
esac

export MASTER_PORT=29501

uv run torchrun \
  --nproc_per_node=2 \
  --master_port "$MASTER_PORT" \
  -m "trainer.trainer.${ALGO}" \
  data.train.path=openbmb/UltraFeedback \
  data.train.max_length=1024 \
  data.train.apply_chat_template=false \
  data.train.prompt_key=instruction \
  data.train.chosen_key=chosen_response \
  data.train.rejected_key=rejected_response \
  actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
  actor.max_length_per_device=4096 \
  trainer.project=UltraFeedback \
  trainer.experiment_name="qwen2-5-1b-inst-${ALGO}" \
  trainer.n_epochs=1 \
  trainer.use_wandb=false
