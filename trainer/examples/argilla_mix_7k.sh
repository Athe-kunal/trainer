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

uv run torchrun \
  --nproc_per_node=2 \
  -m "trainer.trainer.${ALGO}" \
  data.train.path=argilla/dpo-mix-7k\
  '+data.train.kwargs.split=train' \
  data.test.path=argilla/dpo-mix-7k\
  '+data.test.kwargs.split=test' \
  data.train.max_length=1024 \
  data.train.apply_chat_template=false \
  'data.train.system_prompt="You are a helpful assistant."' \
  data.train.apply_chat_template=true \
  data.train.train_on_what=[assistant] \
  data.train.chosen_messages_key=chosen \
  data.train.rejected_messages_key=rejected \
  actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
  actor.max_length_per_device=4096 \
  actor.max_inference_length_per_device=4096 \
  trainer.project=ArgillaMix7k \
  trainer.experiment_name="qwen2-5-1b-inst-${ALGO}" \
  trainer.n_epochs=1 \
  trainer.use_wandb=false
