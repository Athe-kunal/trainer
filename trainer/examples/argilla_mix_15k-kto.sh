#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE=1

# Check if --debug flag is passed
if [[ "${1-}" == "--debug" ]]; then
  export ENABLE_DEBUGPY=1
  shift
  echo "🐛 Debug mode enabled: All ranks will wait for debugger"
  echo "   In VS Code: Press F5 and attach to each rank"
fi

# Common arguments (kept in one place for maintainability)
ARGS=(
  data.train.path=parquet
  "+data.train.kwargs={data_files: https://huggingface.co/datasets/argilla/kto-mix-15k/resolve/main/data/train-00000-of-00001.parquet, split: train}"
  data.test.path=null
  data.test_ratio=0.03
  data.train.max_length=1024
  data.train.batch_size=32
  "data.train.system_prompt=You are a helpful assistant."
  data.train.prompt_key=prompt
  data.train.messages_key=completion
  data.train.label_column=label
  data.train.apply_chat_template=true
  data.train.train_on_what=[assistant]
  actor.model_name=Qwen/Qwen2.5-1.5B-Instruct
  actor.max_length_per_device=4096
  actor.max_inference_length_per_device=4096
  trainer.project=ArgillaMix15kKTO
  trainer.experiment_name=qwen2-5-1b-inst-kto
  trainer.n_epochs=1
  trainer.use_wandb=false
)

# Use torchrun for both debug and non-debug modes
# Debug is controlled via ENABLE_DEBUGPY environment variable
uv run torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m trainer.trainer.kto \
  "${ARGS[@]}"