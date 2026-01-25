set -euo pipefail

uv run torchrun \
    --nproc_per_node=4 \
    -m trainer.trainer.sft \
    data.train.path=openai/gsm8k \
    data.test.path=openai/gsm8k \
    data.train.messages_key='' \
    'data.train.system_prompt="You are a helpful assistant that can answer questions. Let''s think step by step."' \
    data.train.prompt_key=question \
    data.train.response_key=answer \
    "+data.train.kwargs={name: main}" \
    "+data.train.kwargs={split: train}" \
    "+data.test.kwargs={name: main}" \
    "+data.test.kwargs={split: test}" \
    data.test_ratio=0.03 \
    data.train.max_length=16384 \
    data.train.batch_size=32 \
    actor.model_name=Qwen/Qwen3-4B-Instruct-2507 \
    actor.cp_size=4 \
    actor.max_length_per_device=4096 \
    trainer.project=GSM8K \
    trainer.experiment_name=qwen3-4b-inst-2507 \
    trainer.n_epochs=1 \
    trainer.use_wandb=false