set -euo pipefail

# Check if --debug flag is passed
if [[ "$1" == "--debug" ]] 2>/dev/null; then
    export ENABLE_DEBUGPY=1
    echo "🐛 Debug mode enabled: Rank 0 will wait for debugger on port 5678"
    echo "   In VS Code: Press F5 → 'Python Debugger: Attach to Rank 0 (port 5678)'"
fi

uv run torchrun \
    --nproc_per_node=2 \
    -m trainer.trainer.sft \
    data.train.path=openai/gsm8k \
    data.test.path=openai/gsm8k \
    data.train.train_on_what=['assistant'] \
    data.train.apply_chat_template=false \
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
    actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    actor.cp_size=1 \
    actor.ddp_size=2 \
    actor.max_length_per_device=4096 \
    trainer.project=GSM8K \
    trainer.experiment_name=qwen2-5-1b-inst \
    trainer.n_epochs=1 \
    trainer.use_wandb=false 