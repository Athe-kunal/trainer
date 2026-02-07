set -euo pipefail

# Check if --debug flag is passed
if [[ "$1" == "--debug" ]] 2>/dev/null; then
    export ENABLE_DEBUGPY=1
    echo "🐛 Debug mode enabled: Rank 0 will wait for debugger on port 5678"
    echo "   In VS Code: Press F5 → 'Python Debugger: Attach to Rank 0 (port 5678)'"
fi


uv run torchrun \
    --nproc_per_node=2 \
    -m trainer.trainer.rm \
    data.train.path=Chenmien/SkyworkRM \
    data.train.max_length=2048 \
    critic.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    'data.train.system_prompt="You are a helpful assistant."' \
    data.train.train_on_what=[assistant] \
    data.train.chosen_key=chosen \
    data.train.rejected_key=rejected \
    data.train.apply_chat_template=false \
    data.train.prompt_key=instruction \
    critic.max_length_per_device=8192 \
    trainer.project=SkyworkRM \
    trainer.experiment_name=qwen2.5-1.5b-inst-rm