set -euo pipefail

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