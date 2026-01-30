set -euo pipefail

uv run torchrun \
    --nproc_per_node=2 \
    -m trainer.trainer.grpo \
    rollout.train.path=Jiayi-Pan/Countdown-Tasks-3to4 \
    rollout.train.prompts_per_rollout=128 \
    rollout.train.responses_per_prompt=4 \
    rollout.train.sampling_params.max_new_tokens=1024 \
    "rollout.train.sampling_params.stop=['</answer>']" \
    rollout.train.apply_chat_template=false \
    rollout.env_path=envs/countdown.py \
    actor.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    actor.max_length_per_device=8192 \
    trainer.project=Countdown \
    trainer.experiment_name=qwen2.5-1.5b_reinforce \
    trainer.total_steps=512 \
    trainer.test_freq=8 \
    trainer.save_freq= \
    actor.avg_level=sequence \
    actor.kl.type=loss \
    actor.kl.reward_estimator=k3 
