from typing import Any
import datasets
import re
import pprint

from trainer.envs import base_env

DAPO_MATH_SYSTEM_PROMPT = "You are a helpful math assistant. \n\nFor every response, please provide a step-by-step reasoning process enclosed in <think> and </think> tags. After the thinking, you need to output the final answer. \n\nRemember to put your answer inside the <answer> and </answer> tags."


def process_fn(example):
    user_prompt = example["prompt"]
    messages = [
        {"role": "system", "content": DAPO_MATH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return {
        "prompt": messages,
        "data_source": example["data_source"],
        "ability": "math",
        "reward_model": example["reward_model"],
        "extra_info": example["extra_info"],
    }


class DAPOMath17KProcessedDataset(base_env.BaseDataset):

    def prepare_dataset(self, ds_name: Any) -> list[base_env.Prompt]:
        ds = datasets.load_dataset(ds_name)
        train_ds = ds["train"]

        # Process dataset in parallel using map
        processed_ds = train_ds.map(
            process_fn,
            num_proc=None,  # Uses all available CPUs
            remove_columns=train_ds.column_names,
        )

        train_ds = [
            base_env.Prompt(
                prompt=item["prompt"],
                data_source=item["data_source"],
                ability=item["ability"],
                reward_model=item["reward_model"],
                extra_info=item["extra_info"],
            )
            for item in processed_ds
        ]
        return train_ds


def extract_think_and_answer(action_text: str) -> tuple[str | None, str | None]:
    think_match = re.search(r"<think>(.*?)</think>", action_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", action_text, re.DOTALL)

    think_text = think_match.group(1).strip() if think_match else None
    answer_text = answer_match.group(1).strip() if answer_match else None

    return think_text, answer_text


class DAPOMath17KProcessedEnv(base_env.BaseEnv[base_env.Prompt, str]):
    """This is a single turn environment"""

    def __init__(self, prompt: base_env.Prompt, *args: Any, **kwargs: Any) -> None:
        self.prompt = prompt

    def step(
        self, action: str, meta_info: dict[str, Any] | None
    ) -> base_env.EnvStepReturn:
        # The action is the LLM response
        gt = self.prompt.reward_model["ground_truth"]
        think_text, answer_text = extract_think_and_answer(action)
        reward = 0.0
        if think_text is None:
            reward -= 1.0
        if answer_text is None:
            reward -= 1.0
        if answer_text == gt and answer_text and think_text:
            reward += 2.0
        return base_env.EnvStepReturn(
            obs=None,
            reward=reward,
            terminated=True,
            info={
                "inference_engine_logps": meta_info.get("logps") if meta_info else None
            },
            truncated=False,
            done=True,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[base_env.Prompt, dict[str, Any]]:
        return self.prompt, {}

    def render(
        self,
    ) -> base_env.RenderPromptState | list[base_env.RenderPromptState] | None:
        pretty_prompt = pprint.pformat(self.prompt)
        print(f"Prompt: {pretty_prompt}")
        return None
