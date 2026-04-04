import os
from typing import Literal

import requests
from openai import OpenAI, AsyncOpenAI
from openai.types.completion import Completion

BASE_URL = os.environ.get("VLLM_HOST_MODEL", "http://localhost:8000")


def _completion_to_choice_texts(response: Completion) -> list[str]:
    """Map a completion response to texts in choice index order."""
    return ["" if c.text is None else c.text for c in response.choices]


def generate_completions(
    client: OpenAI,
    model: str,
    prompts: list[str],
    max_tokens: int = 32,
    temperature: float = 0,
    n: int = 1,
) -> list[list[str]]:
    """Generate completions using the OpenAI-compatible API.

    Returns one inner list per prompt; each inner list has up to ``n`` strings
    (one per API choice), ordered by choice index.
    """
    results: list[list[str]] = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        results.append(_completion_to_choice_texts(response))
    return results


async def agenerate_completions(
    aclient: AsyncOpenAI,
    model: str,
    prompts: list[str],
    max_tokens: int = 32,
    temperature: float = 0,
    n: int = 1,
) -> list[list[str]]:
    """Generate completions using the async OpenAI-compatible API.

    Returns one inner list per prompt; each inner list has up to ``n`` strings
    (one per API choice), ordered by choice index.
    """
    results: list[list[str]] = []
    for prompt in prompts:
        response = await aclient.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        results.append(_completion_to_choice_texts(response))
    return results


def pause_generation(base_url: str) -> None:
    """Pause generation via HTTP endpoint."""
    url = f"{base_url}/pause"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def resume_generation(base_url: str) -> None:
    """Resume generation via HTTP endpoint."""
    url = f"{base_url}/resume"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def init_weight_transfer_engine(
    base_url: str,
    *,
    master_address: str | None = None,
    master_port: int | None = None,
    rank_offset: int = 1,
    world_size: int | None = None,
    backend: Literal["nccl", "ipc"] = "ipc",
) -> None:
    """Initialize weight transfer via HTTP endpoint (IPC or NCCL)."""
    url = f"{base_url}/init_weight_transfer_engine"
    if backend == "ipc":
        payload: dict[str, dict[str, object]] = {"init_info": dict()}
    else:
        if (
            master_address is None
            or master_port is None
            or world_size is None
        ):
            raise ValueError(
                "NCCL backend requires master_address, master_port, and world_size"
            )
        payload = {
            "init_info": dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
            )
        }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()


def update_weights(
    base_url: str,
    names: list[str],
    dtype_names: list[str],
    shapes: list[list[int]],
    packed: bool = False,
) -> None:
    """Update weights via HTTP endpoint."""
    url = f"{base_url}/update_weights"
    payload = {
        "update_info": dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=packed,
        )
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()


def get_world_size(base_url: str) -> int:
    """Get world size from the vLLM server."""
    url = f"{base_url}/get_world_size"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]
