.PHONY: vllm-serve-nccl vllm-serve-ipc

# Dev-mode vLLM with dummy weights and NCCL weight transfer.
vllm-serve-nccl:
	VLLM_SERVER_DEV_MODE=1 uv run vllm serve facebook/opt-125m \
		--enforce-eager \
		--weight-transfer-config '{"backend": "nccl"}' \
		--load-format dummy

# Dev-mode vLLM with dummy weights and IPC weight transfer.
vllm-serve-ipc:
	VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
		uv run vllm serve facebook/opt-125m --enforce-eager \
		--weight-transfer-config '{"backend": "ipc"}' \
		--load-format dummy \
		--gpu-memory-utilization 0.5
