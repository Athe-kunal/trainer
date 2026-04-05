.PHONY: vllm-serve vllm-serve-ipc vllm-serve-nccl

CONFIG ?= trainer/trainer/config/grpo.yaml
TOPOLOGY ?=

vllm-serve:
	uv run python -c "import os, subprocess; from omegaconf import OmegaConf; \
cfg = OmegaConf.to_container(OmegaConf.load('$(CONFIG)'), resolve=True); \
topology = '$(TOPOLOGY)' or cfg['rollout']['topology']; \
backend = 'ipc' if topology == 'colocate' else 'nccl'; \
if topology not in ('colocate', 'disaggregated'): \
    raise ValueError(f'Unsupported rollout topology: {topology}'); \
rollout_cfg = cfg['rollout']; \
server_cfg = rollout_cfg['vllm']['server']; \
server_args = rollout_cfg['server_args']; \
model_path = server_args['model_path']; \
cmd = ['uv', 'run', 'vllm', 'serve', model_path, \
       '--weight-transfer-config', '{\"backend\": \"' + backend + '\"}']; \
if server_cfg.get('enforce_eager', False): \
    cmd.append('--enforce-eager'); \
if server_cfg.get('load_format'): \
    cmd.extend(['--load-format', str(server_cfg['load_format'])]); \
if topology == 'colocate' and server_cfg.get('gpu_memory_utilization') is not None: \
    cmd.extend(['--gpu-memory-utilization', str(server_cfg['gpu_memory_utilization'])]); \
if server_args.get('tp_size') is not None: \
    cmd.extend(['--tensor-parallel-size', str(server_args['tp_size'])]); \
env = os.environ.copy(); \
if server_cfg.get('dev_mode', False): \
    env['VLLM_SERVER_DEV_MODE'] = '1'; \
if topology == 'colocate' and server_cfg.get('allow_insecure_serialization', False): \
    env['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'; \
print('Launching vLLM with topology=', topology, 'backend=', backend); \
print(' '.join(cmd)); \
subprocess.run(cmd, check=True, env=env)"

vllm-serve-ipc:
	$(MAKE) vllm-serve TOPOLOGY=colocate CONFIG=$(CONFIG)

vllm-serve-nccl:
	$(MAKE) vllm-serve TOPOLOGY=disaggregated CONFIG=$(CONFIG)
