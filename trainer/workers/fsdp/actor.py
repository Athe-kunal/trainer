from typing import Dict, Optional
from omegaconf import DictConfig
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM
from trainer.workers.fsdp.base import FSDPWorker
from trainer.utils.sequences import (
    count_total,
    gather_along_cp,
    slide_along_cp,
)
from trainer.utils.fsdp.context_parallelism import update_ring_attn_params
from trainer.utils.functions import (
    compute_logps_and_entropy,
    aggregate_values,
)
from trainer.utils.algorithms import (
    dpo_loss,
    apo_down_loss,
    apo_zero_loss,
    grpo_loss,
    orpo_loss,
    simpo_loss,
    kto_loss,
)
from trainer.utils.logging import (
    progress_bar,
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log,
)


class FSDPActor(FSDPWorker):

    def __init__(self, config: DictConfig, train: bool):
        super().__init__(config, train)

        if config.use_liger_kernel:
            assert (
                config.tp_size == 1
            ), "Liger kernel is not compatible with tensor parallelism."
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            model_cls = AutoLigerKernelForCausalLM
        else:
            model_cls = AutoModelForCausalLM

        with self._init_weight_context():
            self.model = model_cls.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                dtype=getattr(torch, config.dtype),
            )

        self._prepare_model_optimizer()

    def _forward(
        self,
        minibatch: Dict[str, torch.Tensor],
        prefix: str = "",
        return_entropy: bool = False,
    ) -> Dict[str, torch.Tensor]:

        minibatch, cu_seqlens = slide_along_cp(
            minibatch, self.device_mesh["cp"].get_group(), self.device_mesh["tp"].size()
        )
        update_ring_attn_params(self.device_mesh["cp"].get_group(), cu_seqlens)
        logits = self.model(
            input_ids=minibatch["states"],
            position_ids=minibatch["position_ids"],
            use_cache=False,
        ).logits.to(torch.float32)
        # bfloat16 is unstable for the subsequent `logsumexp` operation.
        # See https://github.com/OpenRLHF/OpenRLHF/pull/634.
        compute_logps_and_entropy(
            logits / getattr(self.config, "temperature", 1.0),
            minibatch,
            self.device_mesh["tp"].get_group(),
            prefix,
            return_entropy,
        )
        return gather_along_cp(
            minibatch, self.device_mesh["cp"].get_group(), cu_seqlens
        )

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(
        self,
        tensor_dict: Optional[Dict[str, torch.Tensor]],
        step: int,
        pair: bool = False,
    ) -> Optional[Dict[str, torch.Tensor]]:
        # `labels` is a 1D [N] tensor (KTO); exclude it from scatter/gather
        # which assumes all tensors are 2D [batch, seq_len].
        labels = tensor_dict.pop("labels", None) if tensor_dict is not None else None
        minibatches = self._scatter_data(tensor_dict, pair=pair)
        self._load_model_to_device(torch.cuda.current_device())

        prefix = "old_" if self.train else "ref_"
        self.model.eval()
        processed_minibatches = []
        for minibatch in progress_bar(minibatches, desc=f"Compute {prefix}logps"):
            processed_minibatch = self._forward(minibatch, prefix)
            processed_minibatches.append(processed_minibatch)

        if not self.train:
            self._load_model_to_device("cpu")
        result = self._gather_data(processed_minibatches)
        if labels is not None and result is not None:
            result["labels"] = labels
        return result

    @time_logger("update_actor")
    def sft_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict)
        self.model.train(train)

        total_actions, total_sequences = count_total(
            minibatches, ("action_mask", "eos_mask"), self.device_mesh["dp"].get_group()
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="SFT step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            loss = aggregate_values(
                -minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences,
            )
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            suffix = "train" if train else "test"
            metrics[f"loss/{suffix}"].append(loss.item())
        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def dpo_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="DPO step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            suffix = "train" if train else "test"
            losses, metric = dpo_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def apo_down_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="APO Down step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            suffix = "train" if train else "test"
            losses, metric = apo_down_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def apo_zero_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="APO Zero step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            suffix = "train" if train else "test"
            losses, metric = apo_zero_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def orpo_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="ORPO step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            suffix = "train" if train else "test"
            losses, metric = orpo_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def kto_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="KTO step"):
            labels = minibatch.pop("labels")
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            minibatch["labels"] = labels
            suffix = "train" if train else "test"
            losses, metric = kto_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def simpo_step(
        self, tensor_dict: Optional[Dict[str, torch.Tensor]], train: bool, step: int
    ):
        minibatches = self._scatter_data(tensor_dict, pair=True)
        self.model.train(train)

        total_pairs = (
            count_total(minibatches, "eos_mask", self.device_mesh["dp"].get_group())
            // 2
        )
        metrics = defaultdict(list)
        idx = 0
        for minibatch in progress_bar(minibatches, desc="SimPO step"):
            with torch.set_grad_enabled(train):
                minibatch = self._forward(minibatch)
            suffix = "train" if train else "test"
            losses, metric = simpo_loss(self.config, minibatch, suffix)
            loss = losses.sum() / total_pairs
            if train:
                self._scale_loss(loss).backward()
                idx += 1
            metric[f"loss/{suffix}"] = [loss.item()]
            for k, v in metric.items():
                metrics[k].extend(v)

        if train:
            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                metrics["grad_norm"].append(grad_norm)
        gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

    @time_logger("update_actor")
    def ppo_update(self, tensor_dict: Optional[Dict[str, torch.Tensor]], step: int):
        if step < self.config.freeze_steps:
            return
        batches = self._scatter_data(tensor_dict, pack_minibatches=True)
        self._load_model_to_device(torch.cuda.current_device())

        self.model.train()
        tbar = progress_bar(
            total=sum([len(batch) for batch in batches]), desc="Update Actor"
        )
        metrics = defaultdict(list)
        idx = 0
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch, ("action_mask", "eos_mask"), self.device_mesh["dp"].get_group()
            )
            metric = defaultdict(list)
            for minibatch in batch:

                minibatch = self._forward(minibatch, return_entropy=True)
                losses, clip_ratios, llm_old_approx_kl = grpo_loss(
                    self.config, minibatch
                )

                loss, clip_ratio, llm_old_approx_kl, entropy = aggregate_values(
                    (losses, clip_ratios, llm_old_approx_kl, minibatch["entropy"]),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences,
                )

                self._scale_loss(loss).backward()
                idx += 1
                tbar.update()
                metric["actor/entropy"].append(entropy.item())
                metric["actor/loss"].append(loss.item())
                metric["actor/clip_ratio"].append(clip_ratio.item())
                metric["actor/llm_old_approx_kl"].append(llm_old_approx_kl.item())

            # Update optimizer after accumulating gradients
            do_update = idx % self.config.grad_accumulation_steps == 0
            grad_norm = self._optimizer_step(do_update)
            if do_update:
                idx = 0
                metrics["actor/grad_norm"].append(grad_norm)
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, self.device_mesh["dp"].get_group())
                )

        rank0_log(metrics, step)
        if self.config.adv_estimator == "gae":
            self._load_model_to_device("cpu")

    @time_logger("update_rollout")
    def update_rollout(self, rollout, step):

        state_dict = self._get_model_state_dict()
        rollout.update(progress_bar(state_dict.items(), desc="Update rollout"))
