# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from contextlib import nullcontext

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
BASE_LORA_NAME = "__base__"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[WARN] Invalid float for {name}={raw!r}; using default {default}")
        return default


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid int for {name}={raw!r}; using default {default}")
        return default
    return max(min_value, value)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer | dict = None):
        """When optimizer is None, it is Reference Policy.
        
        Args:
            config: Actor configuration
            actor_module: The actor model module
            actor_optimizer: Either a single optimizer or a dict mapping adapter names to optimizers
        """
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.is_multi_adapter = isinstance(actor_optimizer, dict)
        self._debug_lora = bool(getattr(config, "debug_lora", False)) or _env_flag(
            "RLLM_DEBUG_LORA"
        )
        self._debug_lora_fail_on_no_update = _env_flag(
            "RLLM_DEBUG_LORA_FAIL_ON_NO_UPDATE"
        )
        self._debug_lora_delta_tol = _env_float(
            "RLLM_DEBUG_LORA_DELTA_TOL",
            1e-12,
        )
        self._debug_lora_sample_size = _env_int(
            "RLLM_DEBUG_LORA_SAMPLE_SIZE",
            8,
            min_value=1,
        )
        self._debug_lora_adv_zero_tol = _env_float(
            "RLLM_DEBUG_LORA_ADV_ZERO_TOL",
            1e-12,
        )
        role = "Ref" if actor_optimizer is None else "Actor"
        if self._debug_lora:
            print(
                "[DEBUG] line 71 DataParallelPPOActor __init__: "
                f"actor_optimizer={actor_optimizer}, is_multi_adapter={self.is_multi_adapter}"
            )
            if self.is_multi_adapter and torch.distributed.get_rank() == 0:
                print(
                    "[LoRA][actor][debug] "
                    f"delta_tol={self._debug_lora_delta_tol} "
                    f"sample_size={self._debug_lora_sample_size} "
                    f"fail_on_no_update={self._debug_lora_fail_on_no_update} "
                    f"adv_zero_tol={self._debug_lora_adv_zero_tol}"
                )

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        import os
        import time
        debug_dir = "/home/cpark97/rllm_prev_version/rllm/run_debug/20251227/solver_judge_efficient_agent"
        debug_file = f"{debug_dir}/dp_actor_forward_micro_batch_{os.getpid()}.txt"
        
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] _forward_micro_batch ENTRY\n")
                f.flush()
        except:
            pass
        
        # Handle adapter switching for any provided lora_name.
        adapter_ctx = nullcontext()
        if "lora_name" in micro_batch:
            try:
                with open(debug_file, "a") as f:
                    f.write(f"[{time.time()}] Multi-adapter mode, lora_name={micro_batch['lora_name']}\n")
                    f.flush()
            except:
                pass
            
            adapter_name = micro_batch["lora_name"]
            if hasattr(self.actor_module, "set_adapter"):
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"[{time.time()}] About to unwrap FSDP and set_adapter\n")
                        f.flush()
                except:
                    pass
                
                # Get the base model (unwrap FSDP if needed)
                base_model = self.actor_module
                if isinstance(base_model, (FSDP, FSDPModule)):
                    if hasattr(base_model, '_fsdp_wrapped_module'):
                        base_model = base_model._fsdp_wrapped_module
                    elif hasattr(base_model, 'module'):
                        base_model = base_model.module
                
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"[{time.time()}] About to call base_model.set_adapter({adapter_name})\n")
                        f.flush()
                except:
                    pass
                
                if adapter_name in (None, "", BASE_LORA_NAME):
                    if hasattr(base_model, "disable_adapter"):
                        adapter_ctx = base_model.disable_adapter()
                elif hasattr(base_model, "set_adapter"):
                    base_model.set_adapter(adapter_name)
                    
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"[{time.time()}] set_adapter completed\n")
                        f.flush()
                except:
                    pass
        
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] Preparing inputs for forward pass\n")
                f.flush()
        except:
            pass
        
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                with adapter_ctx:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                with adapter_ctx:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_fingerprint(self, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """Build a lightweight fingerprint over optimizer parameters.

        This is used in debug_lora mode to detect whether a given adapter's
        parameters actually changed after optimizer.step().
        """
        sum_acc = None
        abs_sum_acc = None
        sq_sum_acc = None
        sample_sum_acc = None
        tensor_count = 0
        total_numel = 0

        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param is None:
                    continue
                data = param.detach()
                if data.numel() == 0:
                    continue
                flat = data.float().reshape(-1)
                if sum_acc is None:
                    device = flat.device
                    sum_acc = torch.zeros((), device=device, dtype=torch.float64)
                    abs_sum_acc = torch.zeros((), device=device, dtype=torch.float64)
                    sq_sum_acc = torch.zeros((), device=device, dtype=torch.float64)
                    sample_sum_acc = torch.zeros((), device=device, dtype=torch.float64)

                sum_acc += flat.sum(dtype=torch.float64)
                abs_sum_acc += flat.abs().sum(dtype=torch.float64)
                sq_sum_acc += (flat * flat).sum(dtype=torch.float64)
                sample_n = min(self._debug_lora_sample_size, flat.numel())
                if sample_n > 0:
                    sample_sum_acc += flat[:sample_n].sum(dtype=torch.float64)

                tensor_count += 1
                total_numel += int(flat.numel())

        if sum_acc is None:
            return {
                "sum": 0.0,
                "abs_sum": 0.0,
                "sq_sum": 0.0,
                "sample_sum": 0.0,
                "tensor_count": 0.0,
                "numel": 0.0,
            }

        return {
            "sum": float(sum_acc.item()),
            "abs_sum": float(abs_sum_acc.item()),
            "sq_sum": float(sq_sum_acc.item()),
            "sample_sum": float(sample_sum_acc.item()),
            "tensor_count": float(tensor_count),
            "numel": float(total_numel),
        }

    def _optimizer_step(
        self,
        adapter_name=None,
        expect_update_overrides: dict[str, bool] | None = None,
        adapter_adv_abs_max: dict[str, float] | None = None,
    ):
        """Perform optimizer step.
        
        Args:
            adapter_name: If using multi-adapter mode, specifies which adapter's optimizer to step.
                         If None in multi-adapter mode, all optimizers are stepped.
            expect_update_overrides: Optional per-adapter overrides for whether an update is
                expected (for example, when effective advantage is exactly zero).
            adapter_adv_abs_max: Optional per-adapter max absolute advantage used for debug.
        """
        assert self.config.grad_clip is not None

        # Select the appropriate optimizer(s)
        if self.is_multi_adapter:
            if adapter_name is not None:
                optimizers_to_step = {adapter_name: self.actor_optimizer[adapter_name]}
            else:
                optimizers_to_step = self.actor_optimizer
        else:
            optimizers_to_step = {"default": self.actor_optimizer}

        lora_debug_metrics = {}
        before_fingerprints = {}
        tracked_adapter_names = []
        if self._debug_lora and self.is_multi_adapter:
            tracked_adapter_names = list(self.actor_optimizer.keys())
            for name in tracked_adapter_names:
                before_fingerprints[name] = self._optimizer_fingerprint(
                    self.actor_optimizer[name]
                )
        
        # Unscale if using scaler
        if self.scaler is not None:
            for optimizer in optimizers_to_step.values():
                self.scaler.unscale_(optimizer)
        
        # Clip gradients (only once for the model)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # Step optimizer(s)
        if self.scaler is not None:
            for optimizer in optimizers_to_step.values():
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                for optimizer in optimizers_to_step.values():
                    optimizer.zero_grad()
            else:
                for optimizer in optimizers_to_step.values():
                    optimizer.step()

        if self._debug_lora and self.is_multi_adapter:
            expected_updates = set(optimizers_to_step.keys())
            debug_rows = []
            mismatch_rows = []
            no_update_rows = []
            for name in tracked_adapter_names:
                before = before_fingerprints[name]
                after = self._optimizer_fingerprint(self.actor_optimizer[name])
                delta_sum = abs(after["sum"] - before["sum"])
                delta_abs_sum = abs(after["abs_sum"] - before["abs_sum"])
                delta_sq_sum = abs(after["sq_sum"] - before["sq_sum"])
                delta_sample_sum = abs(after["sample_sum"] - before["sample_sum"])
                max_delta = max(delta_sum, delta_abs_sum, delta_sq_sum, delta_sample_sum)
                updated = max_delta > self._debug_lora_delta_tol
                expected = name in expected_updates
                expected_reason = "step_adapter" if expected else "not_stepped"
                if expected and expect_update_overrides is not None and name in expect_update_overrides:
                    if not expect_update_overrides[name]:
                        expected = False
                        expected_reason = "zero_advantage"
                adv_abs_max = None
                if adapter_adv_abs_max is not None and name in adapter_adv_abs_max:
                    adv_abs_max = float(adapter_adv_abs_max[name])

                lora_debug_metrics[f"actor/lora_delta/{name}"] = max_delta
                lora_debug_metrics[f"actor/lora_updated/{name}"] = 1.0 if updated else 0.0
                lora_debug_metrics[f"actor/lora_expected_update/{name}"] = (
                    1.0 if expected else 0.0
                )
                lora_debug_metrics[f"actor/lora_checked_tensors/{name}"] = before[
                    "tensor_count"
                ]
                lora_debug_metrics[f"actor/lora_checked_numel/{name}"] = before[
                    "numel"
                ]
                if adv_abs_max is not None:
                    lora_debug_metrics[f"actor/lora_adv_abs_max/{name}"] = adv_abs_max
                    lora_debug_metrics[f"actor/lora_adv_signal/{name}"] = (
                        1.0 if adv_abs_max > self._debug_lora_adv_zero_tol else 0.0
                    )

                debug_row = (
                    f"{name}:delta={max_delta:.3e},expected={expected},updated={updated}"
                )
                if adv_abs_max is not None:
                    debug_row += f",adv_abs_max={adv_abs_max:.3e}"
                if expected_reason != "step_adapter":
                    debug_row += f",expected_reason={expected_reason}"
                debug_rows.append(debug_row)
                if expected != updated:
                    row = (
                        f"{name}(expected={expected},updated={updated},delta={max_delta:.3e},reason={expected_reason})"
                    )
                    mismatch_rows.append(row)
                    if expected and (not updated):
                        no_update_rows.append(row)

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(
                "[LoRA][actor] "
                f"rank={rank} step_adapters={sorted(expected_updates)} "
                f"check={' | '.join(debug_rows)}"
            )
            if mismatch_rows:
                mismatch_msg = (
                    "[LoRA][actor][warn] update expectation mismatch: "
                    + ", ".join(mismatch_rows)
                )
                print(mismatch_msg)
                if self._debug_lora_fail_on_no_update and no_update_rows:
                    raise RuntimeError(
                        "[LoRA][actor][warn] expected adapter update missing: "
                        + ", ".join(no_update_rows)
                    )

        return grad_norm, lora_debug_metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        import os
        import time
        debug_dir = "/home/cpark97/rllm_prev_version/rllm/run_debug/20251227/solver_judge_efficient_agent"
        debug_file = f"{debug_dir}/dp_actor_compute_log_prob_{os.getpid()}.txt"
        
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] dp_actor.compute_log_prob ENTRY, batch size={len(data.batch['input_ids'])}\n")
                f.flush()
        except:
            pass
        
        
        # set to eval
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] Setting actor_module to eval mode\n")
                f.flush()
        except:
            pass
        
        self.actor_module.eval()
        
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] Extracting meta_info\n")
                f.flush()
        except:
            pass

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_lora_names = "lora_names" in data.non_tensor_batch.keys()
        
        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] has_lora_names={has_lora_names}, is_multi_adapter={self.is_multi_adapter}\n")
                f.flush()
        except:
            pass
        
        if self._debug_lora:
            print(
                "[DEBUG] dp_actor.compute_log_prob: "
                f"has_lora_names={has_lora_names}, is_multi_adapter={self.is_multi_adapter}"
            )
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_lora_names:
            non_tensor_select_keys.append("lora_names")

        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] About to select data keys\n")
                f.flush()
        except:
            pass
        
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] Data selected, use_dynamic_bsz={use_dynamic_bsz}\n")
                f.flush()
        except:
            pass
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        try:
            with open(debug_file, "a") as f:
                f.write(f"[{time.time()}] Created {len(micro_batches)} micro_batches\n")
                f.flush()
        except:
            pass
        
        log_probs_lst = []
        entropy_lst = []
        for idx, micro_batch in enumerate(micro_batches):
            try:
                with open(debug_file, "a") as f:
                    f.write(f"[{time.time()}] Processing micro_batch {idx}/{len(micro_batches)}\n")
                    f.flush()
            except:
                pass
            
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            
            # Extract the single adapter name for this batch when provided.
            if "lora_names" in model_inputs:
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"[{time.time()}] micro_batch {idx}: Checking adapters\n")
                        f.flush()
                except:
                    pass
                
                lora_names = model_inputs["lora_names"]
                # All samples in this batch should have the same adapter
                unique_adapters = list(dict.fromkeys(lora_names))
                
                try:
                    with open(debug_file, "a") as f:
                        f.write(f"[{time.time()}] micro_batch {idx}: unique_adapters={unique_adapters}\n")
                        f.flush()
                except:
                    pass
                
                assert len(unique_adapters) == 1, (
                    f"All samples in a micro-batch must use the same adapter. "
                    f"Got: {unique_adapters}. "
                    f"This likely means _adapter_aware_balance in ray_trainer.py failed to properly group samples. "
                    f"Check that trainer.balance_batch=True and rllm.debug_balance_batch=True for debug output."
                )
                model_inputs["lora_name"] = unique_adapters[0]
            
            try:
                with open(debug_file, "a") as f:
                    f.write(f"[{time.time()}] micro_batch {idx}: About to call _forward_micro_batch\n")
                    f.flush()
            except:
                pass
            
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            
            try:
                with open(debug_file, "a") as f:
                    f.write(f"[{time.time()}] micro_batch {idx}: _forward_micro_batch completed\n")
                    f.flush()
            except:
                pass
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_lora_names = "lora_names" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_lora_names:
            non_tensor_select_keys.append("lora_names")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                # Determine which adapter(s) are in this mini-batch
                mini_batch_adapter_names = set()
                if self.is_multi_adapter and has_lora_names:
                    # Peek at the first micro-batch to see which adapter(s) will be used
                    for mb_idx, mb in enumerate(micro_batches):
                        if "lora_names" in mb.non_tensor_batch:
                            lora_names = mb.non_tensor_batch["lora_names"]
                            unique_adapters = list(dict.fromkeys(lora_names))
                            assert len(unique_adapters) == 1, (
                                f"All samples in a micro-batch must use the same adapter. "
                                f"Micro-batch {mb_idx}/{len(micro_batches)} has adapters: {unique_adapters}. "
                                f"This suggests ppo_mini_batch_size={self.config.ppo_mini_batch_size} is not aligned with adapter boundaries. "
                                f"Mini-batch should contain samples from only one adapter after _adapter_aware_balance reordering."
                            )
                            mini_batch_adapter_names.add(unique_adapters[0])
                
                # Zero gradients for relevant optimizers
                if self.is_multi_adapter:
                    if mini_batch_adapter_names:
                        # Zero-grad only the adapters that will be used
                        for adapter_name in mini_batch_adapter_names:
                            self.actor_optimizer[adapter_name].zero_grad()
                    else:
                        # No lora_names found, zero-grad all (backward compat)
                        for optimizer in self.actor_optimizer.values():
                            optimizer.zero_grad()
                else:
                    self.actor_optimizer.zero_grad()

                adapter_adv_abs_max: dict[str, float] = {}
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    # Extract the single adapter name for this micro-batch when provided.
                    current_adapter_name = None
                    if "lora_names" in model_inputs:
                        lora_names = model_inputs["lora_names"]
                        # All samples in this batch should have the same adapter
                        unique_adapters = list(dict.fromkeys(lora_names))
                        assert len(unique_adapters) == 1, f"All samples in a batch must use the same adapter. Got: {unique_adapters}"
                        current_adapter_name = unique_adapters[0]
                        model_inputs["lora_name"] = current_adapter_name
                    
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    if self._debug_lora and self.is_multi_adapter and current_adapter_name is not None:
                        with torch.no_grad():
                            valid_mask = response_mask > 0
                            if bool(valid_mask.any()):
                                adv_abs_max = float(
                                    torch.max(torch.abs(advantages.detach()[valid_mask])).item()
                                )
                            else:
                                adv_abs_max = 0.0
                        prev_adv_max = adapter_adv_abs_max.get(current_adapter_name, 0.0)
                        adapter_adv_abs_max[current_adapter_name] = max(prev_adv_max, adv_abs_max)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_information_itemor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_information_itemor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        
                        # Add adapter-specific metric names if in multi-adapter mode
                        if current_adapter_name:
                            micro_batch_metrics[f"actor/{current_adapter_name}/kl_loss"] = kl_loss.detach().item() * loss_scale_information_itemor
                            micro_batch_metrics[f"actor/{current_adapter_name}/kl_coef"] = self.config.kl_loss_coef
                        else:
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_information_itemor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_information_itemor
                    else:
                        loss = policy_loss * loss_scale_information_itemor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Add adapter-specific metric names if in multi-adapter mode
                    if current_adapter_name:
                        micro_batch_metrics[f"actor/{current_adapter_name}/pg_loss"] = pg_loss.detach().item() * loss_scale_information_itemor
                    else:
                        micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_information_itemor
                    append_to_dict(metrics, micro_batch_metrics)

                # Step the optimizer (for multi-adapter, step only the current adapter's optimizer)
                expect_update_overrides = None
                adapter_adv_abs_max_global = (
                    dict(adapter_adv_abs_max) if adapter_adv_abs_max else {}
                )
                if (
                    self._debug_lora
                    and self.is_multi_adapter
                    and adapter_adv_abs_max_global
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    # Data-parallel ranks can see different local batches.
                    # Use global max-abs-advantage to decide update expectation.
                    reduce_op = (
                        torch.distributed.ReduceOp.MAX
                        if hasattr(torch.distributed.ReduceOp, "MAX")
                        else torch.distributed.ReduceOp.SUM
                    )
                    dist_device = next(self.actor_module.parameters()).device
                    for name in sorted(adapter_adv_abs_max_global.keys()):
                        value = torch.tensor(
                            adapter_adv_abs_max_global[name],
                            device=dist_device,
                            dtype=torch.float32,
                        )
                        torch.distributed.all_reduce(value, op=reduce_op)
                        adapter_adv_abs_max_global[name] = float(value.item())
                if self._debug_lora and self.is_multi_adapter and mini_batch_adapter_names:
                    expect_update_overrides = {}
                    for name in mini_batch_adapter_names:
                        adv_abs_max = adapter_adv_abs_max_global.get(name, 0.0)
                        expect_update_overrides[name] = (
                            adv_abs_max > self._debug_lora_adv_zero_tol
                        )
                grad_norm, lora_debug_metrics = self._optimizer_step(
                    adapter_name=current_adapter_name if self.is_multi_adapter else None,
                    expect_update_overrides=expect_update_overrides,
                    adapter_adv_abs_max=(
                        adapter_adv_abs_max_global
                        if adapter_adv_abs_max_global
                        else None
                    ),
                )
                grad_norm_metric = f"actor/{current_adapter_name}/grad_norm" if current_adapter_name else "actor/grad_norm"
                grad_norm_value = grad_norm.detach().item() if torch.is_tensor(grad_norm) else float(grad_norm)
                mini_batch_metrics = {grad_norm_metric: grad_norm_value}
                append_to_dict(metrics, mini_batch_metrics)
                if lora_debug_metrics:
                    append_to_dict(metrics, lora_debug_metrics)
        
        # Final zero grad for all optimizers
        if self.is_multi_adapter:
            for optimizer in self.actor_optimizer.values():
                optimizer.zero_grad()
        else:
            self.actor_optimizer.zero_grad()
        return metrics
