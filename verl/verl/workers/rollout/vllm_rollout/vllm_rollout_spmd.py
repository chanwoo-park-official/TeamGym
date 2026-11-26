# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, LoRAConfig
from vllm.lora.request import LoRARequest

try:
    from vllm.entrypoints.openai.api_server import set_sleep_signal
except ImportError:
    # Stock vLLM 0.11 removed this downstream-only hook.  The ordinary
    # engine sleep/wake calls below remain sufficient when it is absent.
    def set_sleep_signal(_value: int) -> None:
        return None

try:
    # https://github.com/vllm-project/vllm/commit/96b9aa5aa076e64c68765232aec343e4d0006e2a
    from vllm.config import CompilationMode

    _use_compilation_mode = True
except ImportError:
    from vllm.config import CompilationLevel

    _use_compilation_mode = False

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.model import get_lora_rank_from_adapter
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code

        lora_adapter_path = getattr(model_config, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            lora_rank = get_lora_rank_from_adapter(lora_adapter_path)
        else:
            lora_rank = model_config.lora_rank

        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(lora_rank)}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend information_itemor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_information_itemor = rope_scaling_config.get("information_itemor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_information_itemor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_information_itemor={rope_scaling_information_itemor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_args = {"cudagraph_capture_sizes": cudagraph_capture_sizes}
                if _use_compilation_mode:
                    compilation_args["mode"] = CompilationMode.VLLM_COMPILE
                else:
                    compilation_args["level"] = CompilationLevel.PIECEWISE
                compilation_config["compilation_config"] = CompilationConfig(**compilation_args)
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            print(f"[DEBUG][vLLMAsyncRollout.generate_sequences] lora_int_ids: {lora_int_ids}")
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size
            print(f"[DEBUG][vLLMAsyncRollout.generate_sequences] lora_requests: {lora_requests}")
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        set_sleep_signal(0)
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        set_sleep_signal(1)
        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        max_loras = self.config.get("max_loras", None)
        max_lora_rank = self.config.get("max_lora_rank", None)
        if max_loras is not None:
            try:
                max_loras = int(max_loras)
            except (TypeError, ValueError):
                max_loras = None
        if max_lora_rank is not None:
            try:
                max_lora_rank = int(max_lora_rank)
            except (TypeError, ValueError):
                max_lora_rank = None
        if max_loras is None and self.model_config.lora_rank > 0:
            max_loras = 1
        if max_lora_rank is None and self.model_config.lora_rank > 0:
            max_lora_rank = get_vllm_max_lora_rank(self.model_config.lora_rank)

        self.lora_config = (
            {"max_loras": max_loras, "max_lora_rank": max_lora_rank}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            set_sleep_signal(0)
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            set_sleep_signal(1)
            self.inference_engine.sleep(level=self.sleep_level)

    @staticmethod
    def _ptr_debug_format_tensor(path: str, tensor: torch.Tensor) -> str:
        ptr = int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else -1
        return (
            f"{path}:ptr=0x{ptr:x},dev={tensor.device},dtype={tensor.dtype},"
            f"shape={tuple(tensor.shape)}"
        )

    def _ptr_debug_collect_gpu_tensors_from_lora_module(
        self, module_path: str, module: Any, max_items: int
    ) -> list[str]:
        previews: list[str] = []
        if module is None or max_items <= 0:
            return previews

        attrs = ("lora_a_stacked", "lora_b_stacked", "lora_bias_stacked")
        for attr_name in attrs:
            if len(previews) >= max_items:
                break
            value = getattr(module, attr_name, None)
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                if value.device.type == "cuda":
                    previews.append(
                        self._ptr_debug_format_tensor(f"{module_path}.{attr_name}", value)
                    )
                continue

            if isinstance(value, (tuple, list)):
                for idx, item in enumerate(value):
                    if len(previews) >= max_items:
                        break
                    if not isinstance(item, torch.Tensor):
                        continue
                    if item.device.type != "cuda":
                        continue
                    previews.append(
                        self._ptr_debug_format_tensor(
                            f"{module_path}.{attr_name}[{idx}]",
                            item,
                        )
                    )

        return previews

    def _ptr_debug_collect_gpu_lora_stacks_from_model(self, model: Any, max_items: int) -> list[str]:
        previews: list[str] = []
        if model is None or max_items <= 0:
            return previews
        if not hasattr(model, "named_modules"):
            return previews

        try:
            for module_name, module in model.named_modules(remove_duplicate=False):
                if len(previews) >= max_items:
                    break
                if not (
                    hasattr(module, "lora_a_stacked")
                    or hasattr(module, "lora_b_stacked")
                    or hasattr(module, "lora_bias_stacked")
                ):
                    continue
                module_path = f"model.module.{module_name}" if module_name else "model.module"
                previews.extend(
                    self._ptr_debug_collect_gpu_tensors_from_lora_module(
                        module_path=module_path,
                        module=module,
                        max_items=max_items - len(previews),
                    )
                )
        except Exception as exc:
            previews.append(f"named_modules_lora_scan_error={type(exc).__name__}")
        return previews

    def _ptr_debug_collect_named_gpu_lora_tensors(self, model: Any, max_items: int) -> list[str]:
        previews: list[str] = []
        if model is None or max_items <= 0:
            return previews
        try:
            if hasattr(model, "named_parameters"):
                for name, tensor in model.named_parameters(recurse=True):
                    if len(previews) >= max_items:
                        break
                    if "lora" not in str(name).lower():
                        continue
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.device.type != "cuda":
                        continue
                    previews.append(self._ptr_debug_format_tensor(f"model.param.{name}", tensor))
            if len(previews) < max_items and hasattr(model, "named_buffers"):
                for name, tensor in model.named_buffers(recurse=True):
                    if len(previews) >= max_items:
                        break
                    if "lora" not in str(name).lower():
                        continue
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.device.type != "cuda":
                        continue
                    previews.append(self._ptr_debug_format_tensor(f"model.buffer.{name}", tensor))
        except Exception as exc:
            previews.append(f"named_lora_scan_error={type(exc).__name__}")
        return previews

    def _ptr_debug_collect_gpu_lora_tensors_from_object(self, root: Any, max_items: int) -> list[str]:
        previews: list[str] = []
        if root is None or max_items <= 0:
            return previews

        visited: set[int] = set()
        stack: list[tuple[str, Any, int]] = [("root", root, 0)]
        max_depth = 4
        max_children_per_node = 48

        while stack and len(previews) < max_items:
            path, current, depth = stack.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            if isinstance(current, torch.Tensor):
                if current.device.type == "cuda" and "lora" in path.lower():
                    previews.append(self._ptr_debug_format_tensor(path, current))
                continue

            if depth >= max_depth:
                continue

            try:
                if isinstance(current, dict):
                    for idx, (key, value) in enumerate(current.items()):
                        if idx >= max_children_per_node:
                            break
                        key_text = str(key)
                        if "lora" in key_text.lower():
                            stack.append((f"{path}[{key_text}]", value, depth + 1))
                    continue

                if isinstance(current, (list, tuple, set)):
                    for idx, value in enumerate(current):
                        if idx >= max_children_per_node:
                            break
                        stack.append((f"{path}[{idx}]", value, depth + 1))
                    continue

                state = getattr(current, "__dict__", None)
                if isinstance(state, dict):
                    for idx, (key, value) in enumerate(state.items()):
                        if idx >= max_children_per_node:
                            break
                        key_lower = str(key).lower()
                        if (
                            "lora" in key_lower
                            or "adapter" in key_lower
                            or key_lower in {
                                "model",
                                "model_runner",
                                "worker",
                                "lora_manager",
                                "_lora_manager",
                                "active_loras",
                                "_active_loras",
                                "punica_wrapper",
                            }
                        ):
                            stack.append((f"{path}.{key}", value, depth + 1))
            except Exception:
                continue

        return previews

    def _ptr_debug_collect_gpu_lora_ptr_preview(self, max_items: int) -> list[str]:
        previews: list[str] = []
        worker = getattr(self.inference_engine, "worker", None)
        if worker is None:
            return ["worker=None"]

        model_runner = getattr(worker, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner is not None else None
        previews.extend(self._ptr_debug_collect_named_gpu_lora_tensors(model, max_items=max_items))

        if len(previews) < max_items:
            remaining = max_items - len(previews)
            previews.extend(self._ptr_debug_collect_gpu_lora_stacks_from_model(model, max_items=remaining))

        if len(previews) < max_items:
            remaining = max_items - len(previews)
            lora_manager = getattr(model_runner, "lora_manager", None) if model_runner is not None else None
            adapter_manager = getattr(lora_manager, "_adapter_manager", None) if lora_manager is not None else None
            lora_modules = getattr(adapter_manager, "modules", None) if adapter_manager is not None else None
            if isinstance(lora_modules, dict):
                for module_name, module in lora_modules.items():
                    if len(previews) >= max_items:
                        break
                    previews.extend(
                        self._ptr_debug_collect_gpu_tensors_from_lora_module(
                            module_path=f"lora_manager.modules[{module_name}]",
                            module=module,
                            max_items=max_items - len(previews),
                        )
                    )

        if len(previews) < max_items:
            remaining = max_items - len(previews)
            previews.extend(self._ptr_debug_collect_gpu_lora_tensors_from_object(worker, max_items=remaining))

        # Keep order but avoid duplicates.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in previews:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= max_items:
                break
        return deduped

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        ptr_debug_limit = max(0, int(os.environ.get("RLLM_PTR_DEBUG_MAX_UPDATES", "64")))
        ptr_debug_idx = int(getattr(self, "_ptr_debug_update_count", 0))
        ptr_debug = ptr_debug_limit > 0 and ptr_debug_idx < ptr_debug_limit
        self._ptr_debug_update_count = ptr_debug_idx + 1
        gpu_ptr_debug_limit = max(0, int(os.environ.get("RLLM_PTR_DEBUG_MAX_GPU_UPDATES", "16")))
        gpu_ptr_debug_idx = int(getattr(self, "_ptr_debug_gpu_update_count", 0))
        gpu_ptr_debug = gpu_ptr_debug_limit > 0 and gpu_ptr_debug_idx < gpu_ptr_debug_limit
        self._ptr_debug_gpu_update_count = gpu_ptr_debug_idx + 1

        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        lora_int_id = kwargs.get("lora_int_id", None)
        lora_name = kwargs.get("lora_name", None)
        lora_path = kwargs.get("lora_path", None)
        print(f"[DEBUG][vLLMAsyncRollout.update_weights] lora_name: {lora_name}")
        print(
            "[DEBUG][vLLMAsyncRollout.update_weights] "
            f"peft_config={'set' if peft_config else 'None'}, base_sync_done={base_sync_done}, "
            f"lora_int_id={lora_int_id}, lora_path={lora_path}"
        )
        if peft_config and base_sync_done:
            # In async mode, remove the adapter being refreshed so vLLM reloads new tensors.
            target_lora_int_id = int(lora_int_id) if lora_int_id is not None else VLLM_LORA_INT_ID
            if lora_int_id is None:
                logger.warning(
                    "vLLMAsyncRollout.update_weights got None lora_int_id; "
                    "falling back to default adapter id=%s",
                    VLLM_LORA_INT_ID,
                )
            self.inference_engine.worker.remove_lora(target_lora_int_id)
            lora_tensors = dict(weights)
            if ptr_debug:
                preview = []
                for tensor_name, tensor in list(lora_tensors.items())[:3]:
                    ptr = int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else -1
                    preview.append(
                        f"{tensor_name}:ptr=0x{ptr:x},dev={tensor.device},dtype={tensor.dtype},shape={tuple(tensor.shape)}"
                    )
                print(
                    "[PTR_DEBUG][update_weights] "
                    f"idx={ptr_debug_idx} lora_name={lora_name} lora_int_id={target_lora_int_id} "
                    f"tensor_count={len(lora_tensors)} preview={' | '.join(preview)}",
                    flush=True,
                )
            lora_request = TensorLoRARequest(
                lora_name=lora_name if lora_name is not None else str(target_lora_int_id),
                lora_int_id=target_lora_int_id,
                lora_path=lora_path if lora_path is not None else VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=lora_tensors,
            )
            self.inference_engine.worker.add_lora(lora_request)
            if ptr_debug:
                loaded_loras = "unavailable"
                lora_slots = "unavailable"
                active_adapters = "unavailable"
                try:
                    loaded_loras = self.inference_engine.worker.list_loras()
                except Exception as exc:
                    loaded_loras = f"error:{type(exc).__name__}"
                try:
                    worker = getattr(self.inference_engine, "worker", None)
                    model_runner = getattr(worker, "model_runner", None) if worker is not None else None
                    lora_manager = getattr(model_runner, "lora_manager", None) if model_runner is not None else None
                    adapter_manager = (
                        getattr(lora_manager, "_adapter_manager", None)
                        if lora_manager is not None
                        else None
                    )
                    if adapter_manager is not None:
                        lora_slots = getattr(adapter_manager, "lora_index_to_id", "unavailable")
                        active_adapters = list(getattr(adapter_manager, "_active_adapters", {}).keys())
                except Exception as exc:
                    lora_slots = f"error:{type(exc).__name__}"
                    active_adapters = f"error:{type(exc).__name__}"
                print(
                    "[PTR_DEBUG][update_weights] "
                    f"idx={ptr_debug_idx} post_add lora_name={lora_name} lora_int_id={target_lora_int_id} "
                    f"loaded_loras={loaded_loras} active_adapters={active_adapters} lora_slots={lora_slots}",
                    flush=True,
                )
            if gpu_ptr_debug:
                gpu_preview_size = max(1, int(os.environ.get("RLLM_PTR_DEBUG_GPU_PREVIEW_SIZE", "6")))
                gpu_preview = self._ptr_debug_collect_gpu_lora_ptr_preview(max_items=gpu_preview_size)
                preview_text = " | ".join(gpu_preview) if gpu_preview else "none"
                print(
                    "[PTR_DEBUG][update_weights][gpu] "
                    f"idx={gpu_ptr_debug_idx} lora_name={lora_name} lora_int_id={target_lora_int_id} "
                    f"found={len(gpu_preview)} preview={preview_text}",
                    flush=True,
                )
            logger.info(f"vLLM load weights, loaded_params: {len(lora_tensors)}")
        else:
            print(
                "[DEBUG][vLLMAsyncRollout.update_weights] skipping add_lora "
                f"(peft_config={'set' if peft_config else 'None'}, base_sync_done={base_sync_done})"
            )
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
