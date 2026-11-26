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
import argparse
import asyncio
import json
import logging
import os
from pprint import pprint
from typing import Any, Callable, Optional

import cloudpickle as pickle
import numpy as np
import ray
import vllm.entrypoints.cli.serve
import zmq
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_tcp_uri
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

try:
    from vllm.entrypoints.openai.api_server import set_sleep_signal
except ImportError:
    # Stock vLLM 0.11 removed this downstream-only hook.  The ordinary
    # engine sleep/wake calls below remain sufficient when it is absent.
    def set_sleep_signal(_value: int) -> None:
        return None

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _is_vllm_engine_trace_enabled() -> bool:
    return str(os.environ.get("RLLM_DEBUG_VLLM_ENGINE_TRACE", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name, None)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        addresses = addresses[dp_rank_local * tp_size : (dp_rank_local + 1) * tp_size]
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            if address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))

        for output in outputs:
            if isinstance(output, Exception):
                raise output
        return outputs

    def check_health(self):
        return


class vLLMHttpServerBase:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
        """
        super().__init__()

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self._debug_lora = bool(getattr(self.config, "debug_lora", False)) or (
            os.getenv("RLLM_DEBUG_LORA", "").strip().lower()
            in ("1", "true", "yes", "y", "on")
        )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None
        self._disable_sleep_wake = _env_bool("RLLM_VLLM_DISABLE_SLEEP_WAKE", False)
        self._disable_sleep_wake_on_timeout = _env_bool(
            "RLLM_VLLM_DISABLE_SLEEP_WAKE_ON_TIMEOUT", True
        )
        self._wake_timeout_s = _env_float("RLLM_VLLM_SERVER_WAKE_TIMEOUT_S", 0.0)
        self._sleep_timeout_s = _env_float("RLLM_VLLM_SERVER_SLEEP_TIMEOUT_S", 0.0)
        self._sleep_wake_runtime_disabled = False

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"vLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def _sleep_wake_enabled(self) -> bool:
        if self._disable_sleep_wake:
            return False
        if self._sleep_wake_runtime_disabled:
            return False
        return True

    async def _await_with_timeout(self, coro, timeout_s: float):
        if timeout_s > 0:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        return await coro

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": True,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        # update lora-related args
        enable_lora = self.model_config.lora_rank > 0 or bool(self.config.get("enable_lora", False))
        max_loras, max_lora_rank = _get_rollout_lora_limits(self.config, self.model_config)
        if enable_lora:
            args["enable_lora"] = True
            if max_loras is not None:
                args["max_loras"] = max_loras
            if max_lora_rank is not None:
                args["max_lora_rank"] = max_lora_rank

        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            else:
                server_args.append(f"--{k}")
                server_args.append(str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 2. setup distributed executor backend
        distributed_executor_backend = ExternalZeroMQDistributedExecutor if len(self.workers) > 0 else None
        server_args.distributed_executor_backend = distributed_executor_backend

        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_requests=engine_args.disable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()

        app = build_app(args)
        await init_app_state(engine_client, vllm_config, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        logger.info(
            "Starting uvicorn server (replica_rank=%s, node_rank=%s, address=%s)",
            self.replica_rank,
            self.node_rank,
            self._server_address,
        )
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)
        logger.info(
            "Uvicorn server started (replica_rank=%s, node_rank=%s, address=%s, port=%s)",
            self.replica_rank,
            self.node_rank,
            self._server_address,
            self._server_port,
        )

    async def run_headless(self, args: argparse.Namespace):
        # Create the EngineConfig.
        engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port  # add to config too
        handshake_address = get_tcp_uri(host, port)

        # Create the engines.
        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_model_len = int(self.config.max_model_len)
        prompt_len = len(prompt_ids)
        if prompt_len >= max_model_len:
            logger.warning(
                "[sat_warn] prompt_len exceeds max_model_len, truncating "
                "(prompt_len=%s, max_model_len=%s)",
                prompt_len,
                max_model_len,
            )
            prompt_ids = prompt_ids[-(max_model_len - 1) :]
            prompt_len = len(prompt_ids)
        available_tokens = max(1, max_model_len - prompt_len)
        raw_sampling_params = dict(sampling_params)
        requested_max_tokens = sampling_params.pop("max_tokens", None)
        requested_max_new_tokens = sampling_params.pop("max_new_tokens", None)
        if requested_max_tokens is None:
            requested_max_tokens = requested_max_new_tokens
        if requested_max_tokens is None:
            default_response_len = int(getattr(self.config, "response_length", available_tokens))
            max_tokens = min(available_tokens, max(1, default_response_len))
        else:
            try:
                requested_max_tokens = int(requested_max_tokens)
            except (TypeError, ValueError):
                requested_max_tokens = available_tokens
            max_tokens = min(available_tokens, max(1, requested_max_tokens))
        lora_request_from_sampling_params = sampling_params.pop("lora_request", None)
        incoming_lora_name = None
        incoming_lora_int_id = None
        if isinstance(lora_request_from_sampling_params, dict):
            incoming_lora_name = lora_request_from_sampling_params.get("lora_name")
            incoming_lora_int_id = lora_request_from_sampling_params.get("lora_int_id")
        elif lora_request_from_sampling_params is not None:
            incoming_lora_name = getattr(lora_request_from_sampling_params, "lora_name", None)
            incoming_lora_int_id = getattr(lora_request_from_sampling_params, "lora_int_id", None)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )

        ptr_debug_limit = max(0, int(os.environ.get("RLLM_PTR_DEBUG_MAX_REQUESTS", "512")))
        ptr_debug_idx = int(getattr(self, "_ptr_debug_request_count", 0))
        ptr_debug = ptr_debug_limit > 0 and ptr_debug_idx < ptr_debug_limit
        self._ptr_debug_request_count = ptr_debug_idx + 1
        debug = self._debug_lora or ptr_debug

        def _debug_print(message: str) -> None:
            if debug:
                print(message, flush=True)

        trace_engine = _is_vllm_engine_trace_enabled()
        if trace_engine:
            print(
                "[Debug][vLLM.generate.request] "
                f"request_id={request_id} vllm engine: id={id(self.engine)}, "
                f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, pid={os.getpid()} "
                f"prompt_len={prompt_len} requested_max_tokens={requested_max_tokens} "
                f"incoming_lora_name={incoming_lora_name} incoming_lora_int_id={incoming_lora_int_id}",
                flush=True,
            )

        _debug_print(
            "[PTR_DEBUG][vLLMHttpServer.generate] "
            f"idx={ptr_debug_idx} request_id={request_id} prompt_len={prompt_len} "
            f"max_tokens={max_tokens} temperature={raw_sampling_params.get('temperature')} "
            f"top_p={raw_sampling_params.get('top_p')} n={raw_sampling_params.get('n')}"
        )

        verify_list_loras_per_request = _env_bool(
            "RLLM_VLLM_VERIFY_LIST_LORAS_PER_REQUEST",
            False,
        )
        list_loras_timeout_s = _env_float("RLLM_VLLM_LIST_LORAS_TIMEOUT_S", 5.0)
        strict_missing = _env_bool("RLLM_VLLM_STRICT_LORA_MISSING", False)
        strict_response_lora_match = _env_bool(
            "RLLM_VLLM_STRICT_RESPONSE_LORA_MATCH",
            False,
        )
        allow_unloaded_request = _env_bool(
            "RLLM_VLLM_ALLOW_UNLOADED_LORA_REQUEST",
            False,
        )

        async def _safe_list_loras(timeout_s: float) -> set[int]:
            try:
                lora_list = await asyncio.wait_for(self.engine.list_loras(), timeout=timeout_s)
                return set(lora_list)
            except asyncio.TimeoutError:
                logger.warning(
                    "list_loras timed out after %.2fs during request_id=%s",
                    timeout_s,
                    request_id,
                )
            except Exception as exc:
                logger.warning("list_loras failed during request_id=%s: %s", request_id, exc)
            return set()

        _debug_print(
            f"[DEBUG][vLLMHttpServer.generate] lora_request_from_sampling_params: {lora_request_from_sampling_params}"
        )
        _debug_print(
            f"[DEBUG][vLLMHttpServer.generate] self.model_config.lora_rank: {self.model_config.lora_rank}"
        )
        if lora_request_from_sampling_params is not None and not isinstance(lora_request_from_sampling_params, LoRARequest):
            if isinstance(lora_request_from_sampling_params, dict):
                lora_int_id = lora_request_from_sampling_params.get("lora_int_id", None)
                _debug_print(f"[DEBUG][vLLMHttpServer.generate] in line 421, lora_int_id: {lora_int_id}")
                lora_request = LoRARequest(**lora_request_from_sampling_params)
                if verify_list_loras_per_request:
                    loras = await _safe_list_loras(timeout_s=list_loras_timeout_s)
                    _debug_print(f"[DEBUG][vLLMHttpServer.generate] in line 423, list_loras: {loras}")
                    lora_loaded = (lora_int_id is not None) and (lora_int_id in loras)
                    _debug_print(f"[DEBUG][vLLMHttpServer.generate] in line 426, lora_loaded: {lora_loaded}")
                    if not lora_loaded:
                        if strict_missing:
                            raise RuntimeError(
                                "LoRA request adapter is not listed as loaded on this vLLM replica "
                                f"(request={lora_request_from_sampling_params}, loaded={sorted(loras)})."
                            )
                        if not allow_unloaded_request:
                            _debug_print(
                                "[DEBUG][vLLMHttpServer.generate] in line 431 "
                                "(adapter not in list_loras; dropping LoRARequest)"
                            )
                            lora_request = None
            else:
                _debug_print(f"[DEBUG][vLLMHttpServer.generate] in line 434")
                raise TypeError(f"Unsupported lora_request type: {type(lora_request_from_sampling_params)}")
        elif isinstance(lora_request_from_sampling_params, LoRARequest):
            lora_request = lora_request_from_sampling_params
        else:
            _debug_print("[DEBUG][vLLMHttpServer.generate] no explicit lora_request")
            lora_request = None
        _debug_print(f"[DEBUG][vLLMHttpServer.generate] lora_request: {lora_request}")
        requested_lora_name = getattr(lora_request, "lora_name", None) if lora_request is not None else None
        requested_lora_int_id = getattr(lora_request, "lora_int_id", None) if lora_request is not None else None
        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        response_lora = getattr(final_res, "lora_request", None)
        final_lora_name = (
            getattr(response_lora, "lora_name", None)
            if response_lora is not None
            else requested_lora_name
        )
        final_lora_int_id = (
            getattr(response_lora, "lora_int_id", None)
            if response_lora is not None
            else requested_lora_int_id
        )
        if incoming_lora_int_id is not None and final_lora_int_id is not None:
            try:
                incoming_lora_int_id = int(incoming_lora_int_id)
                final_lora_int_id = int(final_lora_int_id)
            except (TypeError, ValueError):
                pass
            if incoming_lora_int_id != final_lora_int_id:
                message = (
                    "LoRA mismatch between request and response "
                    f"(request_id={request_id}, incoming={incoming_lora_int_id}, final={final_lora_int_id})"
                )
                if strict_response_lora_match:
                    raise RuntimeError(message)
                logger.warning(message)
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        if trace_engine:
            log_prob_count = len(log_probs) if log_probs is not None else 0
            log_likelihood_sum = float(sum(log_probs)) if log_probs else None
            print(
                "[Debug][vLLM.generate.response] "
                f"request_id={request_id} vllm engine: id={id(self.engine)}, "
                f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, pid={os.getpid()} "
                f"response_tokens={len(token_ids)} "
                f"log_likelihood_tokens={log_prob_count} log_likelihood_sum={log_likelihood_sum} "
                f"incoming_lora_name={incoming_lora_name} incoming_lora_int_id={incoming_lora_int_id} "
                f"final_lora_name={final_lora_name} final_lora_int_id={final_lora_int_id}",
                flush=True,
            )
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        set_sleep_signal(0)
        if not self._sleep_wake_enabled():
            logger.info(
                "[wake_guard] skip wake_up (replica_rank=%s, node_rank=%s, disabled=%s, runtime_disabled=%s)",
                self.replica_rank,
                self.node_rank,
                self._disable_sleep_wake,
                self._sleep_wake_runtime_disabled,
            )
            return
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            try:
                await self._await_with_timeout(
                    asyncio.gather(*[worker.wake_up.remote() for worker in self.workers]),
                    self._wake_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[wake_guard] hybrid wake_up timed out after %.2fs (replica_rank=%s, node_rank=%s)",
                    self._wake_timeout_s,
                    self.replica_rank,
                    self.node_rank,
                )
                if self._disable_sleep_wake_on_timeout:
                    self._sleep_wake_runtime_disabled = True
                    logger.error(
                        "[wake_guard] disabling future sleep/wake after timeout "
                        "(replica_rank=%s, node_rank=%s)",
                        self.replica_rank,
                        self.node_rank,
                    )
                    return
                raise
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            if self.node_rank == 0:
                try:
                    await self._await_with_timeout(
                        self.engine.wake_up(tags=["kv_cache", "weights"]),
                        self._wake_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[wake_guard] wake_up timed out after %.2fs (replica_rank=%s, node_rank=%s)",
                        self._wake_timeout_s,
                        self.replica_rank,
                        self.node_rank,
                    )
                    if self._disable_sleep_wake_on_timeout:
                        self._sleep_wake_runtime_disabled = True
                        logger.error(
                            "[wake_guard] disabling future sleep/wake after timeout "
                            "(replica_rank=%s, node_rank=%s)",
                            self.replica_rank,
                            self.node_rank,
                        )
                        return
                    raise
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if not self._sleep_wake_enabled():
            logger.info(
                "[wake_guard] skip sleep (replica_rank=%s, node_rank=%s, disabled=%s, runtime_disabled=%s)",
                self.replica_rank,
                self.node_rank,
                self._disable_sleep_wake,
                self._sleep_wake_runtime_disabled,
            )
            return
        set_sleep_signal(1)
        if self.rollout_mode == RolloutMode.HYBRID:
            try:
                if self.node_rank == 0:
                    await self._await_with_timeout(
                        self.engine.reset_prefix_cache(),
                        self._sleep_timeout_s,
                    )
                await self._await_with_timeout(
                    asyncio.gather(*[worker.sleep.remote() for worker in self.workers]),
                    self._sleep_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[wake_guard] hybrid sleep timed out after %.2fs (replica_rank=%s, node_rank=%s)",
                    self._sleep_timeout_s,
                    self.replica_rank,
                    self.node_rank,
                )
                if self._disable_sleep_wake_on_timeout:
                    self._sleep_wake_runtime_disabled = True
                    logger.error(
                        "[wake_guard] disabling future sleep/wake after timeout "
                        "(replica_rank=%s, node_rank=%s)",
                        self.replica_rank,
                        self.node_rank,
                    )
                    return
                raise
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                try:
                    await self._await_with_timeout(
                        self.engine.reset_prefix_cache(),
                        self._sleep_timeout_s,
                    )
                    await self._await_with_timeout(
                        self.engine.sleep(level=1),
                        self._sleep_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "[wake_guard] sleep timed out after %.2fs (replica_rank=%s, node_rank=%s)",
                        self._sleep_timeout_s,
                        self.replica_rank,
                        self.node_rank,
                    )
                    if self._disable_sleep_wake_on_timeout:
                        self._sleep_wake_runtime_disabled = True
                        logger.error(
                            "[wake_guard] disabling future sleep/wake after timeout "
                            "(replica_rank=%s, node_rank=%s)",
                            self.replica_rank,
                            self.node_rank,
                        )
                        return
                    raise
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()


@ray.remote(num_cpus=1)
class vLLMHttpServer(vLLMHttpServerBase):
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)


_rollout_worker_actor_cls = ray.remote(vLLMAsyncRollout)


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = vLLMHttpServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server whether it's single or multi nodes.
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # create server actor in each node with node affinity
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])


def _get_rollout_lora_limits(rollout_config: RolloutConfig, model_config: HFModelConfig) -> tuple[int | None, int | None]:
    max_loras = rollout_config.get("max_loras", None)
    max_lora_rank = rollout_config.get("max_lora_rank", None)
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

    if max_loras is None and model_config.lora_rank > 0:
        max_loras = 1
    if max_lora_rank is None and model_config.lora_rank > 0:
        max_lora_rank = get_vllm_max_lora_rank(model_config.lora_rank)

    return max_loras, max_lora_rank




def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
