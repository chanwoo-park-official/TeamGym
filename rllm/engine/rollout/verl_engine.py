import asyncio
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.utils.lora_utils import build_lora_name_map, get_agent_names
from rllm.workflows import TerminationEvent, TerminationReason
import os
hydra_run_dir = os.environ.get('HYDRA_RUN_DIR', '/tmp/default')
os.makedirs(hydra_run_dir, exist_ok=True)
BASE_LORA_NAME = "__base__"


def _get_section(config, name: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _get_flag(config, section: str, name: str, default: bool = False) -> bool:
    section_cfg = _get_section(config, section)
    if section_cfg is None:
        return default
    if isinstance(section_cfg, dict):
        return bool(section_cfg.get(name, default))
    return bool(getattr(section_cfg, name, default))


def _is_debug_lora_enabled(config) -> bool:
    return _get_flag(config, "rllm", "debug_lora", False) or _get_flag(
        config, "verl", "debug_lora", False
    )


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        disable_thinking = config.get("rllm", {}).get("disable_thinking", False)
        self.chat_parser = ChatTemplateParser.get_parser(
            tokenizer, processor=processor, disable_thinking=disable_thinking
        )
        self._chat_parser_by_thinking = {
            True: ChatTemplateParser.get_parser(
                tokenizer, processor=processor, disable_thinking=False
            ),
            False: ChatTemplateParser.get_parser(
                tokenizer, processor=processor, disable_thinking=True
            ),
        }

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)
        self._debug_lora = _is_debug_lora_enabled(config)
        self._debug_balance_batch = _get_flag(config, "rllm", "debug_balance_batch", False)
        self._time_debug = _get_flag(config, "rllm", "time_debug", False)
        rllm_cfg = getattr(config, "rllm", None)
        llm_timeout_s = getattr(rllm_cfg, "llm_timeout_s", None) if rllm_cfg is not None else None
        if llm_timeout_s is not None:
            try:
                llm_timeout_s = float(llm_timeout_s)
            except (TypeError, ValueError):
                llm_timeout_s = None
        self.llm_timeout_s = llm_timeout_s if llm_timeout_s and llm_timeout_s > 0 else None

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        if self._debug_balance_batch:
            print(f"train_sampling_params: {self.train_sampling_params}")
            print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl
        self._ptr_debug_max_requests = max(0, int(os.environ.get("RLLM_PTR_DEBUG_MAX_REQUESTS", "512")))
        self._ptr_debug_request_count = 0

    def get_chat_parser(self, enable_thinking: bool | None):
        if enable_thinking is None:
            return self.chat_parser
        return self._chat_parser_by_thinking[bool(enable_thinking)]

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        enable_thinking = kwargs.pop("enable_thinking", None)
        chat_parser = self.get_chat_parser(enable_thinking)

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop(
            "max_tokens",
            sampling_params.pop("max_new_tokens", self.max_response_length),
        )
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = int(self.max_response_length)
        max_tokens = max(1, max_tokens)
        # Forward per-request max token budget to rollout server for server-side cutoff.
        sampling_params["max_tokens"] = max_tokens

        if self._debug_lora and self._ptr_debug_request_count < self._ptr_debug_max_requests:
            phase = "val" if (self.validate or validate) else "train"
            lora_request = sampling_params.get("lora_request")
            lora_int_id = None
            if isinstance(lora_request, dict):
                lora_int_id = lora_request.get("lora_int_id")
            print(
                "[PTR_DEBUG][VerlEngine.request] "
                f"idx={self._ptr_debug_request_count} request_id={application_id} phase={phase} "
                f"max_tokens={max_tokens} temperature={sampling_params.get('temperature')} "
                f"top_p={sampling_params.get('top_p')} n={sampling_params.get('n')} "
                f"lora_int_id={lora_int_id}",
                flush=True,
            )
            self._ptr_debug_request_count += 1

        prompt = chat_parser.parse(
            messages,
            add_generation_prompt=True,
            is_first_msg=True,
            tools=tools,
            accumulate_reasoning=accumulate_reasoning,
        )
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]

        if any(msg.get("images", None) is not None and msg["role"] == "user" for msg in messages) and self.processor is not None:
            image_data = self.chat_parser.process_image_data(messages)  # list[PIL.Image.Image]
            model_inputs = self.processor(text=[prompt], images=image_data)
            prompt_ids = model_inputs.pop("input_ids")[0]  # list[int]
            model_inputs.pop("attention_mask")
            multi_modal_inputs = dict(model_inputs)
        else:
            image_data = None
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        try:
            if self.llm_timeout_s is not None:
                token_output: TokenOutput = await asyncio.wait_for(
                    self.server_manager.generate(
                        request_id=application_id,
                        prompt_ids=request_prompt_ids,
                        image_data=image_data,
                        sampling_params=sampling_params,
                    ),
                    timeout=self.llm_timeout_s,
                )  # type: ignore
            else:
                token_output = await self.server_manager.generate(
                    request_id=application_id,
                    prompt_ids=request_prompt_ids,
                    image_data=image_data,
                    sampling_params=sampling_params,
                )  # type: ignore
        except asyncio.TimeoutError as exc:
            if self._time_debug:
                print(
                    f"[Debug][VerlEngine] LLM request timed out after {self.llm_timeout_s}s "
                    f"(application_id={application_id})"
                )
            raise TerminationEvent(TerminationReason.TIMEOUT) from exc
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = chat_parser.parse_completion(completion_ids)

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])
        
    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])


_ORIG_VERL_INIT = VerlEngine.__init__
_ORIG_VERL_GET = VerlEngine.get_model_response


def _patched_verl_init(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
    _ORIG_VERL_INIT(self, config, rollout_manager, tokenizer, processor=processor, **kwargs)
    rllm_cfg = getattr(config, "rllm", None)
    multi_agent = bool(getattr(rllm_cfg, "multi_agent", False)) if rllm_cfg is not None else False
    if multi_agent:
        try:
            names = get_agent_names(config)
        except ValueError:
            names = []
    else:
        names = []
    self._lora_name_to_id = build_lora_name_map(names) if names else {}
    debug = _is_debug_lora_enabled(config)
    if self._lora_name_to_id and debug:
        print(f"[VerlEngine] LoRA adapter mapping: {self._lora_name_to_id}")


def _get_lora_request_by_name(self, lora_name: str) -> dict | None:
    if lora_name in (None, BASE_LORA_NAME):
        return None
    if lora_name == "default" and lora_name not in self._lora_name_to_id:
        result = {
            "lora_name": VLLM_LORA_NAME,
            "lora_int_id": VLLM_LORA_INT_ID,
            "lora_path": VLLM_LORA_PATH,
        }
        return result
    if lora_name not in self._lora_name_to_id:
        raise ValueError(
            f"Unknown lora_name={lora_name}. Known adapters: {list(self._lora_name_to_id.keys())}"
        )
    result = {
        "lora_name": lora_name,
        "lora_int_id": self._lora_name_to_id[lora_name],
        "lora_path": f"{hydra_run_dir}/{lora_name}",
    }
    return result


async def _patched_verl_get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
    lora_name = kwargs.pop("lora_name", None)
    if lora_name is not None:
        lora_request = _get_lora_request_by_name(self, lora_name)
        if lora_request is not None:
            kwargs["lora_request"] = lora_request
    return await _ORIG_VERL_GET(self, messages, **kwargs)


VerlEngine.__init__ = _patched_verl_init
VerlEngine.get_model_response = _patched_verl_get_model_response
VerlEngine._get_lora_request_by_name = _get_lora_request_by_name
