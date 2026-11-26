"""Timing tracking mixin for workflows to measure LLM, environment, and total execution time."""

import os
import time
from datetime import datetime, timezone
from typing import Any


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _time_debug_from_env() -> bool:
    raw = os.getenv("RLLM_TIME_DEBUG")
    if raw is None:
        raw = os.getenv("RLLM_DEBUG_TIMING")
    return _truthy(raw)


class TimingTrackingMixin:
    """Mixin to add timing tracking to workflows."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timing_data = {
            "llm_time": 0.0,
            "env_time": 0.0,
            "reward_time": 0.0,
            "total_time": 0.0,
            "start_time": None,
            "end_time": None,
        }
        self._timing_breakdown = None
        self._timing_counts = None
        # Track per-step timing
        self._step_timings = []
        self._current_step_timing = None
        self._time_debug = _time_debug_from_env()
        self._debug_timing = self._time_debug
        self._last_env_func = None
        self._last_env_mode = None
        self._last_env_error = None
        self._last_llm_error = None

    def start_timing(self):
        """Start timing the episode."""
        self._timing_data["start_time"] = time.time()  # Keep for calculation
        self._timing_data["start_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._timing_data["end_time"] = None
        self._timing_data["end_timestamp"] = None
        self._timing_data["llm_time"] = 0.0
        self._timing_data["env_time"] = 0.0
        self._timing_data["reward_time"] = 0.0
        self._step_timings = []
        self._current_step_timing = None
        if self._timing_breakdown is not None:
            self._timing_breakdown.clear()
        if self._timing_counts is not None:
            self._timing_counts.clear()

    def _start_new_step_timing(self):
        """Start timing for a new step."""
        if self._current_step_timing is not None:
            # Finish the previous step timing BEFORE appending
            self._finish_current_step_timing()
            self._step_timings.append(self._current_step_timing)

        self._current_step_timing = {
            "llm_time": 0.0,
            "env_time": 0.0,
            "step_start_time": time.time(),  # Keep for calculation
            "step_start_timestamp": datetime.now(timezone.utc).isoformat(),
            "step_end_time": None,
            "step_end_timestamp": None,
        }

    def _finish_current_step_timing(self):
        """Finish timing for the current step."""
        if self._current_step_timing is not None:
            self._current_step_timing["step_end_time"] = time.time()
            self._current_step_timing["step_end_timestamp"] = datetime.now(timezone.utc).isoformat()

    async def timed_llm_call(self, *args, **kwargs):
        """Wrapper for LLM calls with timing.

        Returns:
            ModelOutput from the rollout engine
        """
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None and max_tokens <= 0:
            print(f"[sat_warn] max_tokens <= 0 ({max_tokens}), clamping to 1")
            kwargs["max_tokens"] = 1

        # Start a new step timing when we make an LLM call
        self._start_new_step_timing()

        timing_label = kwargs.pop("timing_label", None)
        uid = kwargs.pop("uid", getattr(self, "uid", None))
        llm_mode = self._timing_env_mode(uid)
        application_id = kwargs.get("application_id")
        if self._time_debug:
            label_info = f" label={timing_label}" if timing_label else ""
            print(
                f"[Debug][Timing] LLM start uid={uid} mode={llm_mode} application_id={application_id}{label_info}"
            )
        start = time.time()
        try:
            result = await self.rollout_engine.get_model_response(*args, **kwargs)
        except Exception as exc:
            reason = getattr(exc, "reason", None)
            if reason is not None:
                reason_value = getattr(reason, "value", reason)
                self._last_llm_error = f"{type(exc).__name__}:{reason_value}"
            else:
                self._last_llm_error = type(exc).__name__
            if self._time_debug:
                print(
                    f"[Debug][Timing] LLM error uid={uid} mode={llm_mode} application_id={application_id} "
                    f"error={type(exc).__name__}"
                )
            raise
        duration = time.time() - start

        self._timing_data["llm_time"] += duration
        if self._current_step_timing is not None:
            self._current_step_timing["llm_time"] += duration

        if timing_label and self._timing_breakdown is not None:
            self._timing_breakdown[timing_label] = (
                self._timing_breakdown.get(timing_label, 0.0) + duration
            )
            if self._timing_counts is not None:
                self._timing_counts[timing_label] = (
                    self._timing_counts.get(timing_label, 0) + 1
                )

        if self._time_debug:
            label_info = f" label={timing_label}" if timing_label else ""
            print(
                f"[Debug][Timing] LLM done uid={uid} mode={llm_mode} application_id={application_id}{label_info} "
                f"duration_s={duration:.2f}"
            )
        return result

    def _timing_env_mode(self, uid: str | None) -> str:
        if uid is None:
            return "unknown"
        uid_str = str(uid)
        if ":tree" in uid_str:
            return "branch"
        return "base"

    async def timed_env_call(self, func, *args, **kwargs):
        """Wrapper for environment calls with timing.

        Args:
            func: The function to call (typically env.reset or env.step)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function call
        """
        timing_label = kwargs.pop("timing_label", None)
        uid = kwargs.get("uid", getattr(self, "uid", None))
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
        env_mode = self._timing_env_mode(uid)
        self._last_env_func = func_name
        self._last_env_mode = env_mode
        if self._time_debug:
            label_info = f" label={timing_label}" if timing_label else ""
            print(
                f"[Debug][Timing] ENV start func={func_name} uid={uid} mode={env_mode}{label_info}"
            )
        start = time.time()
        try:
            result = await self.run_in_executor(func, *args, **kwargs)
        except Exception as exc:
            self._last_env_error = type(exc).__name__
            if self._time_debug:
                print(
                    f"[Debug][Timing] ENV error func={func_name} uid={uid} mode={env_mode} "
                    f"error={type(exc).__name__}"
                )
            raise
        duration = time.time() - start

        self._timing_data["env_time"] += duration
        if self._current_step_timing is not None:
            self._current_step_timing["env_time"] += duration

        if timing_label and self._timing_breakdown is not None:
            self._timing_breakdown[timing_label] = (
                self._timing_breakdown.get(timing_label, 0.0) + duration
            )
            if self._timing_counts is not None:
                self._timing_counts[timing_label] = (
                    self._timing_counts.get(timing_label, 0) + 1
                )

        if self._time_debug:
            label_info = f" label={timing_label}" if timing_label else ""
            print(
                f"[Debug][Timing] ENV done func={func_name} uid={uid} mode={env_mode}{label_info} "
                f"duration_s={duration:.2f}"
            )
        return result

    def add_reward_time(self, duration: float):
        """Add reward computation time.

        Args:
            duration: Time spent computing rewards
        """
        self._timing_data["reward_time"] += duration

    def finalize_timing(self):
        """Finalize timing calculations.

        Returns:
            Dictionary with timing information including start_timestamp and end_timestamp as ISO 8601 strings
        """
        # Finish the last step timing if it exists
        if self._current_step_timing is not None:
            self._finish_current_step_timing()
            self._step_timings.append(self._current_step_timing)
            self._current_step_timing = None

        # Set end time for the episode
        if self._timing_data["start_time"] is not None:
            self._timing_data["end_time"] = time.time()
            self._timing_data["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            self._timing_data["total_time"] = self._timing_data["end_time"] - self._timing_data["start_time"]

        return {
            "start_timestamp": self._timing_data["start_timestamp"],  # ISO 8601 timestamp string
            "end_timestamp": self._timing_data["end_timestamp"],  # ISO 8601 timestamp string
            "llm_time": self._timing_data["llm_time"],
            "env_time": self._timing_data["env_time"],
            "reward_time": self._timing_data["reward_time"],
            "total_time": self._timing_data["total_time"],
        }

    def postprocess_episode(self, episode, termination_reason=None, error=None):
        """Override to add timing metrics to episode.

        This should be called from the subclass's postprocess_episode method.
        """
        # Get timing data before calling parent
        timing_metrics = self.finalize_timing()

        # Call parent's postprocess if it exists
        if hasattr(super(), "postprocess_episode"):
            episode = super().postprocess_episode(episode, termination_reason, error)

        # Add timing to episode info
        episode.info["timing"] = timing_metrics

        # Add per-trajectory metrics
        for trajectory in episode.trajectories:
            trajectory.info["num_steps"] = len(trajectory.steps)

            # Calculate trajectory-level timing (start and end timestamps)
            trajectory_start = None
            trajectory_end = None

            if trajectory.steps:
                # Get start time from first step
                if self._step_timings:
                    trajectory_start = self._step_timings[0].get("step_start_timestamp")
                    trajectory_end = self._step_timings[-1].get("step_end_timestamp")
                elif timing_metrics.get("start_timestamp") is not None:
                    # Fallback to episode timing if no step timings available
                    trajectory_start = timing_metrics["start_timestamp"]
                    trajectory_end = timing_metrics["end_timestamp"]

            # Add trajectory-level timing with start and end timestamps as ISO 8601 strings
            trajectory.info["timing"] = {
                "start_timestamp": trajectory_start,  # ISO 8601 timestamp string
                "end_timestamp": trajectory_end,  # ISO 8601 timestamp string
                "llm_time": timing_metrics["llm_time"],
                "env_time": timing_metrics["env_time"],
                "reward_time": timing_metrics["reward_time"],
                "total_time": timing_metrics["total_time"],
            }

            # Add per-step timing to each step (with ISO 8601 timestamps)
            for i, step in enumerate(trajectory.steps):
                if i < len(self._step_timings):
                    step_timing = self._step_timings[i]
                    step.info["timing"] = {
                        "start_timestamp": step_timing.get("step_start_timestamp"),  # ISO 8601 timestamp string
                        "end_timestamp": step_timing.get("step_end_timestamp"),  # ISO 8601 timestamp string
                        "llm_time": step_timing.get("llm_time", 0.0),
                        "env_time": step_timing.get("env_time", 0.0),
                    }
                else:
                    # No timing data available for this step
                    step.info["timing"] = {
                        "start_timestamp": None,
                        "end_timestamp": None,
                        "llm_time": 0.0,
                        "env_time": 0.0,
                    }

        return episode

    def reset(self, task: Any = None, uid: str | None = None):
        """Override reset to start timing.

        Subclasses should call this if they override reset.
        """
        # Start timing when resetting
        self.start_timing()

        # Call parent's reset if it exists
        if hasattr(super(), "reset"):
            return super().reset(task, uid)
