import copy
import multiprocessing as mp
import os
import time
import traceback
import yaml

from alfworld.agents.environment import get_environment

from rllm.environments.base.base_env import BaseEnv


def _env_flag(primary: str, legacy: str | None = None) -> bool:
    raw = os.getenv(primary)
    if raw is None and legacy is not None:
        raw = os.getenv(legacy)
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


class _ALFWorldCore:
    _PATH_CACHE = {}

    def __init__(self, **kwargs):
        self.index = kwargs.pop("index", 0)
        self.dataset_split = kwargs.pop("dataset_split", "eval_out_of_distribution")
        self._debug_add_env = _env_flag("RLLM_DEBUG_ADD_ENV", "RLLM_DEBUG_ENV")
        self._debug_add_env_verbose = _env_flag(
            "RLLM_DEBUG_ADD_ENV_VERBOSE", "RLLM_DEBUG_ENV_VERBOSE"
        )
        self._debug_stats = None
        if self._debug_add_env:
            self._debug_stats = {
                "dataset_init_s": 0.0,
                "dataset_init_count": 0,
                "init_env_s": 0.0,
                "init_env_count": 0,
                "reset_env_s": 0.0,
                "reset_env_count": 0,
                "reset_dataset_s": 0.0,
                "reset_dataset_count": 0,
                "reset_total_s": 0.0,
                "reset_total_count": 0,
                "parse_obs_s": 0.0,
                "parse_obs_count": 0,
                "parse_info_s": 0.0,
                "parse_info_count": 0,
                "step_s": 0.0,
                "step_count": 0,
                "search_s": 0.0,
                "search_count": 0,
                "cache_hit": 0,
                "cache_miss": 0,
                "reset_env_reuse_count": 0,
                "reset_env_rebuild_count": 0,
                "reset_env_rebuild_missing_count": 0,
                "reset_env_rebuild_dataset_change_count": 0,
                "reset_env_rebuild_retry_count": 0,
            }

        self.env = None
        self.init_dataset(dataset_split=self.dataset_split, index=self.index)
        self._rebuild_env(reason="initialization")
        self.goal_object = None

    def _log_debug(self, message: str) -> None:
        if not self._debug_add_env:
            return
        context = getattr(self, "_debug_context", None)
        if context:
            message = f"{message} context={context}"
        print(f"[Debug][ALFWorldEnv] {message}", flush=True)

    def search_data_path(self, base_path: str, index: int) -> str:
        base_path = os.path.expandvars(base_path)
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Dataset path not found: {base_path}")
        data_path_list = _ALFWorldCore._PATH_CACHE.get(base_path)
        if data_path_list is None:
            scan_start = time.perf_counter() if self._debug_stats is not None else None
            data_path_list = [
                os.path.join(root, dirname)
                for root, dirnames, _ in os.walk(base_path)
                for dirname in dirnames
                if dirname.startswith("trial_") and os.path.isfile(os.path.join(root, dirname, "game.tw-pddl"))
            ]
            data_path_list.sort()
            _ALFWorldCore._PATH_CACHE[base_path] = data_path_list
            if self._debug_stats is not None and scan_start is not None:
                scan_s = time.perf_counter() - scan_start
                self._debug_stats["search_s"] += scan_s
                self._debug_stats["search_count"] += 1
                self._debug_stats["cache_miss"] += 1
                self._log_debug(
                    f"scan base_path={base_path} trials={len(data_path_list)} duration_s={scan_s:.3f}"
                )
        elif self._debug_stats is not None:
            self._debug_stats["cache_hit"] += 1
        assert index < len(data_path_list), f"Index {index} out of range for dataset with {len(data_path_list)} examples."
        return data_path_list[index]

    def init_dataset(self, dataset_split: str, index: int | None) -> None:
        index = index if index is not None else self.index
        if not hasattr(self, 'dataset') or self.dataset_split != dataset_split or self.index != index:
            init_start = time.perf_counter() if self._debug_stats is not None else None
            with open(__file__.replace("alfworld_env.py", "base_config.yaml")) as reader:
                config = yaml.safe_load(reader)

            if dataset_split == "train": 
                config["dataset"]["data_path"] = self.search_data_path(config["dataset"]["data_path"], index)
            elif dataset_split == "eval_in_distribution":
                config["dataset"]["eval_id_data_path"] = self.search_data_path(config["dataset"]["eval_id_data_path"], index)
            elif dataset_split == "eval_out_of_distribution":
                config["dataset"]["eval_ood_data_path"] = self.search_data_path(config["dataset"]["eval_ood_data_path"], index)
            else:
                raise ValueError(f"Unknown dataset split: {dataset_split}")
            self.dataset = get_environment(config['env']['type'])(config, train_eval=dataset_split)
            self.dataset_split = dataset_split
            self.index = index
            if self._debug_stats is not None and init_start is not None:
                init_s = time.perf_counter() - init_start
                self._debug_stats["dataset_init_s"] += init_s
                self._debug_stats["dataset_init_count"] += 1
                self._log_debug(
                    f"init_dataset split={dataset_split} index={index} "
                    f"env_type={config['env']['type']} duration_s={init_s:.3f}"
                )

    def parse_info(self, info: dict) -> dict:
        admissible_commands = info.get("admissible_commands", [[]])[0]
        if "help" in admissible_commands:
            admissible_commands.remove("help")
        parsed_info = copy.deepcopy(info)
        parsed_info["admissible_commands"] = admissible_commands
        return parsed_info

    def _close_env(self) -> None:
        if self.env is None:
            return
        try:
            self.env.close()
        except Exception:
            pass
        self.env = None

    def _rebuild_env(self, *, reason: str) -> float:
        self._close_env()
        init_env_start = time.perf_counter() if self._debug_stats is not None else None
        self.env = self.dataset.init_env(batch_size=1)
        duration_s = 0.0
        if self._debug_stats is not None and init_env_start is not None:
            duration_s = time.perf_counter() - init_env_start
            self._debug_stats["init_env_s"] += duration_s
            self._debug_stats["init_env_count"] += 1
            if self._debug_add_env_verbose:
                self._log_debug(
                    f"init_env reason={reason} split={self.dataset_split} "
                    f"index={self.index} duration_s={duration_s:.3f}"
                )
        return duration_s

    def reset(self, task: dict = None) -> tuple[dict, dict]:
        """
        Reset the environment to start a new episode.
        Args:
            task (dict, optional): Task information to set up the environment. Defaults to None.
        Returns:
            tuple: (initial observation, info)
        """
        task = task or {}
        total_start = time.perf_counter() if self._debug_stats is not None else None
        target_split = task.get("dataset_split", self.dataset_split)
        target_index = task.get("index", self.index)
        needs_env_rebuild = (
            self.env is None
            or target_split != self.dataset_split
            or target_index != self.index
        )
        dataset_start = time.perf_counter() if self._debug_stats is not None else None
        self.init_dataset(dataset_split=target_split, index=target_index)
        dataset_s = 0.0
        if self._debug_stats is not None and dataset_start is not None:
            dataset_s = time.perf_counter() - dataset_start
            self._debug_stats["reset_dataset_s"] += dataset_s
            self._debug_stats["reset_dataset_count"] += 1
        local_init_env_s = 0.0
        if needs_env_rebuild:
            reason = "env_missing" if self.env is None else "dataset_key_changed"
            if self._debug_stats is not None:
                self._debug_stats["reset_env_rebuild_count"] += 1
                if reason == "env_missing":
                    self._debug_stats["reset_env_rebuild_missing_count"] += 1
                else:
                    self._debug_stats["reset_env_rebuild_dataset_change_count"] += 1
            local_init_env_s += self._rebuild_env(reason=reason)
        elif self._debug_stats is not None:
            self._debug_stats["reset_env_reuse_count"] += 1
        reset_start = time.perf_counter() if self._debug_stats is not None else None
        assert self.env is not None
        try:
            obs, info = self.env.reset()
        except Exception:
            # Recover from transient env-side failures by rebuilding once and retrying.
            if self._debug_stats is not None:
                self._debug_stats["reset_env_rebuild_count"] += 1
                self._debug_stats["reset_env_rebuild_retry_count"] += 1
            local_init_env_s += self._rebuild_env(reason="reset_failed_retry")
            assert self.env is not None
            obs, info = self.env.reset()
        local_reset_s = 0.0
        if self._debug_stats is not None and reset_start is not None:
            local_reset_s = time.perf_counter() - reset_start
            self._debug_stats["reset_env_s"] += local_reset_s
            self._debug_stats["reset_env_count"] += 1
            if self._debug_add_env_verbose:
                self._log_debug(
                    f"reset split={self.dataset_split} index={self.index} duration_s={local_reset_s:.3f}"
                )
        parse_start = time.perf_counter() if self._debug_stats is not None else None
        obs = obs[0]
        obs = [o for o in obs.split("\n") if o.strip() != ""]
        self.goal_object = obs[2].split("Your task is to: ")[1].strip()
        obs = "\n".join(obs[:2]).strip()
        local_parse_obs_s = 0.0
        if self._debug_stats is not None and parse_start is not None:
            local_parse_obs_s = time.perf_counter() - parse_start
            self._debug_stats["parse_obs_s"] += local_parse_obs_s
            self._debug_stats["parse_obs_count"] += 1
        info_start = time.perf_counter() if self._debug_stats is not None else None
        parsed_info = self.parse_info(info)
        local_parse_info_s = 0.0
        if self._debug_stats is not None and info_start is not None:
            local_parse_info_s = time.perf_counter() - info_start
            self._debug_stats["parse_info_s"] += local_parse_info_s
            self._debug_stats["parse_info_count"] += 1
        if self._debug_stats is not None and total_start is not None:
            total_s = time.perf_counter() - total_start
            self._debug_stats["reset_total_s"] += total_s
            self._debug_stats["reset_total_count"] += 1
            if self._debug_add_env_verbose:
                self._log_debug(
                    "reset_breakdown "
                    f"dataset_s={dataset_s:.3f} init_env_s={local_init_env_s:.3f} "
                    f"env_reset_s={local_reset_s:.3f} parse_obs_s={local_parse_obs_s:.3f} "
                    f"parse_info_s={local_parse_info_s:.3f} total_s={total_s:.3f} "
                    f"split={self.dataset_split} index={self.index}"
                )
        return {
            "obs": obs,
            "last_action": None,
            "goal_object": self.goal_object,
        }, parsed_info

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Take an action in the environment.
        Args:
            action (str): The action to take.
        Returns:
            tuple: (observation, reward, done, info)
        """
        step_start = time.perf_counter() if self._debug_stats is not None else None
        obs, reward, done, info = self.env.step([action])
        if self._debug_stats is not None and step_start is not None:
            step_s = time.perf_counter() - step_start
            self._debug_stats["step_s"] += step_s
            self._debug_stats["step_count"] += 1
            if self._debug_add_env_verbose:
                self._log_debug(
                    f"duration_s={step_s:.3f} reward={reward[0]} done={done[0]}"
                )
        obs = obs[0]
        reward = reward[0]
        done = done[0]
        return {
            "obs": obs,
            "last_action": action,
            "goal_object": self.goal_object,
        }, reward, done, self.parse_info(info)

    @staticmethod
    def from_dict(env_info: dict) -> "_ALFWorldCore":
        return _ALFWorldCore(index=env_info["index"], dataset_split=env_info["dataset_split"])
    
    def close(self):
        if self._debug_add_env and self._debug_stats is not None:
            stats = self._debug_stats
            step_count = stats["step_count"]
            reset_count = stats["reset_env_count"]
            total_reset_count = stats["reset_total_count"]
            avg_step_s = stats["step_s"] / step_count if step_count else 0.0
            avg_reset_s = stats["reset_env_s"] / reset_count if reset_count else 0.0
            avg_reset_total_s = (
                stats["reset_total_s"] / total_reset_count if total_reset_count else 0.0
            )
            parse_obs_count = stats["parse_obs_count"]
            parse_info_count = stats["parse_info_count"]
            avg_parse_obs_s = (
                stats["parse_obs_s"] / parse_obs_count if parse_obs_count else 0.0
            )
            avg_parse_info_s = (
                stats["parse_info_s"] / parse_info_count if parse_info_count else 0.0
            )
            self._log_debug(
                "summary "
                f"split={self.dataset_split} index={self.index} "
                f"dataset_init_s={stats['dataset_init_s']:.3f} "
                f"init_env_s={stats['init_env_s']:.3f} "
                f"reset_env_s={stats['reset_env_s']:.3f} "
                f"reset_total_s={stats['reset_total_s']:.3f} "
                f"parse_obs_s={stats['parse_obs_s']:.3f} "
                f"parse_info_s={stats['parse_info_s']:.3f} "
                f"step_s={stats['step_s']:.3f} "
                f"steps={step_count} resets={reset_count} "
                f"avg_step_s={avg_step_s:.3f} avg_reset_s={avg_reset_s:.3f} "
                f"avg_reset_total_s={avg_reset_total_s:.3f} "
                f"avg_parse_obs_s={avg_parse_obs_s:.3f} avg_parse_info_s={avg_parse_info_s:.3f} "
                f"search_s={stats['search_s']:.3f} "
                f"cache_hit={stats['cache_hit']} cache_miss={stats['cache_miss']} "
                f"reset_env_reuse_count={stats['reset_env_reuse_count']} "
                f"reset_env_rebuild_count={stats['reset_env_rebuild_count']} "
                f"reset_env_rebuild_missing_count={stats['reset_env_rebuild_missing_count']} "
                f"reset_env_rebuild_dataset_change_count={stats['reset_env_rebuild_dataset_change_count']} "
                f"reset_env_rebuild_retry_count={stats['reset_env_rebuild_retry_count']}"
            )
        self._close_env()
        return


def _alfworld_worker_loop(conn, core_kwargs):
    core = None
    try:
        core = _ALFWorldCore(**core_kwargs)
        conn.send({"status": "ready"})
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                result = core.reset(task=payload)
                conn.send({"status": "ok", "data": result})
            elif cmd == "step":
                result = core.step(action=payload)
                conn.send({"status": "ok", "data": result})
            elif cmd == "close":
                core.close()
                conn.send({"status": "ok"})
                break
            else:
                raise ValueError(f"Unknown ALFWorld worker command: {cmd}")
    except EOFError:
        pass
    except Exception:
        err = traceback.format_exc()
        try:
            conn.send({"status": "error", "error": err})
        except Exception:
            pass
    finally:
        if core is not None:
            try:
                core.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ALFWorld Environment (process-isolated wrapper)
class ALFWorldEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.index = kwargs.pop("index", 0)
        self.dataset_split = kwargs.pop("dataset_split", "eval_out_of_distribution")
        timeout_s = kwargs.pop("timeout_s", None)
        if timeout_s is None:
            timeout_raw = os.getenv("RLLM_ALFWORLD_RPC_TIMEOUT_S")
            timeout_s = float(timeout_raw) if timeout_raw else None
        self.timeout_s = float(timeout_s) if timeout_s is not None else None
        self._debug_add_env = _env_flag("RLLM_DEBUG_ADD_ENV", "RLLM_DEBUG_ENV")
        self._debug_context = None
        self._core_kwargs = {
            "index": self.index,
            "dataset_split": self.dataset_split,
        }
        self._mp_ctx = mp.get_context("spawn")
        self.parent_conn = None
        self.child_conn = None
        self.process = None
        self._start_worker()

    def _log_ipc(self, message: str) -> None:
        if self._debug_add_env:
            print(f"[Debug][ALFWorldIPC] {message}", flush=True)

    def _close_pipe(self, pipe) -> None:
        if pipe is None:
            return
        try:
            pipe.close()
        except Exception:
            pass

    def _start_worker(self) -> None:
        self._close_pipe(self.parent_conn)
        self._close_pipe(self.child_conn)
        self.parent_conn, self.child_conn = self._mp_ctx.Pipe()
        self.process = self._mp_ctx.Process(
            target=_alfworld_worker_loop,
            args=(self.child_conn, dict(self._core_kwargs)),
        )
        self.process.start()
        self._close_pipe(self.child_conn)
        self.child_conn = None
        ready = self._recv_message("ready")
        if ready.get("status") != "ready":
            raise RuntimeError(f"ALFWorld worker failed to initialize: {ready}")
        self._log_ipc(f"worker started pid={self.process.pid}")

    def _terminate_worker(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self.process = None

    def _ensure_worker(self, *, allow_restart: bool = False) -> None:
        worker_ready = (
            self.process is not None
            and self.process.is_alive()
            and self.parent_conn is not None
        )
        if worker_ready:
            return
        if allow_restart:
            self._log_ipc("worker unavailable; restarting for reset")
            self._terminate_worker()
            self._start_worker()
            return
        raise RuntimeError("ALFWorld worker is not available; reset the environment first.")

    def _recv_message(self, purpose: str) -> dict:
        if self.parent_conn is None:
            raise RuntimeError("ALFWorld worker pipe is not available.")
        if self.timeout_s is not None and not self.parent_conn.poll(self.timeout_s):
            raise TimeoutError(
                f"Timeout after {self.timeout_s} seconds waiting for ALFWorld worker response ({purpose})."
            )
        return self.parent_conn.recv()

    def _request(self, cmd: str, payload=None, *, allow_restart: bool = False):
        self._ensure_worker(allow_restart=allow_restart)
        assert self.parent_conn is not None
        self.parent_conn.send((cmd, payload))
        response = self._recv_message(cmd)
        status = response.get("status")
        if status == "ok":
            return response.get("data")
        if status == "error":
            raise RuntimeError(
                f"ALFWorld worker error during '{cmd}'.\n{response.get('error', '(no traceback)')}"
            )
        raise RuntimeError(f"Unexpected ALFWorld worker response for '{cmd}': {response}")

    def reset(self, task: dict = None) -> tuple[dict, dict]:
        task = task or {}
        self.index = task.get("index", self.index)
        self.dataset_split = task.get("dataset_split", self.dataset_split)
        self._core_kwargs["index"] = self.index
        self._core_kwargs["dataset_split"] = self.dataset_split
        return self._request("reset", task, allow_restart=True)

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        return self._request("step", action)

    @staticmethod
    def from_dict(env_info: dict) -> "ALFWorldEnv":
        return ALFWorldEnv(
            index=env_info["index"],
            dataset_split=env_info["dataset_split"],
        )

    def close(self):
        if self.parent_conn is not None:
            try:
                self.parent_conn.send(("close", None))
                self._recv_message("close")
            except Exception:
                pass
        self._close_pipe(self.parent_conn)
        self.parent_conn = None
        self._close_pipe(self.child_conn)
        self.child_conn = None
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        self.process = None
        return super().close()


if __name__ == "__main__":
    env = ALFWorldEnv(index=0, dataset_split="eval_out_of_distribution")
    obs, info = env.reset()
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        print(f"Observation: {obs}")
        print(f"Admissible Commands: {info['admissible_commands']}")
        action_str = input("Enter your action: ")
        action = action_str
        print(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print(f"Episode finished with total reward: {total_reward}")
