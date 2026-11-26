import json
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback

import gymnasium as gym

from rllm.environments.base.base_env import BaseEnv


def _env_flag(primary: str, legacy: str | None = None) -> bool:
    raw = os.getenv(primary)
    if raw is None and legacy is not None:
        raw = os.getenv(legacy)
    if raw is None:
        return False
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


class BrowserGymEnv(BaseEnv):
    def __init__(
        self,
        env_id="browsergym/openended",
        seed=42,
        task=None,
        timeout: float | None = None,
        timeout_s: float | None = None,
        timeout_ms: float | None = None,
        **env_kwargs,
    ):
        self._debug_add_env = _env_flag("RLLM_DEBUG_ADD_ENV", "RLLM_DEBUG_ENV")
        self._respawn_on_reset_timeout = os.getenv("RLLM_ENV_RESET_RESPAWN", "").lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        self._respawn_on_reset = os.getenv("RLLM_ENV_RESET_ALWAYS_RESPAWN", "").lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        self.env_id = env_id
        self.task = task
        self.env_kwargs = env_kwargs
        self.seed = seed
        self._debug_context = None
        if timeout_s is not None:
            self.timeout = float(timeout_s)
        elif timeout_ms is not None:
            self.timeout = float(timeout_ms) / 1000.0
        else:
            self.timeout = float(timeout) if timeout is not None else None
        if self.timeout is None:
            self.timeout = 300.0
        self._mp_ctx = mp.get_context("spawn")
        self.parent_conn = None
        self.child_conn = None
        self.process = None
        self._start_worker()

    def _log_ipc(self, message: str) -> None:
        if self._debug_add_env:
            print(f"[Debug][EnvIPC] {message}", flush=True)

    def _summarize_task(self) -> dict | str | None:
        if not self.task:
            return None
        if isinstance(self.task, dict):
            summary = {}
            for key in ("env_id", "seed", "task_id", "id"):
                if key in self.task:
                    summary[key] = self.task[key]
            if summary:
                return summary
            return {"keys": sorted(self.task.keys())}
        return str(type(self.task))

    def _format_debug_context(self) -> str | None:
        if not self._debug_context:
            return None
        try:
            return json.dumps(self._debug_context, sort_keys=True, ensure_ascii=True)
        except Exception:
            return str(self._debug_context)

    def _start_worker(self) -> None:
        if self.parent_conn is not None:
            try:
                self.parent_conn.close()
            except Exception:
                pass
        if self.child_conn is not None:
            try:
                self.child_conn.close()
            except Exception:
                pass
        self.parent_conn, self.child_conn = self._mp_ctx.Pipe()
        self.process = self._mp_ctx.Process(
            target=BrowserGymEnv._worker,
            args=(self.child_conn, self.env_id, self.task, self.seed, self.env_kwargs),
        )
        self.process.start()
        self._log_ipc(f"worker started pid={self.process.pid}")

    def _terminate_worker(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self._log_ipc(f"terminating worker pid={self.process.pid}")
            self.process.terminate()
            self.process.join(5)
        self.process = None

    def _restart_worker(self) -> None:
        self._terminate_worker()
        self._start_worker()

    @staticmethod
    def _worker(conn, env_id, task, seed, env_kwargs):
        # Import browsergym modules in the worker process to register environments
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pass

        debug_add_env = _env_flag("RLLM_DEBUG_ADD_ENV", "RLLM_DEBUG_ENV")

        def _log_worker(message: str) -> None:
            if debug_add_env:
                print(f"[Debug][EnvWorker] {message}", flush=True)

        worker_pid = os.getpid()
        _log_worker(
            f"pid={worker_pid} env_id={env_id} task={'yes' if task else 'no'} seed={seed}"
        )

        def _handle_signal(signum, frame):
            _log_worker(f"signal received signum={signum}")
            if debug_add_env:
                traceback.print_stack(frame)

        for sig in ("SIGTERM", "SIGSEGV"):
            if hasattr(signal, sig):
                try:
                    signal.signal(getattr(signal, sig), _handle_signal)
                except Exception:
                    pass

        last_cmd = "init"
        last_cmd_time = time.time()
        stop_heartbeat = threading.Event()

        def _heartbeat():
            while not stop_heartbeat.wait(5):
                age = time.time() - last_cmd_time
                _log_worker(f"heartbeat pid={worker_pid} last_cmd={last_cmd} age_s={age:.1f}")

        if debug_add_env:
            thread = threading.Thread(target=_heartbeat, daemon=True)
            thread.start()

        try:
            env = (
                gym.make(
                    env_id,
                    task_kwargs=task,
                    **env_kwargs,
                    browser_args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-application-cache",
                        "--disk-cache-size=1",
                        "--media-cache-size=1",
                        "--disable-cache",
                        "--disable-gpu",
                        "--disable-software-rasterizer",
                        "--incognito",
                    ],
                    user_data_dir=None,  # Forces incognito
                )
                if task
                else gym.make(env_id, **env_kwargs)
            )
            _log_worker("env initialized")
        except Exception as exc:
            _log_worker(f"env init error={type(exc).__name__}: {exc}")
            if debug_add_env:
                traceback.print_exc()
            stop_heartbeat.set()
            raise

        def _log_reset_state(stage: str) -> None:
            if not debug_add_env:
                return
            target = getattr(env, "unwrapped", env)
            state = {}
            for name in ("browser", "context", "page", "browser_context", "playwright"):
                if hasattr(target, name):
                    value = getattr(target, name)
                    state[name] = "set" if value is not None else "none"
            _log_worker(f"reset {stage} state={state}")

        try:
            while True:
                _log_worker("waiting for cmd")
                cmd, data = conn.recv()
                last_cmd = cmd
                last_cmd_time = time.time()
                _log_worker(f"recv cmd={cmd}")
                if cmd == "reset":
                    _log_worker(f"reset start pid={worker_pid}")
                    _log_reset_state("before(pre-close)")
                    start = time.time()
                    try:
                        obs = env.reset(seed=seed)
                    except Exception as exc:
                        _log_worker(f"reset error={type(exc).__name__}: {exc}")
                        if debug_add_env:
                            traceback.print_exc()
                        stop_heartbeat.set()
                        raise
                    _log_reset_state("after(post-init)")
                    _log_worker(f"reset done pid={worker_pid} duration_s={time.time() - start:.2f}")
                    _log_worker("send reset response")
                    conn.send(obs)
                elif cmd == "step":
                    action = data
                    _log_worker(f"step start pid={worker_pid}")
                    start = time.time()
                    try:
                        obs, reward, terminated, truncated, extra_info = env.step(action)
                    except Exception as exc:
                        _log_worker(f"step error={type(exc).__name__}: {exc}")
                        if debug_add_env:
                            traceback.print_exc()
                        stop_heartbeat.set()
                        raise
                    done = terminated or truncated
                    _log_worker(
                        f"step done duration_s={time.time() - start:.2f} reward={reward} done={done}"
                    )
                    _log_worker("send step response")
                    conn.send((obs, reward, terminated or truncated, extra_info))
                elif cmd == "close":
                    _log_worker("close start")
                    env.close()
                    _log_worker("close done")
                    stop_heartbeat.set()
                    conn.close()
                    break
        except EOFError:
            _log_worker("EOFError closing env")
            env.close()
            stop_heartbeat.set()
        except Exception as exc:
            _log_worker(f"worker error={type(exc).__name__}: {exc}")
            if debug_add_env:
                traceback.print_exc()
            try:
                env.close()
            finally:
                stop_heartbeat.set()
                raise

    def reset(self, *args, **kwargs):
        if self._respawn_on_reset:
            self._log_ipc("respawn worker before reset")
            self._restart_worker()
        elif self.process is None:
            self._start_worker()
        pid = self.process.pid if self.process else None
        self._log_ipc(f"send reset pid={pid}")
        self.parent_conn.send(("reset", None))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                alive = bool(self.process and self.process.is_alive())
                exitcode = self.process.exitcode if self.process else None
                ctx = self._format_debug_context()
                task_summary = self._summarize_task()
                self._log_ipc(
                    f"reset timeout after {self.timeout}s alive={alive} pid={pid} exitcode={exitcode} "
                    f"env_id={self.env_id} seed={self.seed} task={task_summary} ctx={ctx}"
                )
                if self._respawn_on_reset_timeout:
                    self._log_ipc("respawn worker after reset timeout")
                    self._restart_worker()
                raise TimeoutError(
                    f"Timeout after {self.timeout} seconds waiting for response. "
                    f"env_id={self.env_id} seed={self.seed} task={task_summary} ctx={ctx}"
                )
        result = self.parent_conn.recv()
        self._log_ipc(f"recv reset pid={pid}")
        return result

    def step(self, action):
        pid = self.process.pid if self.process else None
        self._log_ipc(f"send step pid={pid}")
        self.parent_conn.send(("step", action))
        if self.timeout is not None:
            if not self.parent_conn.poll(self.timeout):
                alive = bool(self.process and self.process.is_alive())
                exitcode = self.process.exitcode if self.process else None
                ctx = self._format_debug_context()
                task_summary = self._summarize_task()
                self._log_ipc(
                    f"step timeout after {self.timeout}s alive={alive} pid={pid} exitcode={exitcode} "
                    f"env_id={self.env_id} seed={self.seed} task={task_summary} ctx={ctx}"
                )
                raise TimeoutError(
                    f"Timeout after {self.timeout} seconds waiting for response. "
                    f"env_id={self.env_id} seed={self.seed} task={task_summary} ctx={ctx}"
                )
        result = self.parent_conn.recv()
        self._log_ipc(f"recv step pid={pid}")
        return result

    def close(self):
        if self.parent_conn is None or self.process is None:
            return
        pid = self.process.pid if self.process else None
        self._log_ipc(f"send close pid={pid}")
        self.parent_conn.send(("close", None))
        if self.process is not None:
            self.process.join(60 * 2)
            if self.process.is_alive():
                self._log_ipc("close timeout, terminating worker")
                print(f"Process still alive after {self.timeout} seconds. Killing it.")
                self._terminate_worker()

    @staticmethod
    def from_dict(extra_info: dict) -> "BrowserGymEnv":
        headless = extra_info.get("headless", True)
        timeout_s = extra_info.get("timeout_s", None)
        timeout_ms = extra_info.get("timeout_ms", None)
        if timeout_s is None and timeout_ms is None:
            timeout_ms = extra_info.get("timeout", None)
        if timeout_s is None and timeout_ms is not None:
            timeout_s = float(timeout_ms) / 1000.0
        seed = extra_info.get("seed", 42)
        return BrowserGymEnv(
            env_id=extra_info["env_id"],
            seed=seed,
            headless=headless,
            timeout=timeout_s,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True
