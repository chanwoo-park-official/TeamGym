import os
import random
import sys
import threading
import uuid

import gym
import numpy as np
from rllm.environments.base.base_env import BaseEnv


_WEBSHOP_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
        "webshop",
    )
)
_SIMSERVER_CACHE = {}
_SIMSERVER_LOCK = threading.Lock()


def _ensure_webshop_path() -> None:
    if _WEBSHOP_ROOT not in sys.path:
        sys.path.append(_WEBSHOP_ROOT)


def _shared_server_enabled() -> bool:
    flag = os.getenv("RLLM_WEBSHOP_SHARED_SERVER", "1").lower()
    return flag not in ("0", "false", "no", "off")


def _get_shared_simserver(num_products: int | None) -> "SimServer":
    with _SIMSERVER_LOCK:
        server = _SIMSERVER_CACHE.get(num_products)
        if server is None:
            _ensure_webshop_path()
            from web_agent_site.envs.web_agent_text_env import SimServer
            from web_agent_site.utils import DEFAULT_FILE_PATH

            server = SimServer(
                base_url="http://127.0.0.1:3000",
                file_path=DEFAULT_FILE_PATH,
                filter_goals=None,
                limit_goals=-1,
                num_products=num_products,
                human_goals=0,
                show_attrs=False,
            )
            _SIMSERVER_CACHE[num_products] = server
    return server


# WebShop Environment
class WebShopEnv(BaseEnv):
    def __init__(self, **kwargs):
        self.seed = kwargs.pop("seed", 42)
        self.num_products = kwargs.pop("num_products", 1000)
        self._shared_server = None
        self._session_prefix = f"ws_{os.getpid()}_{uuid.uuid4().hex}_"

        _ensure_webshop_path()
        from web_agent_site.envs import WebAgentTextEnv
        from web_agent_site.models.models import RandomPolicy

        server = None
        if _shared_server_enabled():
            server = _get_shared_simserver(self.num_products)
            self._shared_server = server

        self.env = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=self.num_products,
            session_prefix=self._session_prefix,
            server=server,
        )

        # self.reset()



    def reset(self, task: dict = None) -> tuple[dict, dict]:
        """
        Reset the environment to start a new episode.
        Args:
            task (dict, optional): Task information to set up the environment. Defaults to None.
        Returns:
            tuple: (initial observation, info)
        """
        task = task or {}
        self.seed = task.get("seed", self.seed)
        self.num_products = task.get("num_products", self.num_products)

        random.seed(self.seed)
        np.random.seed(self.seed)
        prev_session = getattr(self.env, "session", None)
        obs, info = self.env.reset()
        if self._shared_server is not None and prev_session:
            new_session = getattr(self.env, "session", None)
            if new_session != prev_session:
                self._shared_server.user_sessions.pop(prev_session, None)
        self.goal_object = self.env.get_instruction_text().removeprefix("Instruction: ")
        if info is None:
            info = {"admissible_commands": self.env.get_available_actions()}
        else:
            info["admissible_commands"] = self.env.get_available_actions()
        return {
            "obs": obs,
            "last_action": None,
            "goal_object": self.goal_object,
        }, info

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Take an action in the environment.
        Args:
            action (str): The action to take.
        Returns:
            tuple: (observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)
        if info is None:
            info = {"admissible_commands": self.env.get_available_actions()}
        else:
            info["admissible_commands"] = self.env.get_available_actions()
        info["score"] = reward
        if done and reward == 1.0:
            reward = 1.0
        else:
            reward = 0.0
        return {
            "obs": obs,
            "last_action": action, 
            "goal_object": self.goal_object,
        }, reward, done, info

    def close(self):
        if self._shared_server is not None:
            session = getattr(self.env, "session", None)
            if session:
                self._shared_server.user_sessions.pop(session, None)
        if hasattr(self.env, "close"):
            self.env.close()

    @staticmethod
    def from_dict(env_info: dict) -> "WebShopEnv":
        return WebShopEnv(seed=env_info["seed"], num_products=env_info["num_products"])
