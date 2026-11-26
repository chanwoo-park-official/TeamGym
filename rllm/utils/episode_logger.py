"""Episode JSON Logger for saving detailed episode information."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rllm.agents.agent import Episode


class EpisodeLogger:
    """Logger to save episodes to individual JSON files with step and data hash."""

    def __init__(self, base_dir: str, subdirectory: str = "episodes"):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
        """
        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        # Convert task to a stable string representation
        task_str = json.dumps(task, sort_keys=True, default=str)
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        # Return first `length` characters of hex digest
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the directory path for a specific training or validation step.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the step directory
        """
        step_dir = self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        # Clean episode_id to make it filesystem-safe
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")

        filename = f"episode_hash{task_hash}_id{episode_id_safe}.json"
        return filename

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        episode_data = {
            "training_step": step,
            "epoch": epoch,
            "episode_id": episode.id,
            "task": episode.task,
            "task_hash": self.compute_task_hash(episode.task),
            "is_correct": episode.is_correct,
            "termination_reason": (
                episode.termination_reason.value if episode.termination_reason else None
            ),
            "termination_reason_detail": (
                (episode.info or {}).get("termination_reason_detail")
            ),
            "metrics": episode.metrics,
            "timing": episode.info.get("timing", {}),
            "trajectories": [],
        }

        def _estimate_token_count(text: str) -> int:
            if not text:
                return 0
            tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            return len(tokens)

        def _message_text(message: dict) -> str:
            reasoning = message.get("reasoning") or ""
            content = message.get("content") or ""
            if reasoning and content:
                return f"{reasoning}\n\n{content}".strip()
            return (reasoning or content).strip()

        for traj in episode.trajectories:
            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "timing": traj.info.get("timing", {}),
                "steps": [],
            }
            for traj_step in traj.steps:
                chat_messages = traj_step.chat_completions or []
                chat_token_estimates = [
                    _estimate_token_count(_message_text(message)) for message in chat_messages
                ]
                model_response_text = traj_step.model_response or ""
                step_info = traj_step.info or {}
                step_entry = {
                    "observation": traj_step.observation,
                    "thought": traj_step.thought,
                    "action": traj_step.action,
                    "reward": traj_step.reward,
                    "done": traj_step.done,
                    "model_response": traj_step.model_response,
                    "chat_completions": chat_messages,
                    "info": step_info,
                    "timing": step_info.get("timing", {}),
                    "token_estimates": {
                        "model_response": _estimate_token_count(model_response_text),
                        "chat_total": sum(chat_token_estimates),
                        "chat_messages": chat_token_estimates,
                    },
                }
                for key in (
                    "forced_action",
                    "forced_match",
                    "auditor_action",
                    "format_ok",
                    "audit_depth",
                    "adv_group_id",
                    "adv_action_group_id",
                    "auditor_source",
                    "tree_path",
                    "tree_parent_path",
                    "tree_child_path",
                    "tree_edge_action",
                    "tree_edge_depth",
                    "tree_edge_failed",
                    "tree_edge_fail_reason",
                    "tree_edge_fail_stage",
                ):
                    if key in step_info:
                        step_entry[key] = step_info.get(key)
                traj_data["steps"].append(step_entry)
            episode_data["trajectories"].append(traj_data)

        # Write to individual file in step-specific directory
        step_dir = self.get_step_dir(step, mode, epoch)
        filename = self.get_episode_filename(episode, step)
        filepath = step_dir / filename

        try:
            with open(filepath, "w") as f:
                json_str = json.dumps(episode_data, indent=2, default=str)
                f.write(json_str + "\n")
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                print(f"[EpisodeLogger] Successfully logged episode {i + 1}/{len(episodes)}: {episode.id}")
            except Exception as e:
                print(f"[EpisodeLogger] Failed to log episode {i + 1}/{len(episodes)}: {e}")
                raise


    def calculate_avg_obs_token_count_per_step(self, episodes: list[Episode]) -> float:
        """Calculate the average number of tokens in observations per step across all episodes.

        Args:
            episodes: List of episodes to analyze
        Returns:
            Average number of tokens in observations per step
        """        
        total_tokens = 0
        total_steps = 0

        for episode in episodes:
            for traj in episode.trajectories:
                for traj_step in traj.steps:
                    token_count = traj_step.info.get("context_length", None)
                    if token_count is not None:
                        total_tokens += token_count
                        total_steps += 1

        if total_steps == 0:
            return 0.0
        return total_tokens / total_steps

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a batch summary in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a summary file for the batch
        """
        # Log individual episodes
        self.log_episodes(episodes, step, mode, epoch)

        # Optionally create batch summary in step-specific directory
        if batch_summary and episodes:
            def _auditor_calls(ep: Episode) -> int:
                metrics = getattr(ep, "metrics", {}) or {}
                try:
                    return int(metrics.get("num_verifications", 0) or 0)
                except (TypeError, ValueError):
                    return 0

            auditor_calls = [_auditor_calls(ep) for ep in episodes]
            episode_files = [self.get_episode_filename(ep, step) for ep in episodes]
            episodes_with_auditor_calls = [
                {"file": filename, "num_verifications": calls}
                for filename, calls in zip(episode_files, auditor_calls, strict=False)
                if calls >= 1
            ]
            def _termination_reason(ep: Episode) -> str:
                reason = getattr(ep, "termination_reason", None)
                if reason is None:
                    return "unknown"
                return reason.value if hasattr(reason, "value") else str(reason)

            def _termination_reason_detail(ep: Episode) -> str | None:
                info = getattr(ep, "info", None) or {}
                detail = info.get("termination_reason_detail")
                return str(detail) if detail is not None else None

            def _shorten(text: str, max_len: int = 200) -> str:
                if not text:
                    return ""
                text = str(text)
                if len(text) <= max_len:
                    return text
                return text[: max_len - 3] + "..."

            def _parse_detail(detail: str | None) -> dict[str, str]:
                parsed: dict[str, str] = {}
                if not detail:
                    return parsed
                parts = str(detail).split("|")
                for part in parts[1:]:
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip()
                return parsed

            termination_reasons = [_termination_reason(ep) for ep in episodes]
            termination_details = [_termination_reason_detail(ep) for ep in episodes]
            termination_reason_counts = Counter(termination_reasons)
            termination_reason_rates = {
                reason: count / len(episodes)
                for reason, count in termination_reason_counts.items()
            }
            termination_reason_files: dict[str, list[str]] = {}
            for filename, reason in zip(episode_files, termination_reasons, strict=False):
                termination_reason_files.setdefault(reason, []).append(filename)

            timeout_base = 0
            timeout_branch = 0
            timeout_unknown = 0
            timeout_source_counts = Counter()
            timeout_env_func_counts = Counter()
            timeout_mode = "unknown"
            for reason, detail in zip(termination_reasons, termination_details, strict=False):
                parsed_detail = _parse_detail(detail)
                is_timeout = reason == "timeout"
                if not is_timeout and reason == "error":
                    env_error = parsed_detail.get("env_error")
                    llm_error = parsed_detail.get("llm_error", "").lower()
                    if env_error == "TimeoutError" or "timeout" in llm_error:
                        is_timeout = True
                if not is_timeout:
                    continue
                timeout_mode = parsed_detail.get("env_mode") or parsed_detail.get("mode")
                if timeout_mode == "base":
                    timeout_base += 1
                elif timeout_mode == "branch":
                    timeout_branch += 1
                else:
                    timeout_unknown += 1
                source = parsed_detail.get("source")
                if source:
                    timeout_source_counts[source] += 1
                env_func = parsed_detail.get("env_func")
                if env_func:
                    timeout_env_func_counts[env_func] += 1
            timeout_like_total = timeout_base + timeout_branch + timeout_unknown

            error_episodes = []
            error_type_counts = Counter()
            error_message_counts = Counter()
            for filename, ep, reason in zip(episode_files, episodes, termination_reasons, strict=False):
                if reason != "error":
                    continue
                err_info = (getattr(ep, "info", None) or {}).get("error", {}) or {}
                err_type = err_info.get("error_type") or "unknown"
                err_message = _shorten(err_info.get("error_message") or "")
                error_type_counts[err_type] += 1
                if err_message:
                    error_message_counts[err_message] += 1
                error_episodes.append(
                    {
                        "file": filename,
                        "error_type": err_type,
                        "error_message": err_message or None,
                    }
                )
            error_rate = len(error_episodes) / len(episodes) if episodes else 0
            avg_obs_token_count_per_step = self.calculate_avg_obs_token_count_per_step(episodes)
            summary_data = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "timeout_mode": timeout_mode,
                "num_episodes": len(episodes),
                "episode_files": episode_files,
                "episodes_with_auditor_calls": episodes_with_auditor_calls,
                "termination_reason_counts": dict(termination_reason_counts),
                "termination_reason_rates": termination_reason_rates,
                "termination_reason_files": termination_reason_files,
                "timeout_detail_counts": {
                    "base": timeout_base,
                    "branch": timeout_branch,
                    "unknown": timeout_unknown,
                },
                "timeout_source_counts": dict(timeout_source_counts),
                "timeout_env_func_counts": dict(timeout_env_func_counts),
                "error_stats": {
                    "num_error_episodes": len(error_episodes),
                    "error_rate": error_rate,
                    "error_type_counts": dict(error_type_counts),
                    "error_message_counts": dict(error_message_counts),
                    "error_episodes": error_episodes,
                },
                "summary_stats": {
                    "total_correct": sum(1 for ep in episodes if ep.is_correct),
                    "total_incorrect": sum(1 for ep in episodes if not ep.is_correct),
                    "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes) if episodes else 0,
                    "avg_trajectories_per_episode": sum(len(ep.trajectories) for ep in episodes) / len(episodes) if episodes else 0,
                    "total_num_verifications": sum(auditor_calls),
                    "avg_num_verifications": sum(auditor_calls) / len(auditor_calls) if auditor_calls else 0,
                    "episodes_with_auditor_calls": len(episodes_with_auditor_calls),
                    "num_error_episodes": len(error_episodes),
                    "error_rate": error_rate,
                    "num_timeout_episodes": termination_reason_counts.get("timeout", 0),
                    "timeout_rate": termination_reason_counts.get("timeout", 0) / len(episodes) if episodes else 0,
                    "num_timeout_like_episodes": timeout_like_total,
                    "timeout_like_rate": timeout_like_total / len(episodes) if episodes else 0,
                    "num_timeout_base": timeout_base,
                    "num_timeout_branch": timeout_branch,
                    "num_timeout_unknown": timeout_unknown,
                    "timeout_base_rate": timeout_base / len(episodes) if episodes else 0,
                    "timeout_branch_rate": timeout_branch / len(episodes) if episodes else 0,
                    "timeout_unknown_rate": timeout_unknown / len(episodes) if episodes else 0,
                    "avg_obs_token_count_per_step": avg_obs_token_count_per_step
                },
            }

            step_dir = self.get_step_dir(step, mode, epoch)
            summary_file = step_dir / "batch_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
