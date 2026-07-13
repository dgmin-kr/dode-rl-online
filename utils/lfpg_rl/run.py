from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import torch

# The current dg_env TensorBoard install crashes while trying to import
# TensorFlow during Stable-Baselines3 startup. Exposing this sentinel module
# makes TensorBoard use its lightweight tensorflow_stub path instead.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from .config import Config
from .env import DNLTrainingEnv
from dnl.network.registry import get_default_od_pairs
from utils import (
    TEST_OUTPUTS_NPZ_NAME,
    TEST_STEP_HISTORY_CSV_NAME,
    TestOutputRecord,
    build_protocol_simulation_seed,
    resolve_trial_scenario_ids,
    write_test_outputs_npz,
    write_test_step_history_csv,
)
from utils.baseline_evaluation import build_result_name
from .advantage import extract_link_flow_propagation_guidance_components
from .policy import LFPGRLPolicy


def _result_name(trial_index: int, output_suffix: str | None = None) -> str:
    return build_result_name(
        Config.NETWORK_NAME,
        trial_index,
        output_suffix=output_suffix,
    )


def _trial_name(trial_index: int) -> str:
    return _result_name(trial_index)


TRAIN_PROGRESS_INTERVAL_SECONDS = 30.0
LATEST_CHECKPOINT_EPISODE_INTERVAL = 100


def _format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, float(seconds))))


def _format_metric(value: object) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(numeric_value):
        return "nan"
    return f"{numeric_value:.6g}"


def _log_run_status(message: str) -> None:
    print(f"[{Config.EXPERIMENT_NAME}] {message}", flush=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_matrix_csv(path: Path, labels: tuple[str, ...], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time_step", *labels])
        for time_step, row in enumerate(np.asarray(matrix)):
            writer.writerow([time_step, *row.tolist()])


def _step_metric_arrays_from_final_info(final_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    simulated_observations = np.asarray(final_info.get("simulated_observations", []), dtype=np.float64)
    target_observations = np.asarray(final_info.get("target_observations", []), dtype=np.float64)
    observation_scale = np.asarray(final_info.get("observation_scale", []), dtype=np.float64).reshape(-1)
    if simulated_observations.shape != target_observations.shape or simulated_observations.ndim != 2:
        raise ValueError("simulated_observations and target_observations must be matching 2-D arrays.")
    if observation_scale.shape[0] != simulated_observations.shape[1]:
        raise ValueError(
            "observation_scale length must match observation count: "
            f"expected {simulated_observations.shape[1]}, got {observation_scale.shape[0]}."
        )
    error = simulated_observations - target_observations
    scale = np.where(observation_scale > 0.0, observation_scale, np.nan)
    step_mse_values = np.mean(error ** 2, axis=1)
    step_mae_values = np.mean(np.abs(error), axis=1)
    step_normalized_mse_values = np.nanmean((error / scale[None, :]) ** 2, axis=1)
    target_flow_means = np.mean(target_observations, axis=1)
    return step_mse_values, step_mae_values, step_normalized_mse_values, target_flow_means


def _training_rewards_from_final_info(
    final_info: dict[str, Any],
    online_rewards: np.ndarray,
) -> np.ndarray:
    online_rewards = np.asarray(online_rewards, dtype=np.float32).reshape(-1)
    if not bool(Config.RL_RUNTIME_PARAMS.get("use_finalized_observation_rewards", True)):
        return online_rewards

    _, _, step_normalized_mse_values, _ = _step_metric_arrays_from_final_info(final_info)
    if step_normalized_mse_values.size == 0:
        return online_rewards
    horizon = min(int(online_rewards.shape[0]), int(step_normalized_mse_values.shape[0]))
    if horizon <= 0:
        return online_rewards

    finalized_rewards = online_rewards.copy()
    finalized_rewards[:horizon] = -np.nan_to_num(
        step_normalized_mse_values[:horizon],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return finalized_rewards


TRAIN_METRIC_FIELD_ORDER = [
    "update",
    "completed_episodes",
    "collected_timesteps",
    "rollout_episodes",
    "rollout_timesteps",
    "mean_raw_action",
    "mean_clipped_action",
    "clipped_action_low_fraction",
    "clipped_action_high_fraction",
    "policy_mean_action_mean",
    "policy_std_mean",
    "advantage_shaping_weight",
    "mean_global_advantage_abs",
    "mean_advantage_shaping_component_abs",
    "mean_weighted_advantage_shaping_component_abs",
    "advantage_shaping_to_global_ratio",
    "global_to_advantage_shaping_ratio",
    "mean_link_flow_propagation_guidance_abs",
    "mean_link_sensitivity_abs",
    "mean_temporal_mass",
    "max_temporal_mass",
    "policy_gradient_loss",
    "global_value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
    "loss",
]
_TRAIN_METRICS_CSV_LOGGING_FAILED = False
_LFP_DIRECTION_DIAGNOSTIC_CSV_LOGGING_FAILED = False


def _append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_train_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    global _TRAIN_METRICS_CSV_LOGGING_FAILED
    if _TRAIN_METRICS_CSV_LOGGING_FAILED:
        return
    fieldnames = list(TRAIN_METRIC_FIELD_ORDER)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    try:
        _append_csv_rows(path, fieldnames, [row])
    except OSError as exc:
        _TRAIN_METRICS_CSV_LOGGING_FAILED = True
        print(
            f"[warning] train_metrics.csv logging disabled after write failure at {path}: {exc}",
            flush=True,
        )


@dataclass
class EpisodeRollout:
    observations: np.ndarray
    raw_actions: np.ndarray
    clipped_actions: np.ndarray
    old_log_prob_dims: np.ndarray
    link_flow_propagation_guidance: np.ndarray
    estimated_od_matrix: np.ndarray
    rewards: np.ndarray
    final_info: dict[str, Any]
    link_flow_propagation_guidance_stats: dict[str, float]


class EpisodeLogger:
    def __init__(
        self,
        result_dir: Path,
    ) -> None:
        self.result_dir = result_dir
        self.csv_path = result_dir / Config.REWARD_CSV_NAME
        self.step_metrics_path = result_dir / Config.STEP_METRICS_CSV_NAME
        self.rows: list[dict[str, float]] = []
        self.training_start_time: float | None = None
        self.last_episode_time: float | None = None
        self.elapsed_seconds_offset = 0.0
        self.max_runtime_seconds = (
            None if Config.MAX_RUNTIME_SECONDS is None else float(Config.MAX_RUNTIME_SECONDS)
        )
        self.stopped_by_runtime = False
        self.csv_logging_failed = False
        self._csv_warning_printed = False

    def start(self, *, resume_existing: bool = False) -> None:
        now = time.perf_counter()
        self.training_start_time = now
        self.last_episode_time = now
        self.result_dir.mkdir(parents=True, exist_ok=True)
        if resume_existing:
            self._load_existing_rows()
            if not self.csv_path.exists():
                self._safe_csv_write(self.csv_path, self._reward_fieldnames(), [])
            if not self.step_metrics_path.exists():
                self._safe_csv_write(self.step_metrics_path, self._step_metric_fieldnames(), [])
        else:
            self.rows = []
            self.elapsed_seconds_offset = 0.0
            self._initialize_csv_files()

    def runtime_exceeded(self) -> bool:
        if self.max_runtime_seconds is None or self.training_start_time is None:
            return False
        return (time.perf_counter() - self.training_start_time) >= self.max_runtime_seconds

    def record_episode(self, info: dict[str, Any]) -> bool:
        now = time.perf_counter()
        if self.training_start_time is None:
            self.training_start_time = now
        if self.last_episode_time is None:
            self.last_episode_time = self.training_start_time

        elapsed_seconds = self.elapsed_seconds_offset + now - self.training_start_time
        episode_elapsed_seconds = now - self.last_episode_time
        self.last_episode_time = now

        row = {
            "episode": float(len(self.rows) + 1),
            "reward": float(info.get("episode_reward", np.nan)),
            "length": float(info.get("num_steps", np.nan)),
            "elapsed_seconds": float(elapsed_seconds),
            "elapsed_minutes": float(elapsed_seconds / 60.0),
            "episode_elapsed_seconds": float(episode_elapsed_seconds),
            "episode_mse": float(info.get("episode_mse", np.nan)),
            "episode_mae": float(info.get("episode_mae", np.nan)),
            "episode_normalized_mse": float(info.get("episode_normalized_mse", np.nan)),
            "elapsed_time": time.strftime("%H:%M:%S", time.gmtime(max(0.0, elapsed_seconds))),
        }
        self.rows.append(row)
        step_metric_rows = self._build_step_metric_rows(row, info)
        self._append_reward_csv(row)
        self._append_step_metrics_csv(step_metric_rows)

        if self.runtime_exceeded():
            self.stopped_by_runtime = True
        return self.stopped_by_runtime

    def finalize_numeric_outputs(self) -> None:
        return

    @staticmethod
    def _reward_fieldnames() -> list[str]:
        return [
            "episode",
            "reward",
            "length",
            "elapsed_seconds",
            "elapsed_minutes",
            "episode_elapsed_seconds",
            "episode_mse",
            "episode_mae",
            "episode_normalized_mse",
            "elapsed_time",
        ]

    @staticmethod
    def _step_metric_fieldnames() -> list[str]:
        return [
            "episode",
            "step",
            "point_index",
            "step_mse",
            "step_mae",
            "step_normalized_mse",
            "step_relative_rmse_percent",
            "target_flow_mean",
            "elapsed_seconds",
            "elapsed_minutes",
        ]

    def _initialize_csv_files(self) -> None:
        self._safe_csv_write(self.csv_path, self._reward_fieldnames(), [])
        self._safe_csv_write(self.step_metrics_path, self._step_metric_fieldnames(), [])

    def _load_existing_rows(self) -> None:
        self.rows = []
        self.elapsed_seconds_offset = 0.0
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for raw_row in reader:
                    row: dict[str, float] = {}
                    for key, value in raw_row.items():
                        if key is None:
                            continue
                        try:
                            row[key] = float(value)
                        except (TypeError, ValueError):
                            row[key] = float("nan")
                    if row:
                        self.rows.append(row)
        except OSError as exc:
            self._handle_csv_logging_error(self.csv_path, exc)
            self.rows = []
            self.elapsed_seconds_offset = 0.0
            return
        if self.rows:
            last_elapsed = float(self.rows[-1].get("elapsed_seconds", 0.0))
            self.elapsed_seconds_offset = last_elapsed if np.isfinite(last_elapsed) else 0.0

    def _append_reward_csv(self, row: dict[str, float]) -> None:
        self._safe_csv_append(self.csv_path, self._reward_fieldnames(), [row])

    def _build_step_metric_rows(self, episode_row: dict[str, float], info: dict[str, Any]) -> list[dict[str, float]]:
        step_mse_values, step_mae_values, step_normalized_mse_values, target_flow_means = (
            _step_metric_arrays_from_final_info(info)
        )
        if step_mse_values.size == 0:
            return []
        step_relative_rmse_values = np.full_like(step_mse_values, np.nan, dtype=float)
        valid_target_mask = np.isfinite(target_flow_means) & (target_flow_means > 0.0)
        step_relative_rmse_values[valid_target_mask] = (
            np.sqrt(step_mse_values[valid_target_mask]) / target_flow_means[valid_target_mask] * 100.0
        )

        episode_index = int(episode_row["episode"])
        elapsed_seconds = float(episode_row["elapsed_seconds"])
        elapsed_minutes = float(episode_row["elapsed_minutes"])
        return [
            {
                "episode": float(episode_index),
                "step": float(step_index + 1),
                "point_index": float((episode_index - 1) * len(step_mse_values) + step_index + 1),
                "step_mse": float(step_mse_values[step_index]),
                "step_mae": float(step_mae_values[step_index]),
                "step_normalized_mse": float(step_normalized_mse_values[step_index]),
                "step_relative_rmse_percent": float(step_relative_rmse_values[step_index]),
                "target_flow_mean": float(target_flow_means[step_index]),
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_minutes,
            }
            for step_index in range(len(step_mse_values))
        ]

    def _append_step_metrics_csv(self, rows: list[dict[str, float]]) -> None:
        self._safe_csv_append(self.step_metrics_path, self._step_metric_fieldnames(), rows)

    def _safe_csv_write(self, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            self._handle_csv_logging_error(path, exc)

    def _safe_csv_append(self, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        if not rows or self.csv_logging_failed:
            return
        try:
            _append_csv_rows(path, fieldnames, rows)
        except OSError as exc:
            self._handle_csv_logging_error(path, exc)

    def _handle_csv_logging_error(self, path: Path, exc: OSError) -> None:
        self.csv_logging_failed = True
        if not self._csv_warning_printed:
            self._csv_warning_printed = True
            print(
                f"[warning] CSV logging disabled after write failure at {path}: {exc}",
                flush=True,
            )

def parse_optional_int(value: str) -> Optional[int]:
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def make_env_fn(
    *,
    scenario_dataset_dir: Path,
    scenario_split: str,
    seed: int,
    fixed_scenario_id: str | None = None,
    fixed_simulation_seed: int | None = None,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        return DNLTrainingEnv(
            action_low=Config.ACTION_LOW,
            action_high=Config.ACTION_HIGH,
            scenario_dataset_dir=scenario_dataset_dir,
            scenario_split=scenario_split,
            fixed_scenario_id=fixed_scenario_id,
            fixed_simulation_seed=fixed_simulation_seed,
            seed=seed,
        )

    return _init


def build_vec_env(
    *,
    scenario_dataset_dir: Path,
    scenario_split: str,
    base_seed: int,
    num_envs: int,
    fixed_scenario_id: str | None = None,
    fixed_simulation_seed: int | None = None,
) -> VecMonitor:
    env_fns: list[Callable[[], gym.Env]] = [
        make_env_fn(
            scenario_dataset_dir=scenario_dataset_dir,
            scenario_split=scenario_split,
            seed=base_seed + env_index,
            fixed_scenario_id=fixed_scenario_id,
            fixed_simulation_seed=fixed_simulation_seed,
        )
        for env_index in range(num_envs)
    ]
    vec_env: VecEnv
    if Config.USE_SUBPROC and num_envs > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    return VecMonitor(vec_env)


def write_scenario_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_scenario_output_name(scenario_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(scenario_id).strip()).strip("._")
    return cleaned or "scenario"


def write_single_scenario_eval_outputs(
    *,
    output_dir: Path,
    row: dict[str, Any],
    record: TestOutputRecord,
    trial_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_test_outputs_npz(output_dir / TEST_OUTPUTS_NPZ_NAME, [record])
    write_test_step_history_csv(output_dir / TEST_STEP_HISTORY_CSV_NAME, [record])
    write_scenario_metrics_csv(output_dir / Config.SCENARIO_METRICS_CSV_NAME, [row])
    scenario_summary = {
        "split": str(record.split),
        "num_scenarios": 1,
        "selected_test_scenario_ids": [str(record.scenario_id)],
        "mean_episode_reward": float(row["episode_reward"]),
        "mean_episode_mse": float(row["episode_mse"]),
        "mean_episode_mae": float(row["episode_mae"]),
        "mean_episode_normalized_mse": float(row["episode_normalized_mse"]),
        "scenario_metrics_csv": str(output_dir / Config.SCENARIO_METRICS_CSV_NAME),
        "test_outputs_npz": str(output_dir / TEST_OUTPUTS_NPZ_NAME),
        "test_step_history_csv": str(output_dir / TEST_STEP_HISTORY_CSV_NAME),
    }
    (output_dir / Config.TEST_SUMMARY_JSON_NAME).write_text(
        json.dumps(scenario_summary, indent=2),
        encoding="utf-8",
    )
    config_snapshot = {
        "trial_index": int(trial_seed),
        "evaluation_trial_seed": int(trial_seed),
        "network_name": Config.NETWORK_NAME,
        "test_split": str(record.split),
        "num_test_scenarios": 1,
        "selected_test_scenario_ids": [str(record.scenario_id)],
        "scenario_generation_seed": record.scenario_generation_seed,
        "simulation_seed": record.simulation_seed,
    }
    (output_dir / Config.CONFIG_SNAPSHOT_JSON_NAME).write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )


def build_lfpg_test_step_rows(final_info: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    simulated_link_flows = np.asarray(final_info["simulated_link_flows"], dtype=np.float64)
    step_mse_values, step_mae_values, step_normalized_mse_values, _ = _step_metric_arrays_from_final_info(final_info)

    rewards = np.asarray(final_info.get("episode_rewards", []), dtype=np.float64)
    coarse_od_matrix = np.asarray(final_info.get("coarse_od_matrix", []), dtype=np.float64)
    policy_residual_matrix = np.asarray(final_info.get("policy_residual_matrix", []), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for step_index in range(simulated_link_flows.shape[0]):
        row = {
            "step_index": int(step_index),
            "step_mse": float(step_mse_values[step_index]),
            "step_mae": float(step_mae_values[step_index]),
            "step_normalized_mse": float(step_normalized_mse_values[step_index]),
            "reward": float(rewards[step_index]) if step_index < rewards.size else float("nan"),
        }
        if coarse_od_matrix.ndim == 2 and step_index < coarse_od_matrix.shape[0]:
            row["coarse_action_sum"] = float(np.sum(coarse_od_matrix[step_index]))
        if policy_residual_matrix.ndim == 2 and step_index < policy_residual_matrix.shape[0]:
            row["policy_residual_abs_mean"] = float(np.mean(np.abs(policy_residual_matrix[step_index])))
        rows.append(row)
    return tuple(rows)


LFP_DIRECTION_DIAGNOSTIC_KEYS = [
    "mean_link_flow_propagation_guidance_abs",
]


def append_lfp_direction_diagnostic_csv(path: Path, row: dict[str, Any]) -> None:
    global _LFP_DIRECTION_DIAGNOSTIC_CSV_LOGGING_FAILED
    if _LFP_DIRECTION_DIAGNOSTIC_CSV_LOGGING_FAILED:
        return
    fieldnames = [
        "update",
        "completed_episodes",
        "collected_timesteps",
        *LFP_DIRECTION_DIAGNOSTIC_KEYS,
    ]
    try:
        _append_csv_rows(path, fieldnames, [row])
    except OSError as exc:
        _LFP_DIRECTION_DIAGNOSTIC_CSV_LOGGING_FAILED = True
        print(
            f"[warning] lfp_direction_diagnostic.csv logging disabled after write failure at {path}: {exc}",
            flush=True,
        )


def evaluate_policy_on_split(
    *,
    model: LFPGRLPolicy,
    device: torch.device,
    scenario_ids: list[str],
    scenario_dataset_dir: Path,
    scenario_split: str,
    trial_seed: int,
    metrics_csv_path: Path,
    trial_result_dir: Path | None = None,
    write_scenario_subdirs: bool = False,
) -> dict[str, Any]:
    scenario_rows: list[dict[str, Any]] = []
    output_records: list[TestOutputRecord] = []
    for scenario_index, scenario_id in enumerate(scenario_ids):
        simulation_seed = build_protocol_simulation_seed(
            trial_seed=trial_seed,
            split=scenario_split,
            scenario_id=scenario_id,
        )
        eval_env = build_vec_env(
            scenario_dataset_dir=scenario_dataset_dir,
            scenario_split=scenario_split,
            base_seed=trial_seed + 1009 * (scenario_index + 1),
            num_envs=1,
            fixed_scenario_id=scenario_id,
            fixed_simulation_seed=simulation_seed,
        )
        try:
            flow_scale = np.asarray(eval_env.env_method("get_flow_scale")[0], dtype=np.float32)
            rollouts = collect_episode_batch(
                vec_env=eval_env,
                model=model,
                device=device,
                flow_scale=flow_scale,
                deterministic=True,
                compute_guidance=False,
            )
        finally:
            eval_env.close()

        if len(rollouts) != 1:
            raise RuntimeError(
                f"Expected a single evaluation rollout for scenario_id={scenario_id!r}, got {len(rollouts)}."
            )

        final_info = dict(rollouts[0].final_info)
        final_info["episode_rewards"] = np.asarray(rollouts[0].rewards, dtype=np.float32)
        row = {
            "scenario_id": str(scenario_id),
            "split": str(scenario_split),
            "episode_reward": float(final_info["episode_reward"]),
            "episode_mse": float(final_info["episode_mse"]),
            "episode_mae": float(final_info["episode_mae"]),
            "episode_normalized_mse": float(final_info["episode_normalized_mse"]),
            "num_steps": int(final_info["num_steps"]),
            "scenario_generation_seed": final_info.get("scenario_generation_seed"),
            "simulation_seed": final_info.get("simulation_seed"),
        }
        scenario_rows.append(row)
        output_records.append(
            TestOutputRecord(
                scenario_id=str(scenario_id),
                split=str(scenario_split),
                scenario_generation_seed=(
                    None if final_info.get("scenario_generation_seed") is None else int(final_info["scenario_generation_seed"])
                ),
                simulation_seed=None if final_info.get("simulation_seed") is None else int(final_info["simulation_seed"]),
                od_labels=tuple(final_info["od_labels"]),
                link_labels=tuple(final_info["link_labels"]),
                estimated_od_matrix=np.asarray(final_info["estimated_od_matrix"], dtype=np.float32),
                simulated_link_flows=np.asarray(final_info["simulated_link_flows"], dtype=np.float32),
                observed_link_indices=np.asarray(final_info["observed_link_indices"], dtype=np.int64),
                observation_labels=tuple(final_info.get("observation_labels", ())),
                target_observations=np.asarray(final_info["target_observations"], dtype=np.float32),
                simulated_observations=np.asarray(final_info["simulated_observations"], dtype=np.float32),
                observation_scale=(
                    None
                    if final_info.get("observation_scale") is None
                    else np.asarray(final_info["observation_scale"], dtype=np.float32)
                ),
                step_rows=build_lfpg_test_step_rows(final_info),
            )
        )

    scenario_output_dirs: dict[str, str] = {}
    if not write_scenario_subdirs:
        write_scenario_metrics_csv(metrics_csv_path, scenario_rows)
    if trial_result_dir is not None and output_records:
        trial_result_path = Path(trial_result_dir)
        if not write_scenario_subdirs:
            write_test_outputs_npz(trial_result_path / TEST_OUTPUTS_NPZ_NAME, output_records)
            write_test_step_history_csv(trial_result_path / TEST_STEP_HISTORY_CSV_NAME, output_records)
        if write_scenario_subdirs:
            for row, record in zip(scenario_rows, output_records):
                scenario_dir = trial_result_path / safe_scenario_output_name(record.scenario_id)
                write_single_scenario_eval_outputs(
                    output_dir=scenario_dir,
                    row=row,
                    record=record,
                    trial_seed=trial_seed,
                )
                scenario_output_dirs[str(record.scenario_id)] = str(scenario_dir)
    mean_reward = float(np.mean([row["episode_reward"] for row in scenario_rows])) if scenario_rows else float("nan")
    mean_mse = float(np.mean([row["episode_mse"] for row in scenario_rows])) if scenario_rows else float("nan")
    mean_mae = float(np.mean([row["episode_mae"] for row in scenario_rows])) if scenario_rows else float("nan")
    mean_normalized_mse = (
        float(np.mean([row["episode_normalized_mse"] for row in scenario_rows]))
        if scenario_rows
        else float("nan")
    )
    return {
        "split": str(scenario_split),
        "num_scenarios": int(len(scenario_rows)),
        "mean_episode_reward": mean_reward,
        "mean_episode_mse": mean_mse,
        "mean_episode_mae": mean_mae,
        "mean_episode_normalized_mse": mean_normalized_mse,
        "scenario_metrics_csv": None if write_scenario_subdirs else str(metrics_csv_path),
        "test_outputs_npz": (
            None
            if write_scenario_subdirs or trial_result_dir is None
            else str(Path(trial_result_dir) / TEST_OUTPUTS_NPZ_NAME)
        ),
        "test_step_history_csv": (
            None
            if write_scenario_subdirs or trial_result_dir is None
            else str(Path(trial_result_dir) / TEST_STEP_HISTORY_CSV_NAME)
        ),
        "scenario_output_dirs": scenario_output_dirs,
    }


def resolve_total_timesteps(
) -> int:
    if Config.MAX_RUNTIME_SECONDS is not None:
        return int(Config.RUNTIME_ONLY_TOTAL_TIMESTEPS)
    raise ValueError(
        "No runtime stop criterion is available for LFPG-RL training. "
        "Set MAX_RUNTIME_SECONDS or pass --runtime-seconds."
    )


def compute_scheduled_value(
    completed_updates: int,
    base_value: float,
    schedule: str,
    decay_rollouts: int,
    final_value: float = 0.0,
) -> float:
    completed_updates = max(int(completed_updates), 0)
    schedule = str(schedule).strip().lower()
    decay_rollouts = int(decay_rollouts)

    if schedule == "constant" or decay_rollouts <= 0:
        return float(base_value)

    progress_fraction = float(np.clip(completed_updates / max(decay_rollouts, 1), 0.0, 1.0))
    if schedule == "linear":
        return float(base_value) + (float(final_value) - float(base_value)) * progress_fraction
    if schedule == "exp":
        return float(base_value) * float(np.exp(-progress_fraction))
    raise ValueError(f"Unsupported schedule={schedule!r}.")


def compute_advantage_shaping_weight(completed_updates: int) -> float:
    if not bool(Config.RL_RUNTIME_PARAMS.get("lfp_a_enabled", True)):
        return 0.0
    base_weight = float(Config.RL_RUNTIME_PARAMS.get("lfp_a_advantage_weight", 0.25))
    schedule = str(Config.RL_RUNTIME_PARAMS.get("lfp_a_decay_schedule", "constant")).strip().lower()
    decay_rollouts = int(Config.RL_RUNTIME_PARAMS.get("lfp_a_decay_rollouts", 0))
    return compute_scheduled_value(
        completed_updates=completed_updates,
        base_value=base_weight,
        schedule=schedule,
        decay_rollouts=decay_rollouts,
        final_value=0.0,
    )


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_policy_model(
    *,
    observation_dim: int,
    action_dim: int,
    device: torch.device,
    action_low: float | None = None,
    action_high: float | None = None,
    bound_policy_mean_override: bool | None = None,
) -> LFPGRLPolicy:
    policy_hidden_dims = tuple(int(dim) for dim in Config.PPO_PARAMS["policy_kwargs"]["net_arch"])
    bound_policy_mean = (
        bool(Config.PPO_PARAMS.get("bound_policy_mean", False))
        if bound_policy_mean_override is None
        else bool(bound_policy_mean_override)
    )
    resolved_action_low = float(Config.ACTION_LOW if action_low is None else action_low)
    resolved_action_high = float(Config.ACTION_HIGH if action_high is None else action_high)
    model = LFPGRLPolicy(
        observation_dim=int(observation_dim),
        action_dim=int(action_dim),
        action_low=resolved_action_low,
        action_high=resolved_action_high,
        hidden_dims=policy_hidden_dims,
        bound_policy_mean=bound_policy_mean,
    ).to(device)
    initial_policy_mean = Config.PPO_PARAMS.get("initial_policy_mean")
    if initial_policy_mean is not None:
        initial_policy_mean = float(initial_policy_mean)
        if bound_policy_mean:
            action_span = resolved_action_high - resolved_action_low
            normalized_mean = (initial_policy_mean - resolved_action_low) / max(action_span, 1e-6)
            normalized_mean = float(np.clip(normalized_mean, 1e-4, 1.0 - 1e-4))
            initial_policy_mean = float(np.log(normalized_mean / (1.0 - normalized_mean)))
        with torch.no_grad():
            model.policy_mean.bias.fill_(initial_policy_mean)
    initial_policy_std = Config.PPO_PARAMS.get("initial_policy_std")
    if initial_policy_std is not None:
        initial_policy_std = float(initial_policy_std)
        if initial_policy_std <= 0.0:
            raise ValueError("initial_policy_std must be positive.")
        with torch.no_grad():
            model.log_std.fill_(float(np.log(initial_policy_std)))
    return model


def coarse_residual_policy_enabled(params: dict[str, Any] | None = None) -> bool:
    source = Config.RL_RUNTIME_PARAMS if params is None else params
    return bool(source.get("coarse_residual_policy_enabled", False))


def resolve_policy_action_bounds(params: dict[str, Any] | None = None) -> tuple[float, float]:
    source = Config.RL_RUNTIME_PARAMS if params is None else params
    if coarse_residual_policy_enabled(source):
        residual_high = float(source.get("residual_action_high", max((Config.ACTION_HIGH - Config.ACTION_LOW) * 0.15, 1.0)))
        if residual_high <= 0.0:
            raise ValueError("residual_action_high must be positive.")
        return -residual_high, residual_high
    return float(Config.ACTION_LOW), float(Config.ACTION_HIGH)


def resolve_observation_dim(scenario_dataset: Any, params: dict[str, Any] | None = None) -> int:
    source = Config.RL_RUNTIME_PARAMS if params is None else params
    observation_dim = int(1 + 3 * scenario_dataset.num_links)
    if bool(source.get("include_target_observation_state", False)):
        observation_dim += int(scenario_dataset.num_observations)
    if coarse_residual_policy_enabled(source) and bool(source.get("include_coarse_od_state", True)):
        observation_dim += int(scenario_dataset.num_od)
    return observation_dim


def resolve_policy_checkpoint_path(
    *,
    trial_index: int,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint was not found: {checkpoint_path}")
        return checkpoint_path

    trial_name = _trial_name(int(trial_index))
    candidate_paths = (
        Config.RESULT_DIR / trial_name / Config.FINAL_MODEL_NAME,
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    raise FileNotFoundError(
        "No saved LFPG-RL policy checkpoint was found for "
        f"trial={trial_index}. Expected one of: {', '.join(str(path) for path in candidate_paths)}"
    )


def resolve_resume_checkpoint_path(
    *,
    trial_index: int,
    explicit_path: str | None = None,
) -> Path:
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint was not found: {checkpoint_path}")
        return checkpoint_path

    trial_name = _trial_name(int(trial_index))
    candidate_paths = (
        Config.RESULT_DIR / trial_name / Config.LATEST_MODEL_NAME,
        Config.RESULT_DIR / trial_name / Config.FINAL_MODEL_NAME,
    )
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    raise FileNotFoundError(
        "No saved LFPG-RL resume checkpoint was found for "
        f"trial={trial_index}. Expected one of: {', '.join(str(path) for path in candidate_paths)}"
    )


def load_checkpoint_payload(checkpoint_path: Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def load_model_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> LFPGRLPolicy:
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    scenario_dataset = Config.load_scenario_dataset()
    checkpoint_ppo_params = dict(checkpoint.get("ppo_params", {}))
    checkpoint_runtime_params = dict(checkpoint.get("rl_runtime_params", checkpoint.get("lfpg_rl_params", {})))
    observation_dim = resolve_observation_dim(scenario_dataset, checkpoint_runtime_params)
    checkpoint_action_low = checkpoint.get("policy_action_low")
    checkpoint_action_high = checkpoint.get("policy_action_high")
    if checkpoint_action_low is None or checkpoint_action_high is None:
        checkpoint_action_low, checkpoint_action_high = resolve_policy_action_bounds(checkpoint_runtime_params)
    model = build_policy_model(
        observation_dim=observation_dim,
        action_dim=int(len(get_default_od_pairs(Config.NETWORK_NAME))),
        device=device,
        action_low=float(checkpoint_action_low),
        action_high=float(checkpoint_action_high),
        bound_policy_mean_override=bool(checkpoint_ppo_params.get("bound_policy_mean", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint_state(
    *,
    checkpoint_path: Path,
    model: LFPGRLPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_checkpoint_payload(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].detach().cpu())
    if (
        torch.cuda.is_available()
        and checkpoint.get("torch_cuda_random_state_all") is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state_all"])
    model.train()
    return dict(checkpoint)


def advantage_shaping_enabled() -> bool:
    return bool(Config.RL_RUNTIME_PARAMS.get("lfp_a_enabled", True))


def link_flow_propagation_guidance_enabled() -> bool:
    return advantage_shaping_enabled()


def apply_lfpg_rl_overrides(args: argparse.Namespace) -> None:
    ppo_override_mapping = {
        "ppo_learning_rate": "learning_rate",
        "ppo_ent_coef": "ent_coef",
        "ppo_clip_range": "clip_range",
        "ppo_vf_coef": "vf_coef",
        "ppo_target_kl": "target_kl",
        "ppo_n_epochs": "n_epochs",
        "ppo_batch_size": "batch_size",
        "ppo_max_grad_norm": "max_grad_norm",
        "ppo_gamma": "gamma",
        "ppo_gae_lambda": "gae_lambda",
        "ppo_initial_policy_mean": "initial_policy_mean",
        "ppo_initial_policy_std": "initial_policy_std",
    }
    for arg_name, config_key in ppo_override_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            Config.PPO_PARAMS[config_key] = value

    override_mapping = {
        "lfp_a_advantage_weight": "lfp_a_advantage_weight",
        "lfp_a_decay_schedule": "lfp_a_decay_schedule",
        "lfp_a_decay_rollouts": "lfp_a_decay_rollouts",
        "lfp_a_advantage_clip": "lfp_a_advantage_clip",
    }
    for arg_name, config_key in override_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            Config.LFPG_PARAMS[config_key] = value
            Config.RL_RUNTIME_PARAMS[config_key] = value


def normalize_values(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values
    mean = values.mean()
    variance = torch.mean((values - mean).square())
    return (values - mean) / torch.sqrt(variance + 1e-8)


def compute_episode_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rewards.reshape(-1)
    values = values.reshape(-1)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)

    for step in range(rewards.shape[0] - 1, -1, -1):
        if step == rewards.shape[0] - 1:
            next_non_terminal = 0.0
            next_value = 0.0
        else:
            next_non_terminal = 1.0
            next_value = values[step + 1]
        delta = rewards[step] + float(gamma) * next_value * next_non_terminal - values[step]
        last_gae = delta + float(gamma) * float(gae_lambda) * next_non_terminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


def normalize_od_vector(
    od_vector: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    scale = od_vector.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
    normalized = od_vector / scale
    if clip_value > 0.0:
        normalized = torch.clamp(normalized, min=-float(clip_value), max=float(clip_value))
    return normalized


def collect_episode_batch(
    vec_env: VecMonitor,
    model: LFPGRLPolicy,
    device: torch.device,
    flow_scale: np.ndarray,
    deterministic: bool = False,
    compute_guidance: bool = True,
) -> list[EpisodeRollout]:
    num_envs = int(vec_env.num_envs)
    obs = vec_env.reset()
    observations: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    raw_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    clipped_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    old_log_prob_dims: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    rewards: list[list[float]] = [[] for _ in range(num_envs)]
    final_infos: list[dict[str, Any] | None] = [None for _ in range(num_envs)]
    done_flags = np.zeros(num_envs, dtype=bool)

    while not bool(np.all(done_flags)):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_action_tensor, clipped_action_tensor, log_prob_dims_tensor, _ = model.act(
                obs_tensor,
                deterministic=deterministic,
            )

        raw_action = raw_action_tensor.cpu().numpy().astype(np.float32)
        clipped_action = clipped_action_tensor.cpu().numpy().astype(np.float32)
        log_prob_dims = log_prob_dims_tensor.cpu().numpy().astype(np.float32)

        next_obs, reward, done, infos = vec_env.step(clipped_action)

        for env_index in range(num_envs):
            if done_flags[env_index]:
                continue
            observations[env_index].append(np.asarray(obs[env_index], dtype=np.float32))
            raw_actions[env_index].append(raw_action[env_index].copy())
            clipped_actions[env_index].append(clipped_action[env_index].copy())
            old_log_prob_dims[env_index].append(log_prob_dims[env_index].copy())
            rewards[env_index].append(float(reward[env_index]))

            if bool(done[env_index]):
                final_info = dict(infos[env_index])
                final_info.setdefault("num_steps", len(rewards[env_index]))
                final_infos[env_index] = final_info
                done_flags[env_index] = True

        obs = next_obs

    payloads = vec_env.env_method("consume_completed_episode_payload")
    rollouts: list[EpisodeRollout] = []
    for env_index in range(num_envs):
        final_info = final_infos[env_index]
        payload = payloads[env_index]
        if final_info is None or payload is None:
            raise RuntimeError(
                f"Missing completed episode payload for env_index={env_index}. "
                "LFPG-RL rollout collection requires terminal payload retrieval."
            )

        final_info["flow_scale"] = np.asarray(payload.get("flow_scale", flow_scale), dtype=np.float32).copy()
        final_info["target_observations"] = np.asarray(payload["target_observations"], dtype=np.float32).copy()
        final_info["observation_scale"] = np.asarray(payload["observation_scale"], dtype=np.float32).copy()
        final_info["observed_link_indices"] = np.asarray(payload["observed_link_indices"], dtype=np.int64).copy()
        simulated_link_flows = np.asarray(final_info["simulated_link_flows"], dtype=np.float32)
        target_observations = np.asarray(final_info["target_observations"], dtype=np.float32)
        observed_link_indices = np.asarray(final_info["observed_link_indices"], dtype=np.int64)

        if compute_guidance and link_flow_propagation_guidance_enabled():
            guidance = extract_link_flow_propagation_guidance_components(
                temporal_link_inflows=np.asarray(payload["temporal_link_inflows"], dtype=np.float32),
                simulated_link_flows=simulated_link_flows,
                target_observations=target_observations,
                flow_scale=np.asarray(payload.get("flow_scale", flow_scale), dtype=np.float32),
                gamma=float(Config.PPO_PARAMS["gamma"]),
                observed_link_indices=observed_link_indices,
                observation_scale=(
                    None
                    if payload.get("observation_scale") is None
                    else np.asarray(payload["observation_scale"], dtype=np.float32)
                ),
            )
            link_flow_propagation_guidance = guidance.link_flow_propagation_guidance
            link_flow_propagation_guidance_stats = guidance.stats
            horizon = int(link_flow_propagation_guidance.shape[0])
            if coarse_residual_policy_enabled():
                estimated_od = np.asarray(final_info.get("estimated_od_matrix", []), dtype=np.float32)[:horizon]
                if estimated_od.shape == link_flow_propagation_guidance.shape:
                    lower_blocked = estimated_od <= float(Config.ACTION_LOW) + 1e-6
                    upper_blocked = estimated_od >= float(Config.ACTION_HIGH) - 1e-6
                    blocked_direction = (
                        (lower_blocked & (link_flow_propagation_guidance < 0.0))
                        | (upper_blocked & (link_flow_propagation_guidance > 0.0))
                    )
                    if np.any(blocked_direction):
                        link_flow_propagation_guidance = link_flow_propagation_guidance.copy()
                        link_flow_propagation_guidance[blocked_direction] = 0.0
                    link_flow_propagation_guidance_stats = {
                        **link_flow_propagation_guidance_stats,
                        "lfp_information_abs_mean": float(np.mean(np.abs(link_flow_propagation_guidance))),
                        "lfp_information_abs_max": float(np.max(np.abs(link_flow_propagation_guidance))),
                        "residual_clip_masked_fraction": float(np.mean(blocked_direction)),
                    }
        else:
            horizon = min(
                len(rewards[env_index]),
                int(simulated_link_flows.shape[0]),
                int(target_observations.shape[0]),
            )
            link_flow_propagation_guidance = np.zeros((horizon, int(vec_env.action_space.shape[0])), dtype=np.float32)
            link_flow_propagation_guidance_stats = {
                "lfp_information_abs_mean": 0.0,
                "lfp_information_abs_max": 0.0,
                "temporal_mass_mean": 0.0,
                "temporal_mass_max": 0.0,
                "link_sensitivity_abs_mean": 0.0,
            }
        final_info["link_flow_propagation_guidance_stats"] = dict(link_flow_propagation_guidance_stats)
        online_rewards = np.asarray(rewards[env_index], dtype=np.float32)
        training_rewards = _training_rewards_from_final_info(final_info, online_rewards)[:horizon]
        final_info["online_episode_reward"] = float(final_info.get("episode_reward", np.sum(online_rewards)))
        final_info["online_episode_rewards"] = online_rewards.copy()
        final_info["episode_rewards"] = training_rewards.copy()
        final_info["episode_reward"] = float(np.sum(training_rewards))

        for name, array in {
            "link_flow_propagation_guidance": link_flow_propagation_guidance,
        }.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or Inf.")

        estimated_od_matrix = np.asarray(
            final_info.get("estimated_od_matrix", np.zeros_like(link_flow_propagation_guidance)),
            dtype=np.float32,
        )[:horizon]

        rollouts.append(
            EpisodeRollout(
                observations=np.asarray(observations[env_index][:horizon], dtype=np.float32),
                raw_actions=np.asarray(raw_actions[env_index][:horizon], dtype=np.float32),
                clipped_actions=np.asarray(clipped_actions[env_index][:horizon], dtype=np.float32),
                old_log_prob_dims=np.asarray(old_log_prob_dims[env_index][:horizon], dtype=np.float32),
                link_flow_propagation_guidance=np.asarray(link_flow_propagation_guidance, dtype=np.float32),
                estimated_od_matrix=estimated_od_matrix,
                rewards=np.asarray(training_rewards, dtype=np.float32),
                final_info=final_info,
                link_flow_propagation_guidance_stats=link_flow_propagation_guidance_stats,
            )
        )
    return rollouts


def train_on_rollouts(
    model: LFPGRLPolicy,
    optimizer: torch.optim.Optimizer,
    rollouts: list[EpisodeRollout],
    device: torch.device,
    advantage_shaping_weight: float,
) -> dict[str, float]:
    observations = np.concatenate([rollout.observations for rollout in rollouts], axis=0)
    raw_actions = np.concatenate([rollout.raw_actions for rollout in rollouts], axis=0)
    clipped_actions = np.concatenate([rollout.clipped_actions for rollout in rollouts], axis=0)
    old_log_prob_dims = np.concatenate([rollout.old_log_prob_dims for rollout in rollouts], axis=0)
    link_flow_propagation_guidance = np.concatenate([rollout.link_flow_propagation_guidance for rollout in rollouts], axis=0)

    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    raw_action_tensor = torch.as_tensor(raw_actions, dtype=torch.float32, device=device)
    clipped_action_tensor = torch.as_tensor(clipped_actions, dtype=torch.float32, device=device)
    old_log_prob_tensor = torch.as_tensor(old_log_prob_dims, dtype=torch.float32, device=device)
    link_flow_propagation_guidance_tensor = torch.as_tensor(link_flow_propagation_guidance, dtype=torch.float32, device=device)

    gamma = float(Config.PPO_PARAMS["gamma"])
    gae_lambda = float(Config.PPO_PARAMS["gae_lambda"])
    advantage_shaping_clip = float(Config.RL_RUNTIME_PARAMS.get("lfp_a_advantage_clip", 2.0))

    with torch.no_grad():
        old_outputs = model.evaluate(
            observations=observation_tensor,
            raw_actions=raw_action_tensor,
        )
        old_global_value_tensor = old_outputs["global_value"]
        old_policy_mean_tensor, old_policy_log_std_tensor = model.policy_stats(observation_tensor)
        old_policy_std_tensor = old_policy_log_std_tensor.exp().detach().clamp_min(1e-6)

        raw_global_advantages: list[torch.Tensor] = []
        global_returns: list[torch.Tensor] = []
        guidance_information_parts: list[torch.Tensor] = []
        advantage_shaping_components: list[torch.Tensor] = []
        cursor = 0
        for rollout in rollouts:
            horizon = int(len(rollout.rewards))
            reward_part = torch.as_tensor(rollout.rewards, dtype=torch.float32, device=device)
            value_part = old_global_value_tensor[cursor : cursor + horizon]
            advantage_part, return_part = compute_episode_gae(
                rewards=reward_part,
                values=value_part,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            guidance_information_part = link_flow_propagation_guidance_tensor[cursor : cursor + horizon]
            direction_part = normalize_od_vector(
                od_vector=guidance_information_part,
                clip_value=advantage_shaping_clip,
            )
            action_residual_part = (
                raw_action_tensor[cursor : cursor + horizon]
                - old_policy_mean_tensor[cursor : cursor + horizon].detach()
            ) / old_policy_std_tensor[cursor : cursor + horizon]
            if advantage_shaping_clip > 0.0:
                action_residual_part = torch.clamp(
                    action_residual_part,
                    min=-float(advantage_shaping_clip),
                    max=float(advantage_shaping_clip),
                )
            directional_action_advantage_part = direction_part * action_residual_part
            if advantage_shaping_clip > 0.0:
                directional_action_advantage_part = torch.clamp(
                    directional_action_advantage_part,
                    min=-float(advantage_shaping_clip),
                    max=float(advantage_shaping_clip),
                )

            raw_global_advantages.append(advantage_part)
            global_returns.append(return_part)
            guidance_information_parts.append(guidance_information_part)
            advantage_shaping_components.append(directional_action_advantage_part)
            cursor += horizon

        raw_global_advantage_tensor = torch.cat(raw_global_advantages, dim=0)
        global_return_tensor = torch.cat(global_returns, dim=0)
        link_flow_propagation_guidance_tensor = torch.cat(guidance_information_parts, dim=0)
        advantage_shaping_component_tensor = torch.cat(advantage_shaping_components, dim=0)
        global_advantage_tensor = normalize_values(raw_global_advantage_tensor)

        weighted_advantage_shaping_component_tensor = float(advantage_shaping_weight) * advantage_shaping_component_tensor
        advantage_tensor = global_advantage_tensor.unsqueeze(-1) + weighted_advantage_shaping_component_tensor

    clip_range = float(Config.PPO_PARAMS["clip_range"])
    ent_coef = float(Config.PPO_PARAMS["ent_coef"])
    vf_coef = float(Config.PPO_PARAMS["vf_coef"])
    max_grad_norm = float(Config.PPO_PARAMS["max_grad_norm"])
    n_epochs = int(Config.PPO_PARAMS["n_epochs"])
    batch_size = min(int(Config.PPO_PARAMS["batch_size"]), int(observation_tensor.shape[0]))
    target_kl = Config.PPO_PARAMS.get("target_kl")

    model.train()
    entropy_losses: list[float] = []
    policy_gradient_losses: list[float] = []
    global_value_losses: list[float] = []
    clip_fractions: list[float] = []
    approx_kl_divs: list[float] = []
    advantage_abs_means: list[float] = []
    raw_global_advantage_abs_means: list[float] = []
    global_advantage_abs_means: list[float] = []
    raw_link_flow_propagation_guidance_abs_means: list[float] = []
    advantage_shaping_component_abs_means: list[float] = []
    weighted_advantage_shaping_component_abs_means: list[float] = []
    advantage_shaping_to_global_ratio_means: list[float] = []
    global_to_advantage_shaping_ratio_means: list[float] = []
    continue_training = True

    num_samples = int(observation_tensor.shape[0])
    indices = np.arange(num_samples)
    loss = torch.tensor(0.0, device=device)

    for _ in range(n_epochs):
        np.random.shuffle(indices)
        for start_index in range(0, num_samples, batch_size):
            batch_indices = indices[start_index : start_index + batch_size]
            obs_batch = observation_tensor[batch_indices]
            raw_action_batch = raw_action_tensor[batch_indices]
            old_log_prob_batch = old_log_prob_tensor[batch_indices]
            advantage_batch = advantage_tensor[batch_indices]
            raw_global_advantage_batch = raw_global_advantage_tensor[batch_indices]
            global_advantage_batch = global_advantage_tensor[batch_indices]
            global_return_batch = global_return_tensor[batch_indices]
            advantage_shaping_component_batch = advantage_shaping_component_tensor[batch_indices]
            weighted_advantage_shaping_component_batch = weighted_advantage_shaping_component_tensor[batch_indices]
            link_flow_propagation_guidance_batch = link_flow_propagation_guidance_tensor[batch_indices]

            outputs = model.evaluate(
                observations=obs_batch,
                raw_actions=raw_action_batch,
            )

            ratio = torch.exp(outputs["log_prob_dims"] - old_log_prob_batch)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

            # The global PPO signal is scalar per action vector. Averaging it
            # over all OD dimensions makes vanilla PPO updates vanish on the
            # 930-dimensional Melbourne OD action space, so aggregate the
            # factorized surrogate across OD dimensions before batch averaging.
            global_advantage_dims = global_advantage_batch.unsqueeze(-1)
            global_policy_loss_1 = ratio * global_advantage_dims
            global_policy_loss_2 = clipped_ratio * global_advantage_dims
            global_policy_loss = -torch.mean(
                torch.sum(torch.min(global_policy_loss_1, global_policy_loss_2), dim=1)
            )

            if float(advantage_shaping_weight) != 0.0:
                shaping_policy_loss_1 = ratio * weighted_advantage_shaping_component_batch
                shaping_policy_loss_2 = clipped_ratio * weighted_advantage_shaping_component_batch
                shaping_policy_loss = -torch.mean(
                    torch.sum(torch.min(shaping_policy_loss_1, shaping_policy_loss_2), dim=1)
                )
            else:
                shaping_policy_loss = torch.zeros((), dtype=torch.float32, device=device)
            policy_loss = global_policy_loss + shaping_policy_loss

            global_value_loss = torch.mean((outputs["global_value"] - global_return_batch).square())
            entropy_loss = -torch.mean(outputs["entropy_dims"])
            loss = policy_loss + ent_coef * entropy_loss + vf_coef * global_value_loss

            with torch.no_grad():
                log_ratio = outputs["log_prob_dims"] - old_log_prob_batch
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).cpu().item()
                approx_kl_divs.append(float(approx_kl_div))

            if target_kl is not None and approx_kl_div > 1.5 * float(target_kl):
                continue_training = False
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            policy_gradient_losses.append(float(policy_loss.item()))
            global_value_losses.append(float(global_value_loss.item()))
            entropy_losses.append(float(entropy_loss.item()))
            clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_range).float().mean().item()))
            advantage_abs_means.append(float(torch.mean(torch.abs(advantage_batch)).item()))
            raw_global_advantage_abs_means.append(float(torch.mean(torch.abs(raw_global_advantage_batch)).item()))
            global_advantage_abs_means.append(float(torch.mean(torch.abs(global_advantage_batch)).item()))
            raw_link_flow_propagation_guidance_abs_means.append(float(torch.mean(torch.abs(link_flow_propagation_guidance_batch)).item()))
            advantage_shaping_component_abs_means.append(float(torch.mean(torch.abs(advantage_shaping_component_batch)).item()))
            weighted_advantage_shaping_component_abs_means.append(
                float(torch.mean(torch.abs(weighted_advantage_shaping_component_batch)).item())
            )
            mean_global_component_abs = max(global_advantage_abs_means[-1], 1e-8)
            mean_weighted_advantage_shaping_abs = weighted_advantage_shaping_component_abs_means[-1]
            advantage_shaping_to_global_ratio_means.append(
                mean_weighted_advantage_shaping_abs / mean_global_component_abs
            )
            global_to_advantage_shaping_ratio_means.append(
                mean_global_component_abs / max(mean_weighted_advantage_shaping_abs, 1e-8)
            )

        if not continue_training:
            break

    model.eval()
    with torch.no_grad():
        updated_outputs = model.evaluate(
            observations=observation_tensor,
            raw_actions=raw_action_tensor,
        )
        updated_policy_mean, updated_policy_log_std = model.policy_stats(observation_tensor)

    target_flat = global_return_tensor.reshape(-1)
    prediction_flat = updated_outputs["global_value"].reshape(-1)
    explained_var = float("nan")
    target_variance = torch.var(target_flat).item()
    if target_variance > 0.0:
        explained_var = 1.0 - float(torch.var(target_flat - prediction_flat).item() / target_variance)

    mean_policy_gradient_loss = float(np.mean(policy_gradient_losses)) if policy_gradient_losses else float("nan")
    mean_global_value_loss = float(np.mean(global_value_losses)) if global_value_losses else float("nan")

    return {
        "mean_raw_action": float(raw_action_tensor.mean().item()),
        "mean_clipped_action": float(clipped_action_tensor.mean().item()),
        "clipped_action_low_fraction": float(
            (clipped_action_tensor <= float(model.action_low) + 1e-6).float().mean().item()
        ),
        "clipped_action_high_fraction": float(
            (clipped_action_tensor >= float(model.action_high) - 1e-6).float().mean().item()
        ),
        "policy_mean_action_mean": float(updated_policy_mean.mean().item()),
        "policy_std_mean": float(updated_policy_log_std.exp().mean().item()),
        "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else float("nan"),
        "policy_gradient_loss": mean_policy_gradient_loss,
        "global_value_loss": mean_global_value_loss,
        "approx_kl": float(np.mean(approx_kl_divs)) if approx_kl_divs else float("nan"),
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else float("nan"),
        "loss": float(loss.item()),
        "explained_variance": explained_var,
        "advantage_std": float(advantage_tensor.std(unbiased=False).item()),
        "global_advantage_std": float(global_advantage_tensor.std(unbiased=False).item()),
        "mean_link_flow_propagation_guidance": float(link_flow_propagation_guidance_tensor.mean().item()),
        "mean_advantage_abs": float(np.mean(advantage_abs_means)) if advantage_abs_means else float("nan"),
        "mean_raw_global_advantage_abs": (
            float(np.mean(raw_global_advantage_abs_means)) if raw_global_advantage_abs_means else float("nan")
        ),
        "mean_global_advantage_abs": (
            float(np.mean(global_advantage_abs_means)) if global_advantage_abs_means else float("nan")
        ),
        "mean_link_flow_propagation_guidance_abs": (
            float(np.mean(raw_link_flow_propagation_guidance_abs_means)) if raw_link_flow_propagation_guidance_abs_means else float("nan")
        ),
        "mean_advantage_shaping_component_abs": (
            float(np.mean(advantage_shaping_component_abs_means)) if advantage_shaping_component_abs_means else float("nan")
        ),
        "mean_weighted_advantage_shaping_component_abs": (
            float(np.mean(weighted_advantage_shaping_component_abs_means))
            if weighted_advantage_shaping_component_abs_means
            else float("nan")
        ),
        "advantage_shaping_to_global_ratio": (
            float(np.mean(advantage_shaping_to_global_ratio_means)) if advantage_shaping_to_global_ratio_means else float("nan")
        ),
        "global_to_advantage_shaping_ratio": (
            float(np.mean(global_to_advantage_shaping_ratio_means)) if global_to_advantage_shaping_ratio_means else float("nan")
        ),
        "advantage_shaping_weight": float(advantage_shaping_weight),
        "mean_global_return": float(global_return_tensor.mean().item()),
        "mean_global_value": float(updated_outputs["global_value"].mean().item()),
        "mean_temporal_mass": float(np.mean([rollout.link_flow_propagation_guidance_stats["temporal_mass_mean"] for rollout in rollouts])),
        "max_temporal_mass": float(np.max([rollout.link_flow_propagation_guidance_stats["temporal_mass_max"] for rollout in rollouts])),
        "mean_link_sensitivity_abs": float(
            np.mean([rollout.link_flow_propagation_guidance_stats["link_sensitivity_abs_mean"] for rollout in rollouts])
        ),
    }


def save_model_checkpoint(
    path: Path,
    model: LFPGRLPolicy,
    optimizer: torch.optim.Optimizer,
    train_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "experiment_name": Config.EXPERIMENT_NAME,
        "network_name": Config.NETWORK_NAME,
        "algorithm": Config.ALGORITHM,
        "ppo_params": Config.PPO_PARAMS,
        "rl_env_params": Config.RL_ENV_PARAMS,
        "lfpg_params": Config.LFPG_PARAMS,
        "rl_runtime_params": Config.RL_RUNTIME_PARAMS,
        "policy_action_low": float(model.action_low),
        "policy_action_high": float(model.action_high),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(payload, path)


def evaluate_saved_policy_for_trial(
    *,
    trial_index: int,
    device: torch.device,
    policy_path: str | None = None,
    selected_test_scenario_ids: list[str] | None = None,
    output_suffix: str | None = None,
) -> dict[str, Any]:
    Config.ensure_dirs()
    scenario_dataset = Config.load_scenario_dataset()
    all_test_scenario_ids = list(scenario_dataset.get_split_ids(Config.TEST_SCENARIO_SPLIT))
    test_scenario_ids = list(
        resolve_trial_scenario_ids(
            all_test_scenario_ids,
            trial_seed=int(trial_index),
            max_scenarios=1 if Config.TEST_MAX_SCENARIOS is None else int(Config.TEST_MAX_SCENARIOS),
            explicit_scenario_ids=selected_test_scenario_ids,
        )
    )

    checkpoint_path = resolve_policy_checkpoint_path(
        trial_index=int(trial_index),
        explicit_path=policy_path,
    )
    model = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    trial_result_dir = Config.RESULT_DIR / _result_name(int(trial_index), output_suffix=output_suffix)
    trial_result_dir.mkdir(parents=True, exist_ok=True)
    write_scenario_subdirs = len(test_scenario_ids) > 1
    if write_scenario_subdirs:
        for stale_output_name in (
            TEST_OUTPUTS_NPZ_NAME,
            Config.SCENARIO_METRICS_CSV_NAME,
            TEST_STEP_HISTORY_CSV_NAME,
        ):
            stale_output_path = trial_result_dir / stale_output_name
            if stale_output_path.exists():
                stale_output_path.unlink()

    test_summary = evaluate_policy_on_split(
        model=model,
        device=device,
        scenario_ids=test_scenario_ids,
        scenario_dataset_dir=Config.SCENARIO_DATASET_DIR,
        scenario_split=Config.TEST_SCENARIO_SPLIT,
        trial_seed=int(trial_index),
        metrics_csv_path=trial_result_dir / Config.SCENARIO_METRICS_CSV_NAME,
        trial_result_dir=trial_result_dir,
        write_scenario_subdirs=write_scenario_subdirs,
    )
    test_summary["policy_checkpoint"] = str(checkpoint_path)
    test_summary["policy_checkpoint_kind"] = (
        "explicit_policy"
        if policy_path is not None
        else "final_policy"
    )
    (trial_result_dir / Config.TEST_SUMMARY_JSON_NAME).write_text(
        json.dumps(test_summary, indent=2),
        encoding="utf-8",
    )
    config_snapshot = {
        "trial_index": int(trial_index),
        "evaluation_trial_seed": int(trial_index),
        "experiment_name": Config.EXPERIMENT_NAME,
        "algorithm": Config.ALGORITHM,
        "network_name": Config.NETWORK_NAME,
        "scenario_dataset_dir": str(Config.SCENARIO_DATASET_DIR),
        "test_split": Config.TEST_SCENARIO_SPLIT,
        "num_test_scenarios_total": int(len(all_test_scenario_ids)),
        "num_test_scenarios": int(len(test_scenario_ids)),
        "selected_test_scenario_ids": test_scenario_ids,
        "policy_checkpoint": str(checkpoint_path),
        "policy_checkpoint_kind": (
            "explicit_policy"
            if policy_path is not None
            else "final_policy"
        ),
        "scenario_output_layout": (
            "scenario_subdirectories"
            if write_scenario_subdirs
            else "single_output_directory"
        ),
    }
    (trial_result_dir / Config.CONFIG_SNAPSHOT_JSON_NAME).write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
    for scenario_id in (test_scenario_ids if write_scenario_subdirs else []):
        scenario_dir = trial_result_dir / safe_scenario_output_name(scenario_id)
        scenario_summary_path = scenario_dir / Config.TEST_SUMMARY_JSON_NAME
        if scenario_summary_path.exists():
            scenario_summary = json.loads(scenario_summary_path.read_text(encoding="utf-8"))
            scenario_summary["policy_checkpoint"] = str(checkpoint_path)
            scenario_summary["policy_checkpoint_kind"] = config_snapshot["policy_checkpoint_kind"]
            scenario_summary_path.write_text(json.dumps(scenario_summary, indent=2), encoding="utf-8")
        scenario_config_path = scenario_dir / Config.CONFIG_SNAPSHOT_JSON_NAME
        if scenario_config_path.exists():
            scenario_config = json.loads(scenario_config_path.read_text(encoding="utf-8"))
            scenario_config["policy_checkpoint"] = str(checkpoint_path)
            scenario_config["policy_checkpoint_kind"] = config_snapshot["policy_checkpoint_kind"]
            scenario_config_path.write_text(json.dumps(scenario_config, indent=2), encoding="utf-8")
    return test_summary


def run_one_trial(
    trial_index: int,
    num_envs: int,
    seed_override: Optional[int] = None,
    evaluate_after_training: bool = True,
    selected_test_scenario_ids: list[str] | None = None,
    resume_training: bool = False,
    resume_policy_path: str | None = None,
    max_episodes: int | None = None,
) -> None:
    resume_training = bool(resume_training or resume_policy_path is not None)
    Config.ensure_dirs()
    _log_run_status(
        f"trial={trial_index} loading dataset from {Config.SCENARIO_DATASET_DIR}"
    )
    scenario_dataset = Config.load_scenario_dataset()
    train_scenario_ids = list(scenario_dataset.get_split_ids(Config.TRAIN_SCENARIO_SPLIT))
    all_test_scenario_ids = list(scenario_dataset.get_split_ids(Config.TEST_SCENARIO_SPLIT))
    test_scenario_ids = list(
        resolve_trial_scenario_ids(
            all_test_scenario_ids,
            trial_seed=int(trial_index),
            max_scenarios=1 if Config.TEST_MAX_SCENARIOS is None else int(Config.TEST_MAX_SCENARIOS),
            explicit_scenario_ids=selected_test_scenario_ids,
        )
    )
    if not train_scenario_ids:
        raise ValueError("Training split is empty. Generate a scenario dataset before running LFPG-RL.")

    effective_total_timesteps = resolve_total_timesteps()
    test_eval_status = (
        f"test_eval=enabled test_scenarios={len(test_scenario_ids)}/{len(all_test_scenario_ids)}"
        if bool(evaluate_after_training)
        else "test_eval=disabled(train_only)"
    )
    _log_run_status(
        f"trial={trial_index} train_scenarios={len(train_scenario_ids)} "
        f"{test_eval_status} "
        f"runtime_cap={Config.MAX_RUNTIME_SECONDS}s num_envs={num_envs} "
        f"use_subproc={Config.USE_SUBPROC} train_only={not evaluate_after_training} "
        f"resume={resume_training} max_episodes={max_episodes}"
    )

    trial_result_dir = Config.RESULT_DIR / _trial_name(trial_index)
    trial_result_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path: Path | None = None
    if resume_training:
        resume_checkpoint_path = resolve_resume_checkpoint_path(
            trial_index=int(trial_index),
            explicit_path=resume_policy_path,
        )

    for stale_output_name in (
        Config.CONFIG_SNAPSHOT_JSON_NAME,
        Config.TRAINING_SUMMARY_JSON_NAME,
        Config.TEST_SUMMARY_JSON_NAME,
        Config.SCENARIO_METRICS_CSV_NAME,
        "test_outputs.npz",
        "test_step_history.csv",
        *(
            ()
            if resume_training
            else (
                Config.REWARD_CSV_NAME,
                Config.STEP_METRICS_CSV_NAME,
                Config.FINAL_MODEL_NAME,
                Config.LATEST_MODEL_NAME,
                "train_metrics.csv",
                "lfp_direction_diagnostics.csv",
            )
        ),
    ):
        stale_output_path = trial_result_dir / stale_output_name
        if stale_output_path.exists():
            stale_output_path.unlink()

    evaluation_trial_seed = int(trial_index)
    config_snapshot = {
        "trial_index": trial_index,
        "seed": int(seed_override) if seed_override is not None else 1000 + trial_index,
        "evaluation_trial_seed": evaluation_trial_seed,
        "num_envs": int(num_envs),
        "rollout_episodes_per_update": int(max(1, num_envs)),
        "effective_total_timesteps": effective_total_timesteps,
        "experiment_name": Config.EXPERIMENT_NAME,
        "algorithm": Config.ALGORITHM,
        "network_name": Config.NETWORK_NAME,
        "action_low": Config.ACTION_LOW,
        "action_high": Config.ACTION_HIGH,
        "policy_action_low": resolve_policy_action_bounds()[0],
        "policy_action_high": resolve_policy_action_bounds()[1],
        "route_choice_mode": Config.ROUTE_CHOICE_MODE,
        "stochastic_logit_scale": Config.STOCHASTIC_LOGIT_SCALE,
        "dnl_sample_route_choices": Config.DNL_SAMPLE_ROUTE_CHOICES,
        "dnl_route_choice_sampling_unit": Config.DNL_ROUTE_CHOICE_SAMPLING_UNIT,
        "dnl_max_paths_per_od": Config.DNL_MAX_PATHS_PER_OD,
        "dnl_due_max_iterations": Config.DNL_DUE_MAX_ITERATIONS,
        "dnl_due_tolerance": Config.DNL_DUE_TOLERANCE,
        "dnl_clearance_steps": Config.DNL_CLEARANCE_STEPS,
        "dnl_parallel_kernels": Config.DNL_PARALLEL_KERNELS,
        "dnl_numba_threads": Config.DNL_NUMBA_THREADS,
        "dnl_progress_logging": Config.DNL_PROGRESS_LOGGING,
        "max_runtime_seconds": (
            None if Config.MAX_RUNTIME_SECONDS is None else float(Config.MAX_RUNTIME_SECONDS)
        ),
        "scenario_dataset_dir": str(Config.SCENARIO_DATASET_DIR),
        "train_split": Config.TRAIN_SCENARIO_SPLIT,
        "test_split": Config.TEST_SCENARIO_SPLIT,
        "num_train_scenarios": int(len(train_scenario_ids)),
        "num_test_scenarios_total": int(len(all_test_scenario_ids)),
        "num_test_scenarios": int(len(test_scenario_ids)),
        "selected_test_scenario_ids": test_scenario_ids,
        "num_steps": scenario_dataset.num_steps,
        "num_links": scenario_dataset.num_links,
        "num_od": len(get_default_od_pairs(Config.NETWORK_NAME)),
        "ppo_params": Config.PPO_PARAMS,
        "rl_env_params": Config.RL_ENV_PARAMS,
        "lfpg_params": Config.LFPG_PARAMS,
        "rl_runtime_params": Config.RL_RUNTIME_PARAMS,
        "resume_training": bool(resume_training),
        "resume_checkpoint_path": (
            None if resume_checkpoint_path is None else str(resume_checkpoint_path.resolve())
        ),
        "latest_checkpoint_episode_interval": int(LATEST_CHECKPOINT_EPISODE_INTERVAL),
        "max_episodes": None if max_episodes is None else int(max_episodes),
    }
    (trial_result_dir / Config.CONFIG_SNAPSHOT_JSON_NAME).write_text(
        json.dumps(config_snapshot, indent=2),
        encoding="utf-8",
    )
    lfp_direction_diagnostic_csv_path = trial_result_dir / "lfp_direction_diagnostics.csv"
    if not resume_training and lfp_direction_diagnostic_csv_path.exists():
        lfp_direction_diagnostic_csv_path.unlink()

    seed = int(seed_override) if seed_override is not None else 1000 + trial_index
    set_global_seed(seed)
    device = select_device()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    _log_run_status(
        f"trial={trial_index} seed={seed} device={device} result_dir={trial_result_dir}"
    )

    vec_env = build_vec_env(
        scenario_dataset_dir=Config.SCENARIO_DATASET_DIR,
        scenario_split=Config.TRAIN_SCENARIO_SPLIT,
        base_seed=seed,
        num_envs=int(num_envs),
    )
    flow_scale = np.asarray(vec_env.env_method("get_flow_scale")[0], dtype=np.float32)
    action_dim = int(vec_env.action_space.shape[0])
    policy_action_low = float(np.asarray(vec_env.action_space.low, dtype=np.float32).reshape(-1)[0])
    policy_action_high = float(np.asarray(vec_env.action_space.high, dtype=np.float32).reshape(-1)[0])
    model = build_policy_model(
        observation_dim=int(vec_env.observation_space.shape[0]),
        action_dim=action_dim,
        device=device,
        action_low=policy_action_low,
        action_high=policy_action_high,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(Config.PPO_PARAMS["learning_rate"]))

    resume_train_metrics: dict[str, Any] = {}
    if resume_checkpoint_path is not None:
        resume_checkpoint = load_training_checkpoint_state(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        resume_train_metrics = dict(resume_checkpoint.get("train_metrics", {}))
        _log_run_status(
            f"trial={trial_index} resumed training checkpoint={resume_checkpoint_path} "
            f"checkpoint_episodes={resume_train_metrics.get('completed_episodes', 'unknown')} "
            f"checkpoint_timesteps={resume_train_metrics.get('collected_timesteps', 'unknown')}"
        )

    logger = EpisodeLogger(
        result_dir=trial_result_dir,
    )
    logger.start(resume_existing=resume_training)

    fallback_resume_timesteps = int(len(logger.rows)) * int(scenario_dataset.num_steps)
    total_timesteps_collected = max(
        int(resume_train_metrics.get("collected_timesteps", 0) or 0),
        fallback_resume_timesteps if resume_training else 0,
    )
    update_count = int(resume_train_metrics.get("update", 0) or 0)
    last_train_metrics: dict[str, float] = dict(resume_train_metrics)
    train_metric_rows: list[dict[str, Any]] = []
    train_metrics_csv_path = trial_result_dir / "train_metrics.csv"
    rollout_episodes_per_update = max(1, int(num_envs))
    final_model_path = trial_result_dir / Config.FINAL_MODEL_NAME
    latest_model_path = trial_result_dir / Config.LATEST_MODEL_NAME
    last_latest_checkpoint_episode = (
        int(len(logger.rows) // LATEST_CHECKPOINT_EPISODE_INTERVAL)
        * int(LATEST_CHECKPOINT_EPISODE_INTERVAL)
    )

    next_progress_log_at = start_time
    _log_run_status(
        f"trial={trial_index} training started; progress_interval={TRAIN_PROGRESS_INTERVAL_SECONDS:.0f}s "
        f"initial_episodes={len(logger.rows)} initial_timesteps={total_timesteps_collected} "
        f"initial_updates={update_count}"
    )
    while total_timesteps_collected < effective_total_timesteps:
        if logger.runtime_exceeded():
            logger.stopped_by_runtime = True
            break

        should_stop_after_update = False
        collected_rollouts = collect_episode_batch(
            vec_env=vec_env,
            model=model,
            device=device,
            flow_scale=flow_scale,
        )
        rollouts: list[EpisodeRollout] = []
        for rollout in collected_rollouts:
            rollouts.append(rollout)
            total_timesteps_collected += int(len(rollout.rewards))
            if logger.record_episode(rollout.final_info):
                should_stop_after_update = True
            if max_episodes is not None and len(logger.rows) >= int(max_episodes):
                should_stop_after_update = True
                break
        if not rollouts:
            break

        advantage_shaping_weight = compute_advantage_shaping_weight(update_count)
        update_count += 1
        last_train_metrics = train_on_rollouts(
            model=model,
            optimizer=optimizer,
            rollouts=rollouts,
            device=device,
            advantage_shaping_weight=advantage_shaping_weight,
        )
        rollout_timesteps = int(sum(len(rollout.rewards) for rollout in rollouts))
        last_train_metrics["update"] = int(update_count)
        last_train_metrics["completed_episodes"] = int(len(logger.rows))
        last_train_metrics["collected_timesteps"] = int(total_timesteps_collected)
        last_train_metrics["rollout_episodes"] = int(len(rollouts))
        last_train_metrics["rollout_timesteps"] = rollout_timesteps

        train_metric_row = dict(last_train_metrics)
        train_metric_rows.append(train_metric_row)
        append_train_metrics_csv(train_metrics_csv_path, train_metric_row)
        append_lfp_direction_diagnostic_csv(
            lfp_direction_diagnostic_csv_path,
            {
                "update": int(update_count),
                "completed_episodes": int(len(logger.rows)),
                "collected_timesteps": int(total_timesteps_collected),
                **last_train_metrics,
            },
        )

        if (
            len(logger.rows) >= last_latest_checkpoint_episode + LATEST_CHECKPOINT_EPISODE_INTERVAL
        ):
            save_model_checkpoint(
                latest_model_path,
                model=model,
                optimizer=optimizer,
                train_metrics=last_train_metrics,
            )
            last_latest_checkpoint_episode = (
                int(len(logger.rows) // LATEST_CHECKPOINT_EPISODE_INTERVAL)
                * int(LATEST_CHECKPOINT_EPISODE_INTERVAL)
            )
            _log_run_status(
                f"trial={trial_index} saved latest checkpoint at "
                f"episodes={len(logger.rows)} path={latest_model_path}"
            )

        now = time.perf_counter()
        should_log_progress = (
            update_count == 1
            or now >= next_progress_log_at
            or should_stop_after_update
            or logger.stopped_by_runtime
        )
        if should_log_progress:
            last_episode = logger.rows[-1] if logger.rows else {}
            _log_run_status(
                f"trial={trial_index} update={update_count} episodes={len(logger.rows)} "
                f"timesteps={total_timesteps_collected} elapsed={_format_elapsed(now - start_time)} "
                f"reward={_format_metric(last_episode.get('reward'))} "
                f"normalized_mse={_format_metric(last_episode.get('episode_normalized_mse'))} "
                f"stopped_by_runtime={logger.stopped_by_runtime}"
            )
            next_progress_log_at = now + TRAIN_PROGRESS_INTERVAL_SECONDS

        if should_stop_after_update or logger.stopped_by_runtime:
            break

    vec_env.close()
    final_checkpoint_kind = "last_policy"
    save_model_checkpoint(
        final_model_path,
        model=model,
        optimizer=optimizer,
        train_metrics=last_train_metrics,
    )
    save_model_checkpoint(
        latest_model_path,
        model=model,
        optimizer=optimizer,
        train_metrics=last_train_metrics,
    )
    logger.finalize_numeric_outputs()

    test_summary: dict[str, Any] | None = None
    if evaluate_after_training:
        _log_run_status(f"trial={trial_index} evaluating final policy")
        evaluation_checkpoint_path = final_model_path
        evaluation_model = load_model_checkpoint(
            checkpoint_path=evaluation_checkpoint_path,
            device=device,
        )
        test_summary = evaluate_policy_on_split(
            model=evaluation_model,
            device=device,
            scenario_ids=test_scenario_ids,
            scenario_dataset_dir=Config.SCENARIO_DATASET_DIR,
            scenario_split=Config.TEST_SCENARIO_SPLIT,
            trial_seed=evaluation_trial_seed,
            metrics_csv_path=trial_result_dir / Config.SCENARIO_METRICS_CSV_NAME,
            trial_result_dir=trial_result_dir,
        )
        test_summary["policy_checkpoint"] = str(evaluation_checkpoint_path.resolve())
        test_summary["policy_checkpoint_kind"] = "final_policy"
        (trial_result_dir / Config.TEST_SUMMARY_JSON_NAME).write_text(
            json.dumps(test_summary, indent=2),
            encoding="utf-8",
        )

    ended_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - start_time
    training_summary = {
        "trial_index": trial_index,
        "rollout_episodes_per_update": int(rollout_episodes_per_update),
        "effective_total_timesteps": int(effective_total_timesteps),
        "collected_timesteps": int(total_timesteps_collected),
        "completed_updates": int(update_count),
        "max_runtime_seconds": (
            None if Config.MAX_RUNTIME_SECONDS is None else float(Config.MAX_RUNTIME_SECONDS)
        ),
        "stopped_by_runtime": bool(logger.stopped_by_runtime),
        "completed_episodes": int(len(logger.rows)),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_minutes": float(elapsed_seconds / 60.0),
        "elapsed_hms": time.strftime("%H:%M:%S", time.gmtime(max(0.0, elapsed_seconds))),
        "device": str(device),
        "last_train_metrics": last_train_metrics,
        "final_policy_checkpoint": str(final_model_path.resolve()),
        "latest_policy_checkpoint": str(latest_model_path.resolve()),
        "final_checkpoint_kind": final_checkpoint_kind,
        "resume_training": bool(resume_training),
        "resume_checkpoint_path": (
            None if resume_checkpoint_path is None else str(resume_checkpoint_path.resolve())
        ),
        "max_episodes": None if max_episodes is None else int(max_episodes),
        "test_summary": test_summary,
        "evaluate_after_training": bool(evaluate_after_training),
    }
    (trial_result_dir / Config.TRAINING_SUMMARY_JSON_NAME).write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    _log_run_status(
        f"trial={trial_index} completed updates={update_count} episodes={len(logger.rows)} "
        f"elapsed={_format_elapsed(elapsed_seconds)} stopped_by_runtime={logger.stopped_by_runtime}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="Trial index.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=Config.NUM_ENVS,
        help="Override the number of vectorized environments.",
    )
    parser.add_argument(
        "--use-subproc",
        action=argparse.BooleanOptionalAction,
        default=Config.USE_SUBPROC,
        help="Use subprocess vector environments when num-envs > 1.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=Config.NETWORK_NAME,
        help="Network name. Only melbourne_scats is supported.",
    )
    parser.add_argument(
        "--runtime-seconds",
        type=parse_optional_int,
        default=None,
        help="Optional override for Config.MAX_RUNTIME_SECONDS. Use 'none' to disable runtime stopping.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override. When omitted, uses 1000 + trial.",
    )
    parser.add_argument(
        "--test-max-scenarios",
        type=int,
        default=None,
        help="Optional cap on test scenarios evaluated after training.",
    )
    parser.add_argument(
        "--selected-test-scenario-ids",
        nargs="+",
        default=None,
        help="Explicit test scenario IDs to evaluate. Overrides internal trial-seeded scenario selection.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train and save the policy checkpoint without running held-out test evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, load a saved policy checkpoint, and evaluate it on the test split.",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path for --eval-only.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional result-folder suffix for --eval-only. Example: 'test' writes '<network>_test'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest_model.pt when available, otherwise final_model.pt.",
    )
    parser.add_argument(
        "--resume-policy-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to resume training from.",
    )
    parser.add_argument("--lfp-a-advantage-weight", type=float, default=None)
    parser.add_argument("--ppo-learning-rate", type=float, default=None)
    parser.add_argument("--ppo-ent-coef", type=float, default=None)
    parser.add_argument("--ppo-clip-range", type=float, default=None)
    parser.add_argument("--ppo-vf-coef", type=float, default=None)
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--ppo-n-epochs", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=None)
    parser.add_argument("--ppo-gamma", type=float, default=None)
    parser.add_argument("--ppo-gae-lambda", type=float, default=None)
    parser.add_argument("--ppo-initial-policy-mean", type=float, default=None)
    parser.add_argument("--ppo-initial-policy-std", type=float, default=None)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional training stop after this many completed episodes.",
    )
    parser.add_argument(
        "--lfp-a-decay-schedule",
        type=str,
        choices=("constant", "linear", "exp"),
        default=None,
    )
    parser.add_argument("--lfp-a-decay-rollouts", type=int, default=None)
    parser.add_argument("--lfp-a-advantage-clip", type=float, default=None)
    args = parser.parse_args()
    Config.configure_network(args.network)
    apply_lfpg_rl_overrides(args)
    if args.train_only and args.eval_only:
        raise ValueError("--train-only and --eval-only cannot be used together.")
    if args.runtime_seconds is not None:
        Config.MAX_RUNTIME_SECONDS = None if args.runtime_seconds is None else int(args.runtime_seconds)
    if args.num_envs is not None:
        Config.NUM_ENVS = int(args.num_envs)
    if args.use_subproc is not None:
        Config.USE_SUBPROC = bool(args.use_subproc)
    if args.test_max_scenarios is not None:
        Config.TEST_MAX_SCENARIOS = int(args.test_max_scenarios)

    if args.eval_only:
        device = select_device()
        evaluate_saved_policy_for_trial(
            trial_index=int(args.trial),
            device=device,
            policy_path=args.policy_path,
            selected_test_scenario_ids=args.selected_test_scenario_ids,
            output_suffix=args.output_suffix,
        )
        return

    if Config.MAX_RUNTIME_SECONDS is None:
        raise ValueError(
            "Missing training runtime. Set DEFAULT_MAX_RUNTIME_SECONDS in train.py, "
            "run through train.py, or pass --runtime-seconds directly."
        )
    if Config.NUM_ENVS is None:
        raise ValueError(
            "Missing vectorized environment count. Set DEFAULT_NUM_ENVS in train.py, "
            "run through train.py, or pass --num-envs directly."
        )
    if int(Config.NUM_ENVS) <= 0:
        raise ValueError("The vectorized environment count must be positive.")
    if Config.USE_SUBPROC is None:
        raise ValueError(
            "Missing subprocess-vectorization setting. Set DEFAULT_USE_SUBPROC in train.py "
            "or export DODE_USE_SUBPROC before running this method directly."
        )

    run_one_trial(
        trial_index=int(args.trial),
        num_envs=int(Config.NUM_ENVS),
        seed_override=args.seed,
        evaluate_after_training=not bool(args.train_only),
        selected_test_scenario_ids=args.selected_test_scenario_ids,
        resume_training=bool(args.resume),
        resume_policy_path=args.resume_policy_path,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()



