from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from .config import Config
from dnl.main import build_default_model
from dnl.ltm import ForwardDUOSimulator
from dnl.model import AssignmentResult
from utils import ScenarioDataset


class DNLTrainingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        action_low: float = 0.0,
        action_high: float = 80.0,
        scenario_dataset_dir: str | Path | None = None,
        scenario_split: str = "train",
        fixed_scenario_id: str | None = None,
        fixed_simulation_seed: int | None = None,
        seed: Optional[int] = None,
        record_temporal_inflows: bool | None = None,
    ) -> None:
        super().__init__()
        self.record_temporal_inflows = record_temporal_inflows

        self.scenario_dataset = (
            ScenarioDataset(scenario_dataset_dir) if scenario_dataset_dir is not None else None
        )
        self.scenario_split = str(scenario_split)
        self.fixed_scenario_id = None if fixed_scenario_id is None else str(fixed_scenario_id)
        self.fixed_simulation_seed = None if fixed_simulation_seed is None else int(fixed_simulation_seed)
        self.model = self._build_model(random_seed=0)
        self.observed_link_indices = (
            np.asarray(self.scenario_dataset.observed_link_indices, dtype=np.int64)
            if self.scenario_dataset is not None
            else np.zeros(0, dtype=np.int64)
        )
        self.observation_labels = (
            tuple()
            if self.scenario_dataset is None
            else tuple(getattr(self.scenario_dataset, "observation_labels", tuple()))
        )
        self.num_steps = (
            0 if self.scenario_dataset is None else int(self.scenario_dataset.num_steps)
        )
        self.num_links = (
            0 if self.scenario_dataset is None else int(self.scenario_dataset.num_links)
        )
        self.num_observations = int(self.observed_link_indices.shape[0])
        self.target_observations = np.zeros((self.num_steps, self.num_observations), dtype=np.float32)
        self.num_od = len(self.model.od_pairs)
        self.od_labels = tuple(f"{origin}->{destination}" for origin, destination in self.model.od_pairs)
        self.link_labels = tuple(link.label for link in self.model.network.links)

        if self.scenario_dataset is not None and self.link_labels != tuple(self.scenario_dataset.link_labels):
            raise ValueError(
                "Scenario dataset link labels do not match the current DNL network links. "
                f"Expected {self.link_labels}, got {tuple(self.scenario_dataset.link_labels)}."
            )

        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.capacity = np.array([link.capacity for link in self.model.network.links], dtype=np.float32)
        self.storage = np.array([link.jam_storage for link in self.model.network.links], dtype=np.float32)
        self.free_flow_steps = self.model.loader.free_flow_steps.astype(np.float32)
        input_multiplier = float(Config.RL_RUNTIME_PARAMS.get("flow_scale_multiplier", 1.0))
        reward_multiplier = float(Config.RL_RUNTIME_PARAMS.get("reward_flow_scale_multiplier", 1.0))
        if not np.isfinite([input_multiplier, reward_multiplier]).all() or min(input_multiplier, reward_multiplier) <= 0:
            raise ValueError("Input and reward flow-scale multipliers must be finite and positive.")
        internal_flow_scale = np.maximum(self.capacity, 1.0)
        self.flow_scale = internal_flow_scale * input_multiplier
        self.reward_flow_scale = internal_flow_scale * reward_multiplier
        self.storage_scale = np.maximum(self.storage, 1.0)
        self.observation_scale = self._build_observation_scale()
        self.reward_observation_scale = self.reward_flow_scale[self.observed_link_indices].copy()
        self.policy_action_low = self.action_low
        self.policy_action_high = self.action_high
        self.include_target_observation_state = bool(
            Config.RL_RUNTIME_PARAMS.get("include_target_observation_state", False)
        )
        target_observation_state_dim = self.num_observations if self.include_target_observation_state else 0

        self.action_space = spaces.Box(
            low=np.full(self.num_od, self.policy_action_low, dtype=np.float32),
            high=np.full(self.num_od, self.policy_action_high, dtype=np.float32),
            shape=(self.num_od,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(1 + 3 * self.num_links + target_observation_state_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.estimated_od_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.last_link_flows = np.zeros(self.num_links, dtype=np.float32)
        self.last_occupancies = np.zeros(self.num_links, dtype=np.float32)
        self.last_speed_index = np.ones(self.num_links, dtype=np.float32)
        self.last_result: AssignmentResult | None = None
        self.duo_runtime: ForwardDUOSimulator | None = None
        self.episode_reward = 0.0
        self.completed_episode_payload: dict[str, np.ndarray] | None = None
        self.current_scenario_id: str | None = None
        self.current_generation_seed: int | None = None
        self.current_simulation_seed: int | None = None

        if seed is not None:
            self.reset(seed=seed)

    def _build_model(self, random_seed: int | None) -> Any:
        return build_default_model(
            network_name=Config.NETWORK_NAME,
            route_choice_mode=Config.ROUTE_CHOICE_MODE,
            stochastic_logit_scale=Config.STOCHASTIC_LOGIT_SCALE,
            sample_route_choices=Config.DNL_SAMPLE_ROUTE_CHOICES,
            route_choice_sampling_unit=Config.DNL_ROUTE_CHOICE_SAMPLING_UNIT,
            random_seed=random_seed,
            use_parallel_kernels=Config.DNL_PARALLEL_KERNELS,
            numba_threads=Config.DNL_NUMBA_THREADS,
            record_temporal_inflows=self._needs_temporal_link_inflows(),
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.scenario_dataset is not None:
            if self.fixed_scenario_id is not None:
                scenario = self.scenario_dataset.load(self.scenario_split, self.fixed_scenario_id)
            else:
                scenario = self.scenario_dataset.sample(self.scenario_split, self.np_random)
            self.observed_link_indices = np.asarray(scenario.observed_link_indices, dtype=np.int64)
            self.target_observations = np.asarray(scenario.target_observations, dtype=np.float32)
            if self.target_observations.shape != (self.num_steps, self.num_observations):
                raise ValueError(
                    "target_observations shape mismatch: "
                    f"expected {(self.num_steps, self.num_observations)}, got {self.target_observations.shape}."
                )
            self.current_scenario_id = str(scenario.scenario_id)
            self.current_generation_seed = int(scenario.generation_seed)
        else:
            self.current_scenario_id = None
            self.current_generation_seed = None

        if self.fixed_simulation_seed is not None:
            self.current_simulation_seed = int(self.fixed_simulation_seed)
        else:
            self.current_simulation_seed = int(self.np_random.integers(0, 2**31 - 1))
        reset_start = time.perf_counter()
        self._log_progress(
            "reset-start "
            f"split={self.scenario_split if self.scenario_dataset is not None else 'static'} "
            f"scenario={self.current_scenario_id} generation_seed={self.current_generation_seed} "
            f"simulation_seed={self.current_simulation_seed} "
            f"num_steps={self.num_steps} num_od={self.num_od} num_links={self.num_links} "
            f"observed_links={int(self.observed_link_indices.shape[0])} "
            f"num_observations={self.num_observations} "
            f"record_temporal={self._needs_temporal_link_inflows()}"
        )
        self.model.set_random_seed(self.current_simulation_seed)

        self.current_step = 0
        self.estimated_od_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.last_link_flows = np.zeros(self.num_links, dtype=np.float32)
        self.last_occupancies = np.zeros(self.num_links, dtype=np.float32)
        self.last_speed_index = np.ones(self.num_links, dtype=np.float32)
        self.last_result = None
        duo_start = time.perf_counter()
        self.duo_runtime = self.model.make_duo_runtime(self.num_steps) if self.model.route_choice_mode == "duo" else None
        self._log_progress(
            "reset-done "
            f"mode={self.model.route_choice_mode} "
            f"model_seed_s={duo_start - reset_start:.3f} "
            f"runtime_init_s={time.perf_counter() - duo_start:.3f}"
        )
        self.episode_reward = 0.0

        return self._build_observation(), {
            "num_steps": self.num_steps,
            "num_links": self.num_links,
            "num_od": self.num_od,
            "scenario_id": self.current_scenario_id,
            "scenario_split": self.scenario_split if self.scenario_dataset is not None else None,
            "num_observed_links": int(self.observed_link_indices.shape[0]),
            "observed_link_indices": self.observed_link_indices.copy(),
        }

    def step(self, action: np.ndarray):
        policy_action = np.asarray(action, dtype=np.float32).reshape(self.num_od)
        policy_action = np.clip(policy_action, self.policy_action_low, self.policy_action_high)
        action = np.clip(policy_action, self.action_low, self.action_high).astype(np.float32)

        self.estimated_od_matrix[self.current_step] = action
        step_index = self.current_step
        dnl_start = time.perf_counter()
        self._log_progress(
            "step-start "
            f"step={step_index + 1}/{self.num_steps} mode={self.model.route_choice_mode} "
            f"action_sum={float(np.sum(action)):.3f} "
            f"action_mean={float(np.mean(action)):.3f} "
            f"action_max={float(np.max(action)):.3f} "
            f"action_nonzero={int(np.count_nonzero(action > 0.0))}/{self.num_od} "
            f"target_observed_sum={self._target_measurement_sum(step_index):.3f}"
        )
        if self.model.route_choice_mode == "duo":
            if self.duo_runtime is None:
                raise RuntimeError("DUO runtime was not initialized. Call reset() before step().")
            duo_step = self.duo_runtime.step(action)
            self.last_result = None
            self.last_link_flows = duo_step.link_inflow_row.astype(np.float32)
            self.last_occupancies = duo_step.link_occupancy_row.astype(np.float32)
            self.last_speed_index = self._compute_speed_index(duo_step.snapshot_link_travel_times)
        else:
            self.last_result = self.model.solve(self.estimated_od_matrix[: step_index + 1])
            self.last_link_flows = self.last_result.link_inflows[step_index].astype(np.float32)
            self.last_occupancies = self.last_result.link_occupancies[step_index].astype(np.float32)
            self.last_speed_index = self._compute_speed_index(self.last_result.link_travel_times[step_index])

        step_mse, step_mae, step_normalized_mse = self._compute_step_metrics(step_index)
        reward = -step_normalized_mse
        self.episode_reward += reward
        self._log_progress(
            "step-done "
            f"step={step_index + 1}/{self.num_steps} "
            f"dnl_s={time.perf_counter() - dnl_start:.3f} "
            f"sim_flow_sum={float(np.sum(self.last_link_flows)):.3f} "
            f"sim_flow_max={float(np.max(self.last_link_flows)):.3f} "
            f"observed_mae={step_mae:.3f} "
            f"observed_normalized_mse={step_normalized_mse:.6f} reward={reward:.6f}"
        )

        self.current_step += 1
        terminated = self.current_step >= self.num_steps
        truncated = False

        info: dict[str, Any] = {
            "step_mse": step_mse,
            "step_mae": step_mae,
            "step_normalized_mse": step_normalized_mse,
            "num_observed_links": int(self.observed_link_indices.shape[0]),
            "observed_link_indices": self.observed_link_indices.copy(),
        }
        if terminated:
            if self.model.route_choice_mode == "duo":
                if self.duo_runtime is None:
                    raise RuntimeError("DUO runtime was not initialized. Call reset() before step().")
                finalize_start = time.perf_counter()
                self._log_progress("finalize-start")
                self.last_result = self.model.finalize_duo_runtime(self.duo_runtime)
                self._log_progress(
                    "finalize-done "
                    f"finalize_s={time.perf_counter() - finalize_start:.3f} "
                    f"full_flow_shape={self.last_result.full_link_inflows.shape} "
                    f"temporal_shape={self.last_result.temporal_link_inflows.shape}"
                )
            episode_mse, episode_mae, episode_normalized_mse = self._compute_episode_metrics(
                self.last_result.link_inflows
            )
            info.update(
                {
                    "episode_reward": float(self.episode_reward),
                    "episode_mse": episode_mse,
                    "episode_mae": episode_mae,
                    "episode_normalized_mse": episode_normalized_mse,
                    "estimated_od_matrix": self.estimated_od_matrix.copy(),
                    "simulated_link_flows": self.last_result.link_inflows.copy(),
                    "simulated_observations": self._compute_observations(self.last_result.link_inflows),
                    "target_observations": self.target_observations.copy(),
                    "observation_labels": self.observation_labels,
                    "observation_scale": self.reward_observation_scale.copy(),
                    "od_labels": self.od_labels,
                    "link_labels": self.link_labels,
                    "route_choice_model": self.last_result.route_choice_model,
                    "logit_scale": self.last_result.logit_scale,
                    "scenario_id": self.current_scenario_id,
                    "scenario_split": self.scenario_split if self.scenario_dataset is not None else None,
                    "scenario_generation_seed": self.current_generation_seed,
                    "simulation_seed": self.current_simulation_seed,
                    "num_observed_links": int(self.observed_link_indices.shape[0]),
                    "observed_link_indices": self.observed_link_indices.copy(),
                    "policy_action_low": float(self.policy_action_low),
                    "policy_action_high": float(self.policy_action_high),
                }
            )
            payload_start = time.perf_counter()
            self._log_progress("payload-store-start")
            self.completed_episode_payload = {
                "temporal_link_inflows": self.last_result.temporal_link_inflows,
                "flow_scale": self.reward_flow_scale.copy(),
                "observed_link_indices": self.observed_link_indices.copy(),
                "target_observations": self.target_observations.copy(),
                "observation_scale": self.reward_observation_scale.copy(),
            }
            self._log_progress(f"payload-store-done payload_s={time.perf_counter() - payload_start:.3f}")

        return self._build_observation(), float(reward), terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        if self.current_step >= self.num_steps:
            target_measurement_norm = np.zeros(self.num_observations, dtype=np.float32)
            time_feature = np.array([1.0], dtype=np.float32)
        else:
            target_measurement_norm = self._build_target_measurement_state(self.current_step)
            time_feature = np.array([self.current_step / max(self.num_steps - 1, 1)], dtype=np.float32)

        simulated_flow_norm = self.last_link_flows / self.flow_scale
        occupancy_norm = self.last_occupancies / self.storage_scale

        observation_parts = [time_feature]
        if self.include_target_observation_state:
            observation_parts.append(target_measurement_norm.astype(np.float32))
        observation_parts.extend(
            [
                simulated_flow_norm.astype(np.float32),
                occupancy_norm.astype(np.float32),
                self.last_speed_index.astype(np.float32),
            ]
        )
        return np.concatenate(tuple(observation_parts), dtype=np.float32)


    def _compute_speed_index(self, link_travel_times_row: np.ndarray) -> np.ndarray:
        link_travel_times_row = np.asarray(link_travel_times_row, dtype=np.float32)
        return np.clip(self.free_flow_steps / np.maximum(link_travel_times_row, self.free_flow_steps), 0.0, 1.0)

    def _build_observation_scale(self) -> np.ndarray:
        scale = self.flow_scale[self.observed_link_indices]
        return np.maximum(scale, 1.0).astype(np.float32)

    def _compute_observations(self, link_flows: np.ndarray) -> np.ndarray:
        link_flows = np.asarray(link_flows, dtype=np.float32)
        if link_flows.ndim == 1:
            return link_flows[self.observed_link_indices].astype(np.float32)
        return link_flows[:, self.observed_link_indices].astype(np.float32)

    def _target_measurement_sum(self, step_index: int) -> float:
        return float(np.sum(self.target_observations[step_index]))

    def _build_target_measurement_state(self, step_index: int) -> np.ndarray:
        if step_index < 0 or step_index >= self.num_steps:
            return np.zeros(self.num_observations, dtype=np.float32)
        target = self.target_observations[step_index]
        return (np.asarray(target, dtype=np.float32) / self.observation_scale).astype(np.float32)

    def _compute_step_metrics(
        self,
        step_index: int,
    ) -> tuple[float, float, float]:
        target = self.target_observations[step_index]
        simulated = self._compute_observations(self.last_link_flows)
        error = simulated - target
        normalized_error = error / self.reward_observation_scale
        return (
            float(np.mean(error ** 2)),
            float(np.mean(np.abs(error))),
            float(np.mean(normalized_error ** 2)),
        )

    def _compute_episode_metrics(self, link_inflows: np.ndarray) -> tuple[float, float, float]:
        target = self.target_observations
        simulated = self._compute_observations(link_inflows)
        error = simulated - target
        normalized_error = error / self.reward_observation_scale[None, :]
        return (
            float(np.mean(error ** 2)),
            float(np.mean(np.abs(error))),
            float(np.mean(normalized_error ** 2)),
        )

    def _needs_temporal_link_inflows(self) -> bool:
        if self.record_temporal_inflows is not None:
            return bool(self.record_temporal_inflows)
        params = getattr(Config, "RL_RUNTIME_PARAMS", {})
        return bool(params.get("lfp_a_enabled", True))

    def _log_progress(self, message: str) -> None:
        if not bool(getattr(Config, "DNL_PROGRESS_LOGGING", False)):
            return
        print(
            f"[dnl-progress pid={os.getpid()} method={Config.EXPERIMENT_NAME} "
            f"network={Config.NETWORK_NAME}] {message}",
            flush=True,
        )

    def render(self):
        print(
            f"step={self.current_step}/{self.num_steps}, "
            f"reward={self.episode_reward:.6f}, "
            f"last_flow_mse={float(np.mean(self.last_link_flows ** 2)):.6f}"
        )

    def consume_completed_episode_payload(self) -> dict[str, np.ndarray] | None:
        payload = self.completed_episode_payload
        self.completed_episode_payload = None
        return payload

    def get_flow_scale(self) -> np.ndarray:
        return self.reward_flow_scale.copy()

    def close(self):
        return None
