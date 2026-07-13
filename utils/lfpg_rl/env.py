from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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
from utils.assignment_guidance import solve_assignment_gradient_step
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
    ) -> None:
        super().__init__()

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
        self.flow_scale = np.maximum(self.capacity, 1.0)
        self.storage_scale = np.maximum(self.storage, 1.0)
        self.observation_scale = self._build_observation_scale()
        self.coarse_residual_policy_enabled = bool(
            Config.RL_RUNTIME_PARAMS.get("coarse_residual_policy_enabled", False)
        )
        self.include_coarse_od_state = bool(
            self.coarse_residual_policy_enabled
            and Config.RL_RUNTIME_PARAMS.get("include_coarse_od_state", True)
        )
        self.residual_action_high = float(
            Config.RL_RUNTIME_PARAMS.get(
                "residual_action_high",
                max((self.action_high - self.action_low) * 0.15, 1.0),
            )
        )
        if self.residual_action_high <= 0.0:
            raise ValueError("residual_action_high must be positive when coarse residual policy is enabled.")
        self.policy_action_low = -self.residual_action_high if self.coarse_residual_policy_enabled else self.action_low
        self.policy_action_high = self.residual_action_high if self.coarse_residual_policy_enabled else self.action_high
        self.include_target_observation_state = bool(
            Config.RL_RUNTIME_PARAMS.get("include_target_observation_state", False)
        )
        target_observation_state_dim = self.num_observations if self.include_target_observation_state else 0
        coarse_state_dim = self.num_od if self.include_coarse_od_state else 0

        self.action_space = spaces.Box(
            low=np.full(self.num_od, self.policy_action_low, dtype=np.float32),
            high=np.full(self.num_od, self.policy_action_high, dtype=np.float32),
            shape=(self.num_od,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(1 + 3 * self.num_links + target_observation_state_dim + coarse_state_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.estimated_od_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.last_link_flows = np.zeros(self.num_links, dtype=np.float32)
        self.last_occupancies = np.zeros(self.num_links, dtype=np.float32)
        self.last_speed_index = np.ones(self.num_links, dtype=np.float32)
        self.last_result: AssignmentResult | None = None
        self.duo_runtime: ForwardDUOSimulator | None = None
        self.coarse_od_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.policy_residual_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.current_coarse_action = np.zeros(self.num_od, dtype=np.float32)
        self.current_coarse_info: dict[str, Any] = {}
        self._coarse_action_step_index: int | None = None
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
        self.coarse_od_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.policy_residual_matrix = np.zeros((self.num_steps, self.num_od), dtype=np.float32)
        self.current_coarse_action = np.zeros(self.num_od, dtype=np.float32)
        self.current_coarse_info = {}
        self._coarse_action_step_index = None

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
        if self.coarse_residual_policy_enabled:
            coarse_action = self._get_current_coarse_action()
            residual_action = policy_action.astype(np.float32, copy=True)
            action = np.clip(coarse_action + policy_action, self.action_low, self.action_high).astype(np.float32)
        else:
            coarse_action = np.zeros(self.num_od, dtype=np.float32)
            residual_action = np.zeros(self.num_od, dtype=np.float32)
            action = np.clip(policy_action, self.action_low, self.action_high).astype(np.float32)

        self.estimated_od_matrix[self.current_step] = action
        self.coarse_od_matrix[self.current_step] = coarse_action
        self.policy_residual_matrix[self.current_step] = residual_action
        step_index = self.current_step
        dnl_start = time.perf_counter()
        self._log_progress(
            "step-start "
            f"step={step_index + 1}/{self.num_steps} mode={self.model.route_choice_mode} "
            f"action_sum={float(np.sum(action)):.3f} "
            f"action_mean={float(np.mean(action)):.3f} "
            f"action_max={float(np.max(action)):.3f} "
            f"coarse_sum={float(np.sum(coarse_action)):.3f} "
            f"residual_sum={float(np.sum(residual_action)):.3f} "
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
                    "coarse_od_matrix": self.coarse_od_matrix.copy(),
                    "policy_residual_matrix": self.policy_residual_matrix.copy(),
                    "simulated_link_flows": self.last_result.link_inflows.copy(),
                    "simulated_observations": self._compute_observations(self.last_result.link_inflows),
                    "target_observations": self.target_observations.copy(),
                    "observation_labels": self.observation_labels,
                    "observation_scale": self.observation_scale.copy(),
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
                    "coarse_residual_policy_enabled": bool(self.coarse_residual_policy_enabled),
                    "policy_action_low": float(self.policy_action_low),
                    "policy_action_high": float(self.policy_action_high),
                }
            )
            payload_start = time.perf_counter()
            self._log_progress("payload-store-start")
            self.completed_episode_payload = {
                "temporal_link_inflows": self.last_result.temporal_link_inflows,
                "flow_scale": self.flow_scale.copy(),
                "observed_link_indices": self.observed_link_indices.copy(),
                "target_observations": self.target_observations.copy(),
                "observation_scale": self.observation_scale.copy(),
            }
            self._log_progress(f"payload-store-done payload_s={time.perf_counter() - payload_start:.3f}")

        return self._build_observation(), float(reward), terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        if self.current_step >= self.num_steps:
            target_measurement_norm = np.zeros(self.num_observations, dtype=np.float32)
            time_feature = np.array([1.0], dtype=np.float32)
            coarse_action = np.zeros(self.num_od, dtype=np.float32)
        else:
            target_measurement_norm = self._build_target_measurement_state(self.current_step)
            time_feature = np.array([self.current_step / max(self.num_steps - 1, 1)], dtype=np.float32)
            coarse_action = self._get_current_coarse_action()

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
        if self.include_coarse_od_state:
            coarse_action_norm = coarse_action / max(self.action_high, 1.0)
            observation_parts.append(coarse_action_norm.astype(np.float32))
        return np.concatenate(tuple(observation_parts), dtype=np.float32)

    def _get_current_coarse_action(self) -> np.ndarray:
        if not self.coarse_residual_policy_enabled or self.current_step >= self.num_steps:
            return np.zeros(self.num_od, dtype=np.float32)
        if self._coarse_action_step_index == int(self.current_step):
            return self.current_coarse_action.copy()

        params = dict(Config.RL_RUNTIME_PARAMS.get("coarse_solver_params", {}))
        params.setdefault("max_iterations", 2)
        params.setdefault("max_line_search_steps", 2)
        params.setdefault("warm_start", "previous")
        feedback_enabled = bool(Config.RL_RUNTIME_PARAMS.get("coarse_solver_feedback_enabled", True))
        target_dataset = SimpleNamespace(
            target_observations=np.asarray(self.target_observations, dtype=np.float32),
            observed_link_indices=np.asarray(self.observed_link_indices, dtype=np.int64),
            num_steps=int(self.num_steps),
        )
        context = SimpleNamespace(
            model=self.model,
            target_dataset=target_dataset,
            scenario_id=self.current_scenario_id,
            step_index=int(self.current_step),
            estimated_od_matrix=np.asarray(self.estimated_od_matrix, dtype=np.float64),
            flow_scale=np.asarray(self.flow_scale, dtype=np.float64),
            locked_runtime=self.duo_runtime,
            action_low=float(self.action_low),
            action_high=float(self.action_high),
            rng=self.np_random,
            runtime_exceeded=lambda: False,
        )
        coarse_action, info = solve_assignment_gradient_step(
            context,
            params,
            feedback_enabled=feedback_enabled,
        )
        self.current_coarse_action = np.clip(
            np.asarray(coarse_action, dtype=np.float32).reshape(self.num_od),
            self.action_low,
            self.action_high,
        )
        self.current_coarse_info = dict(info)
        self._coarse_action_step_index = int(self.current_step)
        return self.current_coarse_action.copy()

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
        normalized_error = error / self.observation_scale
        return (
            float(np.mean(error ** 2)),
            float(np.mean(np.abs(error))),
            float(np.mean(normalized_error ** 2)),
        )

    def _compute_episode_metrics(self, link_inflows: np.ndarray) -> tuple[float, float, float]:
        target = self.target_observations
        simulated = self._compute_observations(link_inflows)
        error = simulated - target
        normalized_error = error / self.observation_scale[None, :]
        return (
            float(np.mean(error ** 2)),
            float(np.mean(np.abs(error))),
            float(np.mean(normalized_error ** 2)),
        )

    def _needs_temporal_link_inflows(self) -> bool:
        params = getattr(Config, "RL_RUNTIME_PARAMS", {})
        return (
            bool(params.get("lfp_a_enabled", True))
            or bool(params.get("coarse_residual_policy_enabled", False))
        )

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
        return self.flow_scale.copy()

    def close(self):
        return None
