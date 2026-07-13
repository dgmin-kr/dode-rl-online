"""Shared sequential test-loop support for online non-RL baselines."""

from __future__ import annotations

import csv
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from dnl.ltm import ForwardDUOSimulator
from dnl.model import AssignmentResult, DynamicNetworkLoadingModel
from .result_io import (
    TEST_OUTPUTS_NPZ_NAME,
    TEST_STEP_HISTORY_CSV_NAME,
    TestOutputRecord,
    write_test_outputs_npz,
    write_test_step_history_csv,
)
from .baseline_evaluation import (
    MatrixEvaluation,
    StepObjectiveEvaluation,
    build_model_from_config_with_seed,
    build_target_dataset_from_arrays,
    compute_flow_scale,
    compute_observed_step_error_metrics,
    copy_runtime_for_single_step_candidate,
    evaluate_assignment_result,
    evaluate_step_action,
    finalize_duo_or_full_result,
)


def reset_trial_output_dir(output_dir: Path) -> None:
    """Remove stale files for one method/trial result directory before rerun."""

    output_dir = Path(output_dir)
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = output_parent.resolve()

    if output_dir.exists():
        resolved_output = output_dir.resolve()
        if resolved_output.parent != resolved_parent:
            raise RuntimeError(f"Refusing to delete unexpected result path: {resolved_output}")
        if output_dir.is_dir():
            shutil.rmtree(resolved_output)
        else:
            output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StepLockContext:
    model: DynamicNetworkLoadingModel
    target_dataset: Any
    scenario_id: str
    step_index: int
    estimated_od_matrix: np.ndarray
    flow_scale: np.ndarray
    locked_runtime: ForwardDUOSimulator | None
    action_low: float
    action_high: float
    rng: np.random.Generator
    current_link_flow_row: np.ndarray
    current_link_occupancy_row: np.ndarray
    current_speed_index_row: np.ndarray
    scenario_started_at: float | None
    scenario_deadline_time: float | None
    step_started_at: float | None
    step_deadline_time: float | None

    def remaining_runtime_seconds(self) -> float | None:
        if self.step_deadline_time is None:
            return None
        return max(0.0, float(self.step_deadline_time) - float(time.perf_counter()))

    def runtime_exceeded(self) -> bool:
        remaining_runtime_seconds = self.remaining_runtime_seconds()
        if remaining_runtime_seconds is None:
            return False
        return bool(remaining_runtime_seconds <= 0.0)

    def remaining_scenario_runtime_seconds(self) -> float | None:
        if self.scenario_deadline_time is None:
            return None
        return max(0.0, float(self.scenario_deadline_time) - float(time.perf_counter()))

    def scenario_runtime_exceeded(self) -> bool:
        remaining_runtime_seconds = self.remaining_scenario_runtime_seconds()
        if remaining_runtime_seconds is None:
            return False
        return bool(remaining_runtime_seconds <= 0.0)


@dataclass(frozen=True)
class SequentialScenarioRun:
    scenario_id: str
    scenario_generation_seed: int
    simulation_seed: int
    estimated_od_matrix: np.ndarray
    evaluation: MatrixEvaluation
    observed_link_indices: np.ndarray
    step_rows: tuple[dict[str, Any], ...]
    elapsed_seconds: float


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def build_step_context(
    *,
    model: DynamicNetworkLoadingModel,
    target_dataset: Any,
    scenario_id: str,
    step_index: int,
    estimated_od_matrix: np.ndarray,
    flow_scale: np.ndarray,
    locked_runtime: ForwardDUOSimulator | None,
    action_low: float,
    action_high: float,
    rng: np.random.Generator,
    scenario_started_at: float | None,
    scenario_deadline_time: float | None,
    step_started_at: float | None,
    step_deadline_time: float | None,
) -> StepLockContext:
    current_link_flow_row, current_link_occupancy_row, current_speed_index_row = _build_current_state_rows(
        model=model,
        estimated_od_matrix=estimated_od_matrix,
        step_index=step_index,
        locked_runtime=locked_runtime,
    )
    return StepLockContext(
        model=model,
        target_dataset=target_dataset,
        scenario_id=str(scenario_id),
        step_index=int(step_index),
        estimated_od_matrix=np.asarray(estimated_od_matrix, dtype=np.float64),
        flow_scale=np.asarray(flow_scale, dtype=np.float64),
        locked_runtime=locked_runtime,
        action_low=float(action_low),
        action_high=float(action_high),
        rng=rng,
        current_link_flow_row=current_link_flow_row,
        current_link_occupancy_row=current_link_occupancy_row,
        current_speed_index_row=current_speed_index_row,
        scenario_started_at=scenario_started_at,
        scenario_deadline_time=scenario_deadline_time,
        step_started_at=step_started_at,
        step_deadline_time=step_deadline_time,
    )


def _compute_speed_index(model: DynamicNetworkLoadingModel, link_travel_times_row: np.ndarray) -> np.ndarray:
    free_flow_steps = np.asarray(model.loader.free_flow_steps, dtype=np.float64)
    link_travel_times_row = np.asarray(link_travel_times_row, dtype=np.float64)
    return np.clip(free_flow_steps / np.maximum(link_travel_times_row, free_flow_steps), 0.0, 1.0)


def _build_current_state_rows(
    *,
    model: DynamicNetworkLoadingModel,
    estimated_od_matrix: np.ndarray,
    step_index: int,
    locked_runtime: ForwardDUOSimulator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_links = len(model.network.links)
    if int(step_index) <= 0:
        return (
            np.zeros(num_links, dtype=np.float64),
            np.zeros(num_links, dtype=np.float64),
            np.ones(num_links, dtype=np.float64),
        )

    if model.route_choice_mode == "duo":
        if locked_runtime is None:
            raise RuntimeError("DUO sequential calibration requires a locked runtime state.")
        if hasattr(locked_runtime, "current_state_rows"):
            flow_row, occupancy_row, speed_index_row = locked_runtime.current_state_rows()
            return (
                np.asarray(flow_row, dtype=np.float64).copy(),
                np.asarray(occupancy_row, dtype=np.float64).copy(),
                np.asarray(speed_index_row, dtype=np.float64).copy(),
            )
        current_time = int(step_index)
        occupancy_row = (
            np.asarray(locked_runtime.workspace.cumulative_inflows[current_time], dtype=np.float64)
            - np.asarray(locked_runtime.workspace.cumulative_outflows[current_time], dtype=np.float64)
        )
        flow_row = np.asarray(locked_runtime.workspace.link_inflows[current_time - 1], dtype=np.float64)
        snapshot_link_travel_times = model.loader._estimate_snapshot_link_travel_times(
            cumulative_inflows=locked_runtime.workspace.cumulative_inflows,
            cumulative_outflows=locked_runtime.workspace.cumulative_outflows,
            t=current_time,
        )
        speed_index_row = _compute_speed_index(model, snapshot_link_travel_times)
        return flow_row.copy(), occupancy_row.copy(), speed_index_row.copy()

    partial_result = model.solve(np.asarray(estimated_od_matrix[: int(step_index)], dtype=np.float64))
    flow_row = np.asarray(partial_result.link_inflows[step_index - 1], dtype=np.float64)
    occupancy_row = np.asarray(partial_result.link_occupancies[step_index - 1], dtype=np.float64)
    speed_index_row = _compute_speed_index(model, partial_result.link_travel_times[step_index - 1])
    return flow_row.copy(), occupancy_row.copy(), speed_index_row.copy()


def evaluate_candidate_step(context: StepLockContext, action_row: np.ndarray) -> StepObjectiveEvaluation:
    return evaluate_step_action(
        model=context.model,
        target_dataset=context.target_dataset,
        fixed_od_matrix=context.estimated_od_matrix,
        step_index=context.step_index,
        action_row=action_row,
        flow_scale=context.flow_scale,
        locked_runtime=context.locked_runtime,
    )


def evaluate_candidate_result(context: StepLockContext, action_row: np.ndarray) -> AssignmentResult:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(context.estimated_od_matrix.shape[1])
    if context.model.route_choice_mode == "duo":
        if context.locked_runtime is None:
            raise RuntimeError("DUO sequential calibration requires a locked runtime state.")
        trial_runtime = copy_runtime_for_single_step_candidate(context.locked_runtime)
        step_result = trial_runtime.step(action_row)
        temporal_link_inflows = getattr(trial_runtime.workspace, "temporal_link_inflows", None)
        if temporal_link_inflows is None:
            raise RuntimeError("DUO candidate evaluation requires temporal_link_inflows to be recorded.")
        step_index = int(context.step_index)
        link_inflows = np.zeros((step_index + 1, context.model.network.num_links), dtype=np.float64)
        link_inflows[step_index] = np.asarray(step_result.link_inflow_row, dtype=np.float64)
        return SimpleNamespace(
            link_inflows=link_inflows,
            temporal_link_inflows=np.asarray(
                temporal_link_inflows[:1, :1],
                dtype=np.float64,
            ).copy(),
        )

    step_index = int(context.step_index)
    candidate_od_matrix = np.asarray(context.estimated_od_matrix[: step_index + 1], dtype=np.float64).copy()
    candidate_od_matrix[step_index] = action_row
    return context.model.solve(candidate_od_matrix)


def build_target_dataset_from_scenario(scenario_sample: Any, link_labels: tuple[str, ...]) -> Any:
    return build_target_dataset_from_arrays(
        link_labels=link_labels,
        target_observations=np.asarray(scenario_sample.target_observations, dtype=np.float32),
        observed_link_indices=np.asarray(scenario_sample.observed_link_indices, dtype=np.int64),
        observation_labels=getattr(scenario_sample, "observation_labels", tuple()),
    )


def run_step_locked_scenario(
    *,
    config: Any,
    scenario_sample: Any,
    simulation_seed: int,
    step_solver: Callable[[StepLockContext], tuple[np.ndarray, dict[str, Any]]],
    step_runtime_seconds: float | None = None,
) -> SequentialScenarioRun:
    started_at = time.perf_counter()
    model = build_model_from_config_with_seed(config, random_seed=int(simulation_seed))
    target_dataset = build_target_dataset_from_scenario(
        scenario_sample=scenario_sample,
        link_labels=tuple(model.link_labels),
    )
    resolved_step_runtime_seconds = None
    scenario_runtime_seconds = None
    scenario_deadline_time = None
    if step_runtime_seconds is not None:
        resolved_step_runtime_seconds = max(float(step_runtime_seconds), 0.0)
        scenario_runtime_seconds = resolved_step_runtime_seconds * float(max(int(target_dataset.num_steps), 1))
    flow_scale = compute_flow_scale(model)
    estimated_od_matrix = np.zeros((target_dataset.num_steps, len(model.od_pairs)), dtype=np.float64)
    locked_runtime = model.make_duo_runtime(target_dataset.num_steps) if model.route_choice_mode == "duo" else None
    rng = np.random.default_rng(int(simulation_seed) + 17)
    step_rows: list[dict[str, Any]] = []
    for step_index in range(target_dataset.num_steps):
        step_started_at = time.perf_counter()
        step_deadline_time = None
        if resolved_step_runtime_seconds is not None:
            step_deadline_time = step_started_at + max(float(resolved_step_runtime_seconds), 0.0)
        scenario_remaining_before_step = (
            None
            if scenario_deadline_time is None
            else max(0.0, float(scenario_deadline_time) - float(time.perf_counter()))
        )
        step_remaining_before_step = (
            None
            if step_deadline_time is None
            else max(0.0, float(step_deadline_time) - float(time.perf_counter()))
        )
        context = build_step_context(
            model=model,
            target_dataset=target_dataset,
            scenario_id=str(scenario_sample.scenario_id),
            step_index=step_index,
            estimated_od_matrix=estimated_od_matrix,
            flow_scale=flow_scale,
            locked_runtime=locked_runtime,
            action_low=float(config.ACTION_LOW),
            action_high=float(config.ACTION_HIGH),
            rng=rng,
            scenario_started_at=started_at,
            scenario_deadline_time=scenario_deadline_time,
            step_started_at=step_started_at,
            step_deadline_time=step_deadline_time,
        )
        if context.runtime_exceeded():
            action_row = _build_timeout_fallback_action(
                estimated_od_matrix=estimated_od_matrix,
                step_index=step_index,
                action_low=float(config.ACTION_LOW),
                action_high=float(config.ACTION_HIGH),
            )
            solver_info = {
                "inner_iterations": 0,
                "inner_evaluations": 0,
                "timed_out": True,
                "timeout_mode": "step_budget_exhausted",
            }
        else:
            action_row, solver_info = step_solver(context)
        action_row = np.clip(
            np.asarray(action_row, dtype=np.float64).reshape(len(model.od_pairs)),
            float(config.ACTION_LOW),
            float(config.ACTION_HIGH),
        )
        estimated_od_matrix[step_index] = action_row

        if model.route_choice_mode == "duo":
            if locked_runtime is None:
                raise RuntimeError("DUO sequential calibration requires a locked runtime state.")
            locked_step = locked_runtime.step(action_row)
            simulated_link_flow_row = np.asarray(locked_step.link_inflow_row, dtype=np.float64)
        else:
            partial_result = model.solve(np.asarray(estimated_od_matrix[: step_index + 1], dtype=np.float64))
            simulated_link_flow_row = np.asarray(partial_result.link_inflows[step_index], dtype=np.float64)

        step_mse, step_mae, step_normalized_mse = compute_observed_step_error_metrics(
            simulated_link_flow_row=simulated_link_flow_row,
            target_dataset=target_dataset,
            step_index=step_index,
            flow_scale=flow_scale,
        )
        scenario_remaining_after_step = (
            None
            if scenario_deadline_time is None
            else max(0.0, float(scenario_deadline_time) - float(time.perf_counter()))
        )
        step_remaining_after_step = (
            None
            if step_deadline_time is None
            else max(0.0, float(step_deadline_time) - float(time.perf_counter()))
        )
        step_rows.append(
            {
                "step_index": int(step_index),
                "scenario_id": str(scenario_sample.scenario_id),
                "step_mse": float(step_mse),
                "step_mae": float(step_mae),
                "step_normalized_mse": float(step_normalized_mse),
                "scenario_runtime_budget_seconds": (
                    None if scenario_runtime_seconds is None else float(scenario_runtime_seconds)
                ),
                "step_runtime_budget_seconds": (
                    None if resolved_step_runtime_seconds is None else float(resolved_step_runtime_seconds)
                ),
                "scenario_remaining_seconds_before_step": scenario_remaining_before_step,
                "scenario_remaining_seconds_after_step": scenario_remaining_after_step,
                "step_remaining_seconds_before_step": step_remaining_before_step,
                "step_remaining_seconds_after_step": step_remaining_after_step,
                "scenario_runtime_exceeded": bool(
                    scenario_deadline_time is not None and float(time.perf_counter()) >= float(scenario_deadline_time)
                ),
                "step_runtime_exceeded": bool(
                    step_deadline_time is not None and float(time.perf_counter()) >= float(step_deadline_time)
                ),
                **dict(solver_info),
            }
        )

    final_result = finalize_duo_or_full_result(
        model=model,
        estimated_od_matrix=estimated_od_matrix,
        locked_runtime=locked_runtime,
    )
    evaluation = evaluate_assignment_result(
        od_matrix=estimated_od_matrix,
        result=final_result,
        target_dataset=target_dataset,
        flow_scale=flow_scale,
    )
    elapsed_seconds = float(time.perf_counter() - started_at)
    return SequentialScenarioRun(
        scenario_id=str(scenario_sample.scenario_id),
        scenario_generation_seed=int(scenario_sample.generation_seed),
        simulation_seed=int(simulation_seed),
        estimated_od_matrix=estimated_od_matrix.copy(),
        evaluation=evaluation,
        observed_link_indices=np.asarray(target_dataset.observed_link_indices, dtype=np.int64),
        step_rows=tuple(step_rows),
        elapsed_seconds=elapsed_seconds,
    )


def _build_timeout_fallback_action(
    *,
    estimated_od_matrix: np.ndarray,
    step_index: int,
    action_low: float,
    action_high: float,
) -> np.ndarray:
    if int(step_index) > 0:
        fallback_action = np.asarray(estimated_od_matrix[int(step_index) - 1], dtype=np.float64)
    else:
        fallback_action = np.zeros(int(estimated_od_matrix.shape[1]), dtype=np.float64)
    return np.clip(fallback_action, float(action_low), float(action_high))


def save_sequential_trial_outputs(
    *,
    output_dir: Path,
    scenario_runs: list[SequentialScenarioRun],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[TestOutputRecord] = []
    for scenario_run in scenario_runs:
        evaluation = scenario_run.evaluation
        records.append(
            TestOutputRecord(
                scenario_id=scenario_run.scenario_id,
                split="test",
                scenario_generation_seed=int(scenario_run.scenario_generation_seed),
                simulation_seed=int(scenario_run.simulation_seed),
                od_labels=tuple(f"{origin}->{destination}" for origin, destination in evaluation.result.od_pairs),
                link_labels=tuple(evaluation.result.link_labels),
                estimated_od_matrix=np.asarray(scenario_run.estimated_od_matrix, dtype=np.float32),
                simulated_link_flows=np.asarray(evaluation.result.link_inflows, dtype=np.float32),
                observed_link_indices=np.asarray(scenario_run.observed_link_indices, dtype=np.int64),
                step_rows=tuple(dict(row) for row in scenario_run.step_rows),
                target_observations=np.asarray(evaluation.target_observations, dtype=np.float32),
                simulated_observations=np.asarray(evaluation.simulated_observations, dtype=np.float32),
                observation_labels=tuple(evaluation.observation_labels),
            )
        )

    write_test_outputs_npz(output_dir / TEST_OUTPUTS_NPZ_NAME, records)
    write_test_step_history_csv(output_dir / TEST_STEP_HISTORY_CSV_NAME, records)


def aggregate_scenario_runs(
    *,
    algorithm_name: str,
    network_name: str,
    trial_index: int,
    scenario_runs: list[SequentialScenarioRun],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {
            "scenario_id": run.scenario_id,
            "mse_mean": float(run.evaluation.mse_mean),
            "mae_mean": float(run.evaluation.mae_mean),
            "normalized_mse": float(run.evaluation.normalized_mse),
            "corr_mean": float(run.evaluation.corr_mean),
            "objective": float(run.evaluation.objective),
            "elapsed_seconds": float(run.elapsed_seconds),
            "scenario_generation_seed": int(run.scenario_generation_seed),
            "simulation_seed": int(run.simulation_seed),
        }
        for run in scenario_runs
    ]
    summary = {
        "algorithm": str(algorithm_name),
        "network_name": str(network_name),
        "trial_index": int(trial_index),
        "num_scenarios": int(len(rows)),
        "mean_mse_mean": float(np.mean([row["mse_mean"] for row in rows])) if rows else float("nan"),
        "mean_mae_mean": float(np.mean([row["mae_mean"] for row in rows])) if rows else float("nan"),
        "mean_normalized_mse": (
            float(np.mean([row["normalized_mse"] for row in rows])) if rows else float("nan")
        ),
        "mean_corr_mean": float(np.mean([row["corr_mean"] for row in rows])) if rows else float("nan"),
        "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in rows])) if rows else float("nan"),
    }
    return rows, summary
