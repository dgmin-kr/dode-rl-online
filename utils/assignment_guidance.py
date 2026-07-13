"""Dynamic-assignment gradient helpers shared by RL and baseline methods."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from dnl.model import AssignmentResult
from .baseline_evaluation import (
    compute_observed_step_error_metrics,
    copy_runtime_for_single_step_candidate,
    resolve_observed_link_indices,
)


def compute_assignment_gradient(
    *,
    od_matrix: np.ndarray,
    temporal_link_inflows: np.ndarray,
    simulated_link_flows: np.ndarray,
    target_observations: np.ndarray,
    observed_link_indices: np.ndarray,
    flow_scale: np.ndarray,
) -> np.ndarray:
    """Return the normalized-MSE gradient with respect to OD demand.

    The only target is detector space:
    simulated_link_flows[:, observed_link_indices] - target_observations.
    No full-link target tensor is used.
    """

    od_matrix = np.asarray(od_matrix, dtype=np.float64)
    temporal_link_inflows = np.asarray(temporal_link_inflows, dtype=np.float64)
    simulated_link_flows = np.asarray(simulated_link_flows, dtype=np.float64)
    target_observations = np.asarray(target_observations, dtype=np.float64)
    flow_scale = np.asarray(flow_scale, dtype=np.float64).reshape(-1)

    horizon = min(
        int(od_matrix.shape[0]),
        int(temporal_link_inflows.shape[0]),
        int(temporal_link_inflows.shape[1]),
        int(simulated_link_flows.shape[0]),
        int(target_observations.shape[0]),
    )
    if horizon <= 0:
        raise ValueError("Assignment-gradient inputs are empty.")
    num_od = int(od_matrix.shape[1])
    num_links = int(simulated_link_flows.shape[1])
    if num_od <= 0 or num_links <= 0:
        raise ValueError("Assignment-gradient OD/link dimensions are empty.")
    if flow_scale.shape[0] < num_links:
        raise ValueError(f"flow_scale must have at least {num_links} values, got {flow_scale.shape[0]}.")
    observed_indices = resolve_observed_link_indices(num_links, observed_link_indices)
    if target_observations.ndim != 2 or target_observations.shape[1] != int(observed_indices.shape[0]):
        raise ValueError(
            "target_observations must have shape [horizon, num_observed_links]; "
            f"got {target_observations.shape} for {int(observed_indices.shape[0])} observed links."
        )

    od_slice = od_matrix[:horizon]
    denominator = od_slice[None, :, :, None]
    safe_flow_scale = np.maximum(flow_scale[:num_links], 1.0)

    observed_count = int(observed_indices.shape[0])
    temporal_observed = temporal_link_inflows[:horizon, :horizon, :, observed_indices]
    link_error = (
        simulated_link_flows[:horizon, observed_indices]
        - target_observations[:horizon]
    )
    observed_sensitivity = (
        2.0
        * link_error
        / np.square(safe_flow_scale[observed_indices])[None, :]
        / float(max(horizon * observed_count, 1))
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        assignment_ratio = np.divide(
            temporal_observed,
            denominator[:horizon],
            out=np.zeros_like(temporal_observed, dtype=np.float64),
            where=denominator[:horizon] > 1e-8,
        )

    gradient = np.einsum("ftok,fk->to", assignment_ratio, observed_sensitivity[:horizon])
    return np.asarray(gradient, dtype=np.float64)


def _candidate_result_feedback(context: Any, action_row: np.ndarray) -> Any:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(context.estimated_od_matrix.shape[1])
    if context.model.route_choice_mode == "duo":
        if context.locked_runtime is None:
            raise RuntimeError("DUO sequential guidance requires a locked runtime state.")
        trial_runtime = copy_runtime_for_single_step_candidate(context.locked_runtime)
        step_index = int(context.step_index)
        step_result = trial_runtime.step(action_row)
        temporal_link_inflows = getattr(trial_runtime.workspace, "temporal_link_inflows", None)
        if temporal_link_inflows is None:
            raise RuntimeError("DUO assignment guidance requires temporal_link_inflows to be recorded.")
        return SimpleNamespace(
            step_index=step_index,
            link_inflow_row=np.asarray(step_result.link_inflow_row, dtype=np.float64).copy(),
            temporal_link_inflows_current=np.asarray(
                temporal_link_inflows[:1, :1],
                dtype=np.float64,
            ).copy(),
        )

    step_index = int(context.step_index)
    candidate_od_matrix = np.asarray(context.estimated_od_matrix[: step_index + 1], dtype=np.float64).copy()
    candidate_od_matrix[step_index] = action_row
    return context.model.solve(candidate_od_matrix)


def _candidate_result_open_loop(context: Any, action_row: np.ndarray) -> AssignmentResult:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(context.estimated_od_matrix.shape[1])
    step_index = int(context.step_index)
    candidate_od_matrix = np.asarray(context.estimated_od_matrix[: step_index + 1], dtype=np.float64).copy()
    candidate_od_matrix[step_index] = action_row
    return context.model.solve(candidate_od_matrix)


def evaluate_assignment_step_candidate(
    context: Any,
    action_row: np.ndarray,
    *,
    feedback_enabled: bool,
) -> tuple[Any, float, float, float]:
    result = (
        _candidate_result_feedback(context, action_row)
        if bool(feedback_enabled)
        else _candidate_result_open_loop(context, action_row)
    )
    step_index = int(context.step_index)
    simulated_link_flow_row = (
        np.asarray(result.link_inflow_row, dtype=np.float64)
        if hasattr(result, "link_inflow_row")
        else np.asarray(result.link_inflows[step_index], dtype=np.float64)
    )
    step_mse, step_mae, step_normalized_mse = compute_observed_step_error_metrics(
        simulated_link_flow_row=simulated_link_flow_row,
        target_dataset=context.target_dataset,
        step_index=step_index,
        flow_scale=np.asarray(context.flow_scale, dtype=np.float64),
    )
    return result, float(step_mse), float(step_mae), float(step_normalized_mse)


def _step_gradient_from_result(context: Any, action_row: np.ndarray, result: Any) -> np.ndarray:
    step_index = int(context.step_index)
    target_dataset = context.target_dataset
    if hasattr(result, "temporal_link_inflows_current"):
        temporal_link_inflows = np.asarray(result.temporal_link_inflows_current, dtype=np.float64)
        simulated_link_flows = np.asarray(result.link_inflow_row, dtype=np.float64).reshape(1, -1)
    else:
        temporal_link_inflows = np.asarray(
            result.temporal_link_inflows[step_index : step_index + 1, step_index : step_index + 1],
            dtype=np.float64,
        )
        simulated_link_flows = np.asarray(result.link_inflows[step_index : step_index + 1], dtype=np.float64)
    gradient = compute_assignment_gradient(
        od_matrix=np.asarray(action_row, dtype=np.float64).reshape(1, -1),
        temporal_link_inflows=temporal_link_inflows,
        simulated_link_flows=simulated_link_flows,
        target_observations=np.asarray(
            target_dataset.target_observations[step_index : step_index + 1],
            dtype=np.float64,
        ),
        observed_link_indices=np.asarray(target_dataset.observed_link_indices, dtype=np.int64),
        flow_scale=np.asarray(context.flow_scale, dtype=np.float64),
    )
    return np.asarray(gradient, dtype=np.float64).reshape(-1)


def _initial_action(context: Any, params: dict[str, Any]) -> np.ndarray:
    strategy = str(params.get("warm_start", "zero")).strip().lower()
    num_od = int(context.estimated_od_matrix.shape[1])
    if strategy == "previous" and int(context.step_index) > 0:
        initial = np.asarray(context.estimated_od_matrix[int(context.step_index) - 1], dtype=np.float64)
    elif strategy == "mean_previous" and int(context.step_index) > 0:
        initial = np.mean(
            np.asarray(context.estimated_od_matrix[: int(context.step_index)], dtype=np.float64),
            axis=0,
        )
    else:
        initial = np.zeros(num_od, dtype=np.float64)
    return np.clip(initial, float(context.action_low), float(context.action_high))


def _resolve_iteration_limit(params: dict[str, Any], *, default: int | None = None) -> int | None:
    raw_value = params.get("max_iterations", default)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        return None
    return value


def _can_run_iteration(iteration_index: int, max_iterations: int | None) -> bool:
    return max_iterations is None or int(iteration_index) < int(max_iterations)


def _remaining_runtime_seconds(context: Any) -> float | None:
    remaining_fn = getattr(context, "remaining_runtime_seconds", None)
    if remaining_fn is None:
        return None
    remaining = remaining_fn()
    if remaining is None:
        return None
    return max(0.0, float(remaining))


def _has_runtime_for_candidate_evaluation(
    context: Any,
    *,
    last_evaluation_seconds: float | None,
) -> bool:
    remaining = _remaining_runtime_seconds(context)
    if remaining is None:
        return True
    if remaining <= 0.0:
        return False
    if last_evaluation_seconds is None:
        return remaining > 1.0
    required = max(1.0, 1.25 * float(last_evaluation_seconds) + 0.5)
    return bool(remaining > required)


def _timed_assignment_step_evaluation(
    context: Any,
    action: np.ndarray,
    *,
    feedback_enabled: bool,
) -> tuple[AssignmentResult, np.ndarray, np.ndarray, float, float]:
    started_at = time.perf_counter()
    result, simulated_link_flow_row, target_flow_row, normalized_mse = evaluate_assignment_step_candidate(
        context,
        action,
        feedback_enabled=feedback_enabled,
    )
    elapsed_seconds = float(time.perf_counter() - started_at)
    return result, simulated_link_flow_row, target_flow_row, float(normalized_mse), elapsed_seconds


def solve_assignment_gradient_step(
    context: Any,
    params: dict[str, Any],
    *,
    feedback_enabled: bool,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve one online OD row using dynamic-assignment gradient information."""

    params = dict(params)
    current_action = _initial_action(context, params)
    current_result, _, _, current_normalized_mse, last_evaluation_seconds = _timed_assignment_step_evaluation(
        context,
        current_action,
        feedback_enabled=feedback_enabled,
    )
    current_objective = -float(current_normalized_mse)
    step_size = float(params.get("initial_step_size", 0.8))
    min_step_size = float(params.get("min_step_size", 0.05))
    max_step_size = float(params.get("max_step_size", 2.0))
    step_growth = float(params.get("step_growth", 1.03))
    step_shrink = float(params.get("step_shrink", 0.5))
    max_line_search_steps = int(params.get("max_line_search_steps", 4))
    max_iterations = _resolve_iteration_limit(params)
    if max_iterations is None and getattr(context, "step_deadline_time", None) is None:
        max_iterations = 4
    gradient_threshold = float(params.get("gradient_threshold", 1e-12))

    gradient_norm = 0.0
    accepted_updates = 0
    total_evaluations = 1
    completed_iterations = 0
    timeout = False
    best_action_global = current_action.copy()
    best_objective_global = float(current_objective)

    iteration = 0
    while _can_run_iteration(iteration, max_iterations):
        if bool(context.runtime_exceeded()):
            timeout = True
            break
        iteration += 1

        gradient = _step_gradient_from_result(context, current_action, current_result)
        gradient_norm = float(np.linalg.norm(gradient))

        if gradient_norm <= gradient_threshold and current_normalized_mse > 1e-12:
            gradient = -np.ones_like(current_action, dtype=np.float64)
            gradient_norm = float(np.linalg.norm(gradient))

        gradient_scale = max(float(np.mean(np.abs(gradient))), 1e-8)
        normalized_gradient = gradient / gradient_scale
        accepted = False
        best_action = current_action.copy()
        best_result = current_result
        best_objective = current_objective
        best_normalized_mse = current_normalized_mse
        trial_step_size = float(step_size)

        for _ in range(max_line_search_steps + 1):
            if bool(context.runtime_exceeded()):
                timeout = True
                break
            candidate_action = np.clip(
                current_action - trial_step_size * normalized_gradient,
                float(context.action_low),
                float(context.action_high),
            )
            if np.allclose(candidate_action, current_action):
                trial_step_size = max(trial_step_size * step_shrink, min_step_size)
                continue

            if not _has_runtime_for_candidate_evaluation(
                context,
                last_evaluation_seconds=last_evaluation_seconds,
            ):
                timeout = True
                break
            (
                candidate_result,
                _,
                _,
                candidate_normalized_mse,
                last_evaluation_seconds,
            ) = _timed_assignment_step_evaluation(
                context,
                candidate_action,
                feedback_enabled=feedback_enabled,
            )
            candidate_objective = -float(candidate_normalized_mse)
            total_evaluations += 1

            if candidate_objective > best_objective:
                best_action = candidate_action.copy()
                best_result = candidate_result
                best_objective = candidate_objective
                best_normalized_mse = float(candidate_normalized_mse)

            if candidate_objective > best_objective_global:
                best_action_global = candidate_action.copy()
                best_objective_global = float(candidate_objective)

            if candidate_objective > current_objective:
                current_action = candidate_action.copy()
                current_result = candidate_result
                current_objective = candidate_objective
                current_normalized_mse = float(candidate_normalized_mse)
                accepted = True
                accepted_updates += 1
                break

            if bool(context.runtime_exceeded()):
                timeout = True
                break

            trial_step_size = max(trial_step_size * step_shrink, min_step_size)

        if timeout:
            completed_iterations = iteration
            break

        if accepted:
            step_size = min(max(trial_step_size * step_growth, min_step_size), max_step_size)
        else:
            step_size = max(step_size * step_shrink, min_step_size)
            current_action = best_action.copy()
            current_result = best_result
            current_objective = best_objective
            current_normalized_mse = best_normalized_mse

        completed_iterations = iteration

    return np.clip(best_action_global, float(context.action_low), float(context.action_high)), {
        "inner_iterations": int(completed_iterations),
        "inner_evaluations": int(total_evaluations),
        "accepted_updates": int(accepted_updates),
        "final_step_size": float(step_size),
        "gradient_norm": float(gradient_norm),
        "timed_out": bool(timeout or context.runtime_exceeded()),
        "feedback_enabled": bool(feedback_enabled),
        "warm_start": str(params.get("warm_start", "zero")),
    }
