from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.baseline_support import StepLockContext, evaluate_candidate_result


@dataclass(frozen=True)
class WSPSAStepEvaluation:
    action_row: np.ndarray
    simulated_link_flow_row: np.ndarray
    measurement_error_vector: np.ndarray
    step_mse: float
    step_normalized_mse: float
    result: Any


def solve_w_spsa_step(
    context: StepLockContext,
    *,
    params: dict[str, Any],
    state: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    num_od = int(context.estimated_od_matrix.shape[1])
    action_low = float(context.action_low)
    action_high = float(context.action_high)
    action_span = max(action_high - action_low, 1e-6)

    max_iterations = _resolve_iteration_limit(params)
    if max_iterations is None and getattr(context, "step_deadline_time", None) is None:
        max_iterations = 4
    correlation_threshold = float(max(params.get("correlation_threshold", 0.0), 0.0))
    binary_weights = bool(params.get("binary_weights", False))

    current_action = _build_initial_action(
        num_od=num_od,
        action_low=action_low,
        action_high=action_high,
        params=params,
        state=state,
    )
    current_eval = _evaluate_candidate(context, current_action)
    best_action = current_action.copy()
    best_eval = current_eval
    total_evaluations = 1
    gradient_norm = 0.0
    final_ak = 0.0
    final_ck = 0.0
    mean_active_links = 0.0
    completed_iterations = 0

    iteration = 0
    while _can_run_iteration(iteration, max_iterations):
        if context.runtime_exceeded():
            break
        iteration += 1
        ak = float(params["a"]) / ((float(params["A"]) + iteration) ** float(params["alpha"]))
        ck = float(params["c"]) / (iteration ** float(params["gamma"]))
        final_ak = float(ak)
        final_ck = float(ck)

        delta = context.rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=num_od)
        perturbation = ck * action_span * delta
        action_plus = np.clip(current_action + perturbation, action_low, action_high)
        action_minus = np.clip(current_action - perturbation, action_low, action_high)

        plus_eval = _evaluate_candidate(context, action_plus)
        minus_eval = _evaluate_candidate(context, action_minus)
        total_evaluations += 2
        for evaluated_action, evaluated_result in (
            (action_plus, plus_eval),
            (action_minus, minus_eval),
        ):
            if evaluated_result.step_normalized_mse < best_eval.step_normalized_mse:
                best_action = evaluated_action.copy()
                best_eval = evaluated_result
        if context.runtime_exceeded():
            completed_iterations = iteration
            break

        weight_matrix = _build_weight_matrix(
            step_index=int(context.step_index),
            action_plus=plus_eval.action_row,
            plus_result=plus_eval.result,
            action_minus=minus_eval.action_row,
            minus_result=minus_eval.result,
            correlation_threshold=correlation_threshold,
            binary_weights=binary_weights,
            observed_link_indices=context.target_dataset.observed_link_indices,
        )
        mean_active_links = float(np.mean(np.count_nonzero(weight_matrix > 0.0, axis=1)))

        weighted_error_plus = np.sum(weight_matrix * plus_eval.measurement_error_vector[None, :], axis=1)
        weighted_error_minus = np.sum(weight_matrix * minus_eval.measurement_error_vector[None, :], axis=1)
        denominator = plus_eval.action_row - minus_eval.action_row
        stable_denominator = np.where(np.abs(denominator) > 1e-8, denominator, np.sign(delta) * 1e-8)
        gradient = (weighted_error_plus - weighted_error_minus) / stable_denominator
        gradient_norm = float(np.linalg.norm(gradient))

        candidate_action = np.clip(
            current_action - ak * gradient,
            action_low,
            action_high,
        )
        candidate_eval = _evaluate_candidate(context, candidate_action)
        total_evaluations += 1
        completed_iterations = iteration
        if context.runtime_exceeded():
            if candidate_eval.step_normalized_mse < best_eval.step_normalized_mse:
                best_action = candidate_action.copy()
                best_eval = candidate_eval
            break

        current_action = candidate_action.copy()
        current_eval = candidate_eval
        if current_eval.step_normalized_mse < best_eval.step_normalized_mse:
            best_action = current_action.copy()
            best_eval = current_eval

    state["previous_action"] = best_action.copy()

    return best_action, {
        "inner_iterations": int(completed_iterations),
        "inner_evaluations": int(total_evaluations),
        "gradient_norm": float(gradient_norm),
        "final_ak": float(final_ak),
        "final_ck": float(final_ck),
        "mean_active_links": float(mean_active_links),
        "final_step_mse": float(best_eval.step_mse),
        "final_step_normalized_mse": float(best_eval.step_normalized_mse),
        "timed_out": bool(context.runtime_exceeded()),
    }


def _evaluate_candidate(context: StepLockContext, action_row: np.ndarray) -> WSPSAStepEvaluation:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(context.estimated_od_matrix.shape[1])
    result = evaluate_candidate_result(context, action_row)
    step_index = int(context.step_index)
    simulated_link_flow_row = np.asarray(result.link_inflows[step_index], dtype=np.float64)
    observed_indices = np.asarray(context.target_dataset.observed_link_indices, dtype=np.int64).reshape(-1)
    target_observation_row = np.asarray(context.target_dataset.target_observations[step_index], dtype=np.float64)
    flow_error = simulated_link_flow_row[observed_indices] - target_observation_row
    normalized_flow_error = flow_error / np.asarray(context.flow_scale, dtype=np.float64)[observed_indices]
    full_measurement_error_vector = np.zeros_like(simulated_link_flow_row, dtype=np.float64)
    full_measurement_error_vector[observed_indices] = normalized_flow_error ** 2

    return WSPSAStepEvaluation(
        action_row=action_row.copy(),
        simulated_link_flow_row=simulated_link_flow_row.copy(),
        measurement_error_vector=full_measurement_error_vector.copy(),
        step_mse=float(np.mean(flow_error ** 2)),
        step_normalized_mse=float(np.mean(normalized_flow_error ** 2)),
        result=result,
    )


def _build_initial_action(
    *,
    num_od: int,
    action_low: float,
    action_high: float,
    params: dict[str, Any],
    state: dict[str, Any],
) -> np.ndarray:
    previous_action = state.get("previous_action")
    strategy = str(params.get("warm_start", "zero")).strip().lower()
    if strategy == "previous" and previous_action is not None:
        initial = np.asarray(previous_action, dtype=np.float64).reshape(num_od)
    else:
        initial = np.zeros(num_od, dtype=np.float64)
    return np.clip(initial, action_low, action_high)


def _resolve_iteration_limit(params: dict[str, Any]) -> int | None:
    raw_value = params.get("max_iterations")
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        return None
    return value


def _can_run_iteration(iteration_index: int, max_iterations: int | None) -> bool:
    return max_iterations is None or int(iteration_index) < int(max_iterations)


def _build_weight_matrix(
    *,
    step_index: int,
    action_plus: np.ndarray,
    plus_result: Any,
    action_minus: np.ndarray,
    minus_result: Any,
    correlation_threshold: float,
    binary_weights: bool,
    observed_link_indices: np.ndarray,
) -> np.ndarray:
    plus_temporal = np.asarray(plus_result.temporal_link_inflows, dtype=np.float64)
    minus_temporal = np.asarray(minus_result.temporal_link_inflows, dtype=np.float64)
    if plus_temporal.shape[0] == 1 and plus_temporal.shape[1] == 1:
        plus_temporal_window = plus_temporal[:1, 0]
    else:
        plus_temporal_window = plus_temporal[step_index : step_index + 1, step_index]
    if minus_temporal.shape[0] == 1 and minus_temporal.shape[1] == 1:
        minus_temporal_window = minus_temporal[:1, 0]
    else:
        minus_temporal_window = minus_temporal[step_index : step_index + 1, step_index]
    plus_weights = _extract_route_choice_weights(
        action_row=action_plus,
        temporal_link_inflows=plus_temporal_window,
    )
    minus_weights = _extract_route_choice_weights(
        action_row=action_minus,
        temporal_link_inflows=minus_temporal_window,
    )
    weights = 0.5 * (plus_weights + minus_weights)
    weights = _apply_observed_indices(weights, observed_link_indices)
    weights = _normalize_weight_rows(weights)

    if correlation_threshold > 0.0:
        weights = np.where(weights >= correlation_threshold, weights, 0.0)
        weights = _apply_observed_indices(weights, observed_link_indices)
        weights = _normalize_weight_rows(weights)

    if binary_weights:
        weights = np.where(weights > 0.0, 1.0, 0.0)
        weights = _apply_observed_indices(weights, observed_link_indices)
        weights = _normalize_weight_rows(weights)

    zero_rows = np.sum(weights, axis=1) <= 1e-12
    if np.any(zero_rows):
        observed_mask = _indices_to_mask(weights.shape[1], observed_link_indices)
        weights[zero_rows] = 0.0
        weights[np.ix_(np.flatnonzero(zero_rows), np.flatnonzero(observed_mask))] = (
            1.0 / max(int(np.sum(observed_mask)), 1)
        )
    return weights


def _extract_route_choice_weights(
    *,
    action_row: np.ndarray,
    temporal_link_inflows: np.ndarray,
) -> np.ndarray:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(-1)
    temporal_link_inflows = np.asarray(temporal_link_inflows, dtype=np.float64)
    if temporal_link_inflows.ndim == 3:
        temporal_link_inflows = np.sum(temporal_link_inflows, axis=0)
    if temporal_link_inflows.ndim != 2:
        raise ValueError(
            "temporal_link_inflows must have shape [num_od, num_links] or "
            f"[time, num_od, num_links], got {temporal_link_inflows.shape}."
        )
    denominator = np.maximum(action_row[:, None], 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.divide(
            temporal_link_inflows,
            denominator,
            out=np.zeros_like(temporal_link_inflows, dtype=np.float64),
            where=denominator > 1e-8,
        )
    return np.clip(weights, 0.0, None)


def _normalize_weight_rows(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    row_sums = np.sum(weights, axis=1, keepdims=True)
    normalized = np.zeros_like(weights, dtype=np.float64)
    np.divide(weights, row_sums, out=normalized, where=row_sums > 1e-12)
    return normalized


def _indices_to_mask(num_links: int, observed_link_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(observed_link_indices, dtype=np.int64).reshape(-1)
    if indices.size <= 0:
        raise ValueError("observed_link_indices must contain at least one detector link.")
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= int(num_links):
        raise ValueError(
            "observed_link_indices out of range: "
            f"valid range is [0, {int(num_links) - 1}], got {indices.tolist()}."
        )
    mask = np.zeros(int(num_links), dtype=bool)
    mask[indices] = True
    return mask


def _apply_observed_indices(weights: np.ndarray, observed_link_indices: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).copy()
    observed_mask = _indices_to_mask(weights.shape[1], observed_link_indices)
    weights[:, ~observed_mask] = 0.0
    return weights
