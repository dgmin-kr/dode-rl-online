"""Shared online Kalman-filter baselines for detector-only OD calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .assignment_guidance import compute_assignment_gradient
from .baseline_support import StepLockContext, evaluate_candidate_result


@dataclass(frozen=True)
class KFStepEvaluation:
    action_row: np.ndarray
    simulated_link_flow_row: np.ndarray
    simulated_observation_row: np.ndarray
    step_mse: float
    step_normalized_mse: float
    result: Any


def solve_kf_step(
    context: StepLockContext,
    *,
    params: dict[str, Any],
    state: dict[str, Any],
    lfpg_enabled: bool,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Solve one action row with an ensemble Kalman update in action space.

    The update observes only detector rows:
    ``simulated_link_flows[observed_link_indices] - target_observations``.
    No full-link target tensor and no OD prior are used.
    """

    params = dict(params)
    num_od = int(context.estimated_od_matrix.shape[1])
    action_low = float(context.action_low)
    action_high = float(context.action_high)
    observed_indices = np.asarray(context.target_dataset.observed_link_indices, dtype=np.int64).reshape(-1)
    target_observation_row = np.asarray(
        context.target_dataset.target_observations[int(context.step_index)],
        dtype=np.float64,
    ).reshape(-1)
    flow_scale_observed = np.asarray(context.flow_scale, dtype=np.float64)[observed_indices]

    center_action = _build_transition_center(
        num_od=num_od,
        action_low=action_low,
        action_high=action_high,
        params=params,
        state=state,
    )
    center_eval = _evaluate_candidate(context, center_action)
    best_action = center_action.copy()
    best_eval = center_eval
    total_evaluations = 1
    completed_iterations = 0

    ensemble_size = max(int(params.get("ensemble_size", 12)), 2)
    max_cycles = _resolve_cycle_limit(params)
    if max_cycles is None and getattr(context, "step_deadline_time", None) is None:
        max_cycles = 4
    process_noise_scale = float(params.get("process_noise_scale", 0.3))
    process_noise_floor = float(params.get("process_noise_floor", 0.25))
    kalman_update_norm = 0.0
    posterior_spread = 0.0
    lfpg_gradient_norm = 0.0
    lfpg_accepted = False
    evaluated_ensemble_size = 0
    total_ensemble_evaluations = 0
    last_posterior_mean: np.ndarray | None = None
    last_posterior_actions: np.ndarray | None = None
    prior_ensemble = _build_transition_prior_ensemble(
        num_od=num_od,
        action_low=action_low,
        action_high=action_high,
        params=params,
        state=state,
    )

    while _can_run_cycle(completed_iterations, max_cycles):
        if context.runtime_exceeded():
            break
        completed_iterations += 1
        ensemble_actions = _sample_action_ensemble(
            center_action=center_action,
            action_low=action_low,
            action_high=action_high,
            ensemble_size=ensemble_size,
            process_noise_scale=process_noise_scale,
            process_noise_floor=process_noise_floor,
            rng=context.rng,
            prior_actions=prior_ensemble,
        )
        prior_ensemble = None

        prior_actions: list[np.ndarray] = []
        simulated_observations: list[np.ndarray] = []
        evaluated_ensemble_size = 0

        for ensemble_action in ensemble_actions:
            if context.runtime_exceeded():
                break
            evaluation = _evaluate_candidate(context, ensemble_action)
            total_evaluations += 1
            prior_actions.append(evaluation.action_row)
            simulated_observations.append(evaluation.simulated_observation_row)
            evaluated_ensemble_size += 1
            total_ensemble_evaluations += 1
            if evaluation.step_normalized_mse < best_eval.step_normalized_mse:
                best_eval = evaluation
                best_action = evaluation.action_row.copy()

        if evaluated_ensemble_size < 2 or context.runtime_exceeded():
            break

        prior_action_matrix = np.asarray(prior_actions, dtype=np.float64)
        simulated_observation_matrix = np.asarray(simulated_observations, dtype=np.float64)
        posterior_mean, posterior_actions, kalman_update_norm, posterior_spread = _ensemble_kalman_update(
            prior_actions=prior_action_matrix,
            simulated_observations=simulated_observation_matrix,
            target_observation_row=target_observation_row,
            flow_scale_observed=flow_scale_observed,
            ridge=float(params.get("ridge", 1e-3)),
            inflation=float(params.get("inflation", 1.0)),
            observation_noise_scale=float(params.get("observation_noise_scale", 0.12)),
            observation_noise_floor=float(params.get("observation_noise_floor", 0.05)),
            action_low=action_low,
            action_high=action_high,
        )
        last_posterior_mean = posterior_mean.copy()
        last_posterior_actions = posterior_actions.copy()

        candidate_actions = _select_posterior_candidates(
            posterior_mean=posterior_mean,
            posterior_actions=posterior_actions,
            candidate_count=int(params.get("candidate_count", 2)),
        )
        posterior_best_eval: KFStepEvaluation | None = None
        for candidate_action in candidate_actions:
            if context.runtime_exceeded():
                break
            evaluation = _evaluate_candidate(context, candidate_action)
            total_evaluations += 1
            if posterior_best_eval is None or evaluation.step_normalized_mse < posterior_best_eval.step_normalized_mse:
                posterior_best_eval = evaluation
            if evaluation.step_normalized_mse < best_eval.step_normalized_mse:
                best_eval = evaluation
                best_action = evaluation.action_row.copy()

        if bool(lfpg_enabled) and posterior_best_eval is not None and not context.runtime_exceeded():
            guided_eval, guided_evaluations, lfpg_gradient_norm = _evaluate_lfpg_guided_candidates(
                context=context,
                base_eval=posterior_best_eval,
                params=params,
                action_low=action_low,
                action_high=action_high,
            )
            total_evaluations += guided_evaluations
            if guided_eval is not None and guided_eval.step_normalized_mse < best_eval.step_normalized_mse:
                best_eval = guided_eval
                best_action = guided_eval.action_row.copy()
                lfpg_accepted = True

        center_action = best_action.copy()
        prior_ensemble = _recenter_ensemble(
            ensemble_actions=posterior_actions,
            old_center=posterior_mean,
            new_center=center_action,
            action_low=action_low,
            action_high=action_high,
        )

    state["previous_action"] = best_action.copy()
    if last_posterior_actions is not None and last_posterior_mean is not None:
        state["previous_posterior_actions"] = _recenter_ensemble(
            ensemble_actions=last_posterior_actions,
            old_center=last_posterior_mean,
            new_center=best_action,
            action_low=action_low,
            action_high=action_high,
        )

    return best_action, {
        "inner_iterations": int(completed_iterations),
        "inner_evaluations": int(total_evaluations),
        "ensemble_size": int(evaluated_ensemble_size),
        "total_ensemble_evaluations": int(total_ensemble_evaluations),
        "kalman_update_norm": float(kalman_update_norm),
        "posterior_spread": float(posterior_spread),
        "lfpg_enabled": bool(lfpg_enabled),
        "lfpg_accepted": bool(lfpg_accepted),
        "lfpg_gradient_norm": float(lfpg_gradient_norm),
        "final_step_mse": float(best_eval.step_mse),
        "final_step_normalized_mse": float(best_eval.step_normalized_mse),
        "timed_out": bool(context.runtime_exceeded()),
        "warm_start": str(params.get("warm_start", "zero")),
    }


def _build_transition_center(
    *,
    num_od: int,
    action_low: float,
    action_high: float,
    params: dict[str, Any],
    state: dict[str, Any],
) -> np.ndarray:
    strategy = str(params.get("warm_start", "zero")).strip().lower()
    previous_action = state.get("previous_action")
    if strategy == "previous" and previous_action is not None:
        transition_alpha = float(params.get("transition_alpha", 1.0))
        center = transition_alpha * np.asarray(previous_action, dtype=np.float64).reshape(num_od)
    else:
        center = np.zeros(num_od, dtype=np.float64)
    return np.clip(center, action_low, action_high)


def _build_transition_prior_ensemble(
    *,
    num_od: int,
    action_low: float,
    action_high: float,
    params: dict[str, Any],
    state: dict[str, Any],
) -> np.ndarray | None:
    strategy = str(params.get("warm_start", "zero")).strip().lower()
    previous_posterior_actions = state.get("previous_posterior_actions")
    if strategy != "previous" or previous_posterior_actions is None:
        return None
    transition_alpha = float(params.get("transition_alpha", 1.0))
    prior = transition_alpha * np.asarray(previous_posterior_actions, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != int(num_od):
        return None
    return np.clip(prior, float(action_low), float(action_high))


def _sample_action_ensemble(
    *,
    center_action: np.ndarray,
    action_low: float,
    action_high: float,
    ensemble_size: int,
    process_noise_scale: float,
    process_noise_floor: float,
    rng: np.random.Generator,
    prior_actions: np.ndarray | None = None,
) -> np.ndarray:
    center_action = np.asarray(center_action, dtype=np.float64).reshape(-1)
    action_span = max(float(action_high) - float(action_low), 1e-6)
    noise_scale = max(float(process_noise_scale) * action_span, float(process_noise_floor))
    ensemble = _resize_prior_ensemble(
        prior_actions=prior_actions,
        center_action=center_action,
        ensemble_size=int(ensemble_size),
    )
    if ensemble is None:
        ensemble = np.repeat(center_action[None, :], int(ensemble_size), axis=0)
    for ensemble_index in range(1, int(ensemble_size)):
        ensemble[ensemble_index] = ensemble[ensemble_index] + rng.normal(0.0, noise_scale, size=center_action.shape)
    return np.clip(ensemble, float(action_low), float(action_high))


def _resize_prior_ensemble(
    *,
    prior_actions: np.ndarray | None,
    center_action: np.ndarray,
    ensemble_size: int,
) -> np.ndarray | None:
    if prior_actions is None:
        return None
    prior = np.asarray(prior_actions, dtype=np.float64)
    if prior.ndim != 2 or prior.shape[1] != center_action.shape[0] or prior.shape[0] <= 0:
        return None
    if prior.shape[0] >= int(ensemble_size):
        selected = prior[: int(ensemble_size)].copy()
    else:
        repeats = int(np.ceil(float(ensemble_size) / float(prior.shape[0])))
        selected = np.tile(prior, (repeats, 1))[: int(ensemble_size)].copy()
    prior_mean = np.mean(selected, axis=0)
    selected = center_action[None, :] + (selected - prior_mean[None, :])
    selected[0] = center_action
    return selected


def _recenter_ensemble(
    *,
    ensemble_actions: np.ndarray,
    old_center: np.ndarray,
    new_center: np.ndarray,
    action_low: float,
    action_high: float,
) -> np.ndarray:
    ensemble = np.asarray(ensemble_actions, dtype=np.float64)
    old_center = np.asarray(old_center, dtype=np.float64).reshape(-1)
    new_center = np.asarray(new_center, dtype=np.float64).reshape(-1)
    recentered = new_center[None, :] + (ensemble - old_center[None, :])
    if recentered.shape[0] > 0:
        recentered[0] = new_center
    return np.clip(recentered, float(action_low), float(action_high))


def _resolve_cycle_limit(params: dict[str, Any]) -> int | None:
    raw_value = params.get("max_cycles")
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        return None
    return value


def _can_run_cycle(cycle_index: int, max_cycles: int | None) -> bool:
    return max_cycles is None or int(cycle_index) < int(max_cycles)


def _evaluate_candidate(context: StepLockContext, action_row: np.ndarray) -> KFStepEvaluation:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(context.estimated_od_matrix.shape[1])
    result = evaluate_candidate_result(context, action_row)
    step_index = int(context.step_index)
    simulated_link_flow_row = np.asarray(result.link_inflows[step_index], dtype=np.float64)
    observed_indices = np.asarray(context.target_dataset.observed_link_indices, dtype=np.int64).reshape(-1)
    target_observation_row = np.asarray(context.target_dataset.target_observations[step_index], dtype=np.float64)
    simulated_observation_row = simulated_link_flow_row[observed_indices]
    flow_error = simulated_observation_row - target_observation_row
    normalized_flow_error = flow_error / np.maximum(
        np.asarray(context.flow_scale, dtype=np.float64)[observed_indices],
        1.0,
    )
    return KFStepEvaluation(
        action_row=action_row.copy(),
        simulated_link_flow_row=simulated_link_flow_row.copy(),
        simulated_observation_row=simulated_observation_row.copy(),
        step_mse=float(np.mean(flow_error ** 2)),
        step_normalized_mse=float(np.mean(normalized_flow_error ** 2)),
        result=result,
    )


def _ensemble_kalman_update(
    *,
    prior_actions: np.ndarray,
    simulated_observations: np.ndarray,
    target_observation_row: np.ndarray,
    flow_scale_observed: np.ndarray,
    ridge: float,
    inflation: float,
    observation_noise_scale: float,
    observation_noise_floor: float,
    action_low: float,
    action_high: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    prior_actions = np.asarray(prior_actions, dtype=np.float64)
    simulated_observations = np.asarray(simulated_observations, dtype=np.float64)
    target_observation_row = np.asarray(target_observation_row, dtype=np.float64).reshape(-1)
    flow_scale_observed = np.asarray(flow_scale_observed, dtype=np.float64).reshape(-1)

    ensemble_size = int(prior_actions.shape[0])
    action_mean = np.mean(prior_actions, axis=0)
    observation_mean = np.mean(simulated_observations, axis=0)
    action_anomalies = prior_actions - action_mean[None, :]
    observation_anomalies = simulated_observations - observation_mean[None, :]
    covariance_denominator = float(max(ensemble_size - 1, 1))

    observation_noise = np.maximum(
        float(observation_noise_floor) * np.maximum(flow_scale_observed, 1.0),
        float(observation_noise_scale)
        * np.maximum(np.abs(target_observation_row), 0.5 * np.maximum(flow_scale_observed, 1.0)),
    )
    p_xy = action_anomalies.T @ observation_anomalies / covariance_denominator
    p_yy = observation_anomalies.T @ observation_anomalies / covariance_denominator
    p_yy = p_yy + np.diag(np.maximum(observation_noise ** 2, 1e-8))
    p_yy = p_yy + float(ridge) * np.eye(p_yy.shape[0], dtype=np.float64)

    innovation = target_observation_row - observation_mean
    kalman_delta = p_xy @ np.linalg.solve(p_yy, innovation)
    posterior_mean = np.clip(action_mean + kalman_delta, float(action_low), float(action_high))

    member_deltas = np.asarray(
        [
            p_xy @ np.linalg.solve(p_yy, target_observation_row - simulated_observations[row_index])
            for row_index in range(ensemble_size)
        ],
        dtype=np.float64,
    )
    posterior_actions = np.clip(
        posterior_mean[None, :] + float(inflation) * (action_anomalies + member_deltas - kalman_delta[None, :]),
        float(action_low),
        float(action_high),
    )
    posterior_spread = float(np.mean(np.linalg.norm(posterior_actions - posterior_mean[None, :], axis=1)))
    return posterior_mean, posterior_actions, float(np.linalg.norm(kalman_delta)), posterior_spread


def _select_posterior_candidates(
    *,
    posterior_mean: np.ndarray,
    posterior_actions: np.ndarray,
    candidate_count: int,
) -> list[np.ndarray]:
    posterior_mean = np.asarray(posterior_mean, dtype=np.float64).reshape(-1)
    posterior_actions = np.asarray(posterior_actions, dtype=np.float64)
    candidates: list[np.ndarray] = [posterior_mean.copy()]
    if int(candidate_count) <= 1 or posterior_actions.size == 0:
        return candidates
    distances = np.linalg.norm(posterior_actions - posterior_mean[None, :], axis=1)
    for row_index in np.argsort(-distances)[: max(int(candidate_count) - 1, 0)]:
        candidate = np.asarray(posterior_actions[int(row_index)], dtype=np.float64).reshape(-1)
        if not any(np.allclose(candidate, existing, atol=1e-8, rtol=1e-6) for existing in candidates):
            candidates.append(candidate.copy())
    return candidates


def _evaluate_lfpg_guided_candidates(
    *,
    context: StepLockContext,
    base_eval: KFStepEvaluation,
    params: dict[str, Any],
    action_low: float,
    action_high: float,
) -> tuple[KFStepEvaluation | None, int, float]:
    gradient = _assignment_gradient_from_evaluation(context, base_eval)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= float(params.get("lfpg_gradient_threshold", 1e-12)):
        return None, 0, gradient_norm

    gradient_scale = max(float(np.mean(np.abs(gradient))), 1e-8)
    normalized_gradient = gradient / gradient_scale
    step_size = float(params.get("lfpg_step_size", 0.8))
    shrink = float(params.get("lfpg_step_shrink", 0.5))
    candidate_count = max(int(params.get("lfpg_candidate_count", 2)), 1)
    best_eval: KFStepEvaluation | None = None
    evaluations = 0

    for _ in range(candidate_count):
        if context.runtime_exceeded():
            break
        candidate_action = np.clip(
            base_eval.action_row - step_size * normalized_gradient,
            float(action_low),
            float(action_high),
        )
        if np.allclose(candidate_action, base_eval.action_row, atol=1e-8, rtol=1e-6):
            step_size *= shrink
            continue
        evaluation = _evaluate_candidate(context, candidate_action)
        evaluations += 1
        if best_eval is None or evaluation.step_normalized_mse < best_eval.step_normalized_mse:
            best_eval = evaluation
        if evaluation.step_normalized_mse < base_eval.step_normalized_mse:
            break
        step_size *= shrink

    return best_eval, evaluations, gradient_norm


def _assignment_gradient_from_evaluation(
    context: StepLockContext,
    evaluation: KFStepEvaluation,
) -> np.ndarray:
    step_index = int(context.step_index)
    result = evaluation.result
    temporal_source = np.asarray(result.temporal_link_inflows, dtype=np.float64)
    if temporal_source.shape[0] == 1 and temporal_source.shape[1] == 1:
        temporal_link_inflows = temporal_source[:1, :1]
    else:
        temporal_link_inflows = temporal_source[step_index : step_index + 1, step_index : step_index + 1]
    simulated_link_flows = np.asarray(
        result.link_inflows[step_index : step_index + 1],
        dtype=np.float64,
    )
    gradient = compute_assignment_gradient(
        od_matrix=np.asarray(evaluation.action_row, dtype=np.float64).reshape(1, -1),
        temporal_link_inflows=temporal_link_inflows,
        simulated_link_flows=simulated_link_flows,
        target_observations=np.asarray(
            context.target_dataset.target_observations[step_index : step_index + 1],
            dtype=np.float64,
        ),
        observed_link_indices=np.asarray(context.target_dataset.observed_link_indices, dtype=np.int64),
        flow_scale=np.asarray(context.flow_scale, dtype=np.float64),
    )
    return np.asarray(gradient, dtype=np.float64).reshape(-1)
