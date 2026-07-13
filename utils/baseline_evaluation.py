"""DNL model builders and detector-observation metrics shared by baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from dnl.ltm import ForwardDUOSimulator
from dnl.main import build_default_model
from dnl.model import AssignmentResult, DynamicNetworkLoadingModel
from dnl.network.registry import get_network_display_name


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _normalize_output_suffix(output_suffix: str | None) -> str:
    if output_suffix is None:
        return ""
    suffix = str(output_suffix).strip()
    if not suffix:
        return ""
    if any(separator in suffix for separator in ("/", "\\")) or suffix in {".", ".."}:
        raise ValueError(f"Invalid output suffix: {output_suffix!r}")
    return suffix if suffix.startswith("_") else f"_{suffix}"


def build_result_name(
    network_name: str,
    trial_index: int | None = None,
    *,
    output_suffix: str | None = None,
) -> str:
    display_name = get_network_display_name(network_name)
    suffix = _normalize_output_suffix(output_suffix)
    if suffix:
        return f"{display_name}{suffix}"
    if trial_index is None:
        raise ValueError("trial_index is required when output_suffix is not provided.")
    return f"{display_name}_trial_{trial_index:02d}"


def build_trial_name(network_name: str, trial_index: int) -> str:
    return build_result_name(network_name, trial_index)


@dataclass(frozen=True)
class TargetObservationDataset:
    link_labels: tuple[str, ...]
    target_observations: np.ndarray
    observed_link_indices: np.ndarray
    observation_labels: tuple[str, ...] = ()

    @property
    def num_steps(self) -> int:
        return int(self.target_observations.shape[0])

    @property
    def num_links(self) -> int:
        return int(len(self.link_labels))

    @property
    def observed_link_count(self) -> int:
        return int(self.observed_link_indices.shape[0])


@dataclass(frozen=True)
class MatrixEvaluation:
    od_matrix: np.ndarray
    result: AssignmentResult
    objective: float
    mse_mean: float
    mae_mean: float
    normalized_mse: float
    corr_mean: float
    target_observations: np.ndarray
    simulated_observations: np.ndarray
    observed_link_indices: np.ndarray
    observation_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepObjectiveEvaluation:
    step_index: int
    action_row: np.ndarray
    objective: float
    step_mse: float
    step_mae: float
    step_normalized_mse: float
    simulated_link_flow_row: np.ndarray


def build_model_from_config(config: Any) -> DynamicNetworkLoadingModel:
    return build_default_model(
        network_name=config.NETWORK_NAME,
        route_choice_mode=config.ROUTE_CHOICE_MODE,
        stochastic_logit_scale=config.STOCHASTIC_LOGIT_SCALE,
        sample_route_choices=getattr(config, "DNL_SAMPLE_ROUTE_CHOICES", None),
        route_choice_sampling_unit=getattr(config, "DNL_ROUTE_CHOICE_SAMPLING_UNIT", None),
        random_seed=getattr(config, "DNL_RANDOM_SEED", None),
        use_parallel_kernels=config.DNL_PARALLEL_KERNELS,
        numba_threads=config.DNL_NUMBA_THREADS,
    )


def build_model_from_config_with_seed(config: Any, random_seed: int | None) -> DynamicNetworkLoadingModel:
    return build_default_model(
        network_name=config.NETWORK_NAME,
        route_choice_mode=config.ROUTE_CHOICE_MODE,
        stochastic_logit_scale=config.STOCHASTIC_LOGIT_SCALE,
        sample_route_choices=getattr(config, "DNL_SAMPLE_ROUTE_CHOICES", None),
        route_choice_sampling_unit=getattr(config, "DNL_ROUTE_CHOICE_SAMPLING_UNIT", None),
        random_seed=random_seed,
        use_parallel_kernels=config.DNL_PARALLEL_KERNELS,
        numba_threads=config.DNL_NUMBA_THREADS,
    )


def compute_flow_scale(model: DynamicNetworkLoadingModel) -> np.ndarray:
    return np.maximum(
        np.asarray([link.capacity for link in model.network.links], dtype=np.float64),
        1.0,
    )


def resolve_observed_link_indices(
    num_links: int,
    observed_link_indices: np.ndarray | list[int] | tuple[int, ...],
) -> np.ndarray:
    indices = np.asarray(observed_link_indices, dtype=np.int64).reshape(-1)
    if indices.size <= 0:
        raise ValueError("observed_link_indices must contain at least one detector link.")
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= int(num_links):
        raise ValueError(
            "observed_link_indices out of range: "
            f"valid range is [0, {int(num_links) - 1}], got {indices.tolist()}."
        )
    if len(set(int(index) for index in indices.tolist())) != int(indices.size):
        raise ValueError("observed_link_indices must not contain duplicates.")
    return indices.astype(np.int64, copy=False)


def observed_indices_to_mask(num_links: int, observed_link_indices: np.ndarray) -> np.ndarray:
    indices = resolve_observed_link_indices(int(num_links), observed_link_indices)
    mask = np.zeros(int(num_links), dtype=bool)
    mask[indices] = True
    return mask


def compute_observation_scale(
    *,
    flow_scale: np.ndarray,
    observed_link_indices: np.ndarray,
) -> np.ndarray:
    scale = np.asarray(flow_scale, dtype=np.float64).reshape(-1)
    indices = resolve_observed_link_indices(scale.shape[0], observed_link_indices)
    return np.maximum(scale[indices], 1.0)


def compute_simulated_observations(
    simulated_link_flows: np.ndarray,
    target_dataset: TargetObservationDataset,
) -> np.ndarray:
    simulated = np.asarray(simulated_link_flows, dtype=np.float64)
    if simulated.ndim == 1:
        return simulated[target_dataset.observed_link_indices].astype(np.float64)
    return simulated[:, target_dataset.observed_link_indices].astype(np.float64)


def compute_observed_error_metrics(
    *,
    simulated_link_flows: np.ndarray,
    target_dataset: TargetObservationDataset,
    flow_scale: np.ndarray,
) -> tuple[float, float, float]:
    simulated_observations = compute_simulated_observations(simulated_link_flows, target_dataset)
    target = np.asarray(target_dataset.target_observations, dtype=np.float64)
    if simulated_observations.shape != target.shape:
        raise ValueError(
            "simulated detector observations and target_observations must have the same shape: "
            f"{simulated_observations.shape} vs {target.shape}."
        )
    scale = compute_observation_scale(
        flow_scale=flow_scale,
        observed_link_indices=target_dataset.observed_link_indices,
    )
    error = simulated_observations - target
    normalized_error = error / scale.reshape((1,) * (error.ndim - 1) + (scale.shape[0],))
    return (
        float(np.mean(error ** 2)),
        float(np.mean(np.abs(error))),
        float(np.mean(normalized_error ** 2)),
    )


def compute_observed_corr_mean(
    *,
    simulated_link_flows: np.ndarray,
    target_dataset: TargetObservationDataset,
) -> float:
    simulated_observations = compute_simulated_observations(simulated_link_flows, target_dataset)
    target = np.asarray(target_dataset.target_observations, dtype=np.float64)
    if simulated_observations.shape != target.shape:
        raise ValueError(
            "simulated detector observations and target_observations must have the same shape: "
            f"{simulated_observations.shape} vs {target.shape}."
        )

    correlations: list[float] = []
    for observation_index in range(target.shape[1]):
        target_col = target[:, observation_index]
        simulated_col = simulated_observations[:, observation_index]
        target_std = float(np.std(target_col))
        simulated_std = float(np.std(simulated_col))
        if target_std <= 1e-12 or simulated_std <= 1e-12:
            same_profile = bool(np.allclose(target_col, simulated_col, atol=1e-8, rtol=1e-6))
            correlations.append(1.0 if same_profile else 0.0)
            continue
        correlations.append(float(np.corrcoef(target_col, simulated_col)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def build_uniform_initial_od_matrix(
    model: DynamicNetworkLoadingModel,
    target_dataset: TargetObservationDataset,
    action_high: float,
) -> np.ndarray:
    num_steps = int(target_dataset.num_steps)
    num_od = len(model.od_pairs)
    if num_steps <= 0 or num_od <= 0:
        raise ValueError("Target dataset or OD set is empty.")
    _ = action_high
    return np.zeros((num_steps, num_od), dtype=np.float64)


def get_optimization_runtime_budget(
    max_runtime_seconds: float | None,
    finalization_buffer_seconds: float = 60.0,
) -> float | None:
    if max_runtime_seconds is None:
        return None
    runtime_budget = float(max_runtime_seconds) - float(finalization_buffer_seconds)
    return max(0.0, runtime_budget)


def evaluate_od_matrix(
    model: DynamicNetworkLoadingModel,
    target_dataset: TargetObservationDataset,
    od_matrix: np.ndarray,
    flow_scale: np.ndarray | None = None,
) -> MatrixEvaluation:
    od_matrix = np.asarray(od_matrix, dtype=np.float64)
    result = model.solve(od_matrix)
    return evaluate_assignment_result(
        od_matrix=od_matrix,
        result=result,
        target_dataset=target_dataset,
        flow_scale=compute_flow_scale(model) if flow_scale is None else flow_scale,
    )


def build_target_dataset_from_arrays(
    *,
    link_labels: tuple[str, ...] | list[str],
    target_observations: np.ndarray,
    observed_link_indices: np.ndarray | list[int] | tuple[int, ...],
    observation_labels: tuple[str, ...] | list[str] | None = None,
) -> TargetObservationDataset:
    labels = tuple(str(label) for label in link_labels)
    observations = np.asarray(target_observations, dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError(
            "target_observations must be shaped [num_steps, num_observations]; "
            f"got {observations.shape}."
        )
    indices = resolve_observed_link_indices(len(labels), observed_link_indices)
    if observations.shape[1] != int(indices.shape[0]):
        raise ValueError(
            "target_observations second dimension must match observed_link_indices length: "
            f"{observations.shape[1]} vs {int(indices.shape[0])}."
        )
    obs_labels = tuple(str(label) for label in (observation_labels or ()))
    if obs_labels and len(obs_labels) != int(indices.shape[0]):
        raise ValueError(
            "observation_labels length must match observed_link_indices length: "
            f"expected {int(indices.shape[0])}, got {len(obs_labels)}."
        )
    if not obs_labels:
        obs_labels = tuple(labels[int(index)] for index in indices)
    return TargetObservationDataset(
        link_labels=labels,
        target_observations=observations,
        observed_link_indices=indices,
        observation_labels=obs_labels,
    )


def evaluate_assignment_result(
    *,
    od_matrix: np.ndarray,
    result: AssignmentResult,
    target_dataset: TargetObservationDataset,
    flow_scale: np.ndarray,
) -> MatrixEvaluation:
    simulated_link_flows = np.asarray(result.link_inflows, dtype=np.float64)
    if tuple(result.link_labels) != tuple(target_dataset.link_labels):
        raise ValueError(
            "DNL link labels do not match the target observation dataset. "
            f"Expected {tuple(result.link_labels)}, got {target_dataset.link_labels}."
        )

    flow_scale = np.asarray(flow_scale, dtype=np.float64)
    mse_mean, mae_mean, normalized_mse = compute_observed_error_metrics(
        simulated_link_flows=simulated_link_flows,
        target_dataset=target_dataset,
        flow_scale=flow_scale,
    )
    corr_mean = compute_observed_corr_mean(
        simulated_link_flows=simulated_link_flows,
        target_dataset=target_dataset,
    )
    simulated_observations = compute_simulated_observations(simulated_link_flows, target_dataset)

    return MatrixEvaluation(
        od_matrix=np.asarray(od_matrix, dtype=np.float64).copy(),
        result=result,
        objective=-normalized_mse,
        mse_mean=mse_mean,
        mae_mean=mae_mean,
        normalized_mse=normalized_mse,
        corr_mean=corr_mean,
        target_observations=np.asarray(target_dataset.target_observations, dtype=np.float32).copy(),
        simulated_observations=np.asarray(simulated_observations, dtype=np.float32),
        observed_link_indices=np.asarray(target_dataset.observed_link_indices, dtype=np.int64).copy(),
        observation_labels=tuple(target_dataset.observation_labels),
    )


def compute_observed_step_error_metrics(
    simulated_link_flow_row: np.ndarray,
    target_dataset: TargetObservationDataset,
    step_index: int,
    flow_scale: np.ndarray,
) -> tuple[float, float, float]:
    simulated = np.asarray(simulated_link_flow_row, dtype=np.float64).reshape(-1)
    indices = target_dataset.observed_link_indices
    target = np.asarray(target_dataset.target_observations[int(step_index)], dtype=np.float64)
    simulated_observations = simulated[indices]
    scale = compute_observation_scale(flow_scale=flow_scale, observed_link_indices=indices)
    error = simulated_observations - target
    normalized_error = error / scale
    return (
        float(np.mean(error ** 2)),
        float(np.mean(np.abs(error))),
        float(np.mean(normalized_error ** 2)),
    )


def evaluate_step_action(
    model: DynamicNetworkLoadingModel,
    target_dataset: TargetObservationDataset,
    fixed_od_matrix: np.ndarray,
    step_index: int,
    action_row: np.ndarray,
    flow_scale: np.ndarray,
    locked_runtime: ForwardDUOSimulator | None,
) -> StepObjectiveEvaluation:
    action_row = np.asarray(action_row, dtype=np.float64).reshape(len(model.od_pairs))

    if model.route_choice_mode == "duo":
        if locked_runtime is None:
            raise RuntimeError("DUO sequential optimization requires a locked runtime state.")
        trial_runtime = copy_runtime_for_single_step_candidate(locked_runtime)
        duo_step = trial_runtime.step(action_row)
        simulated_link_flow_row = duo_step.link_inflow_row
    else:
        candidate_od_matrix = np.asarray(fixed_od_matrix[: int(step_index) + 1], dtype=np.float64).copy()
        candidate_od_matrix[step_index] = action_row
        result = model.solve(candidate_od_matrix)
        simulated_link_flow_row = result.link_inflows[step_index]

    step_mse, step_mae, step_normalized_mse = compute_observed_step_error_metrics(
        simulated_link_flow_row=simulated_link_flow_row,
        target_dataset=target_dataset,
        step_index=step_index,
        flow_scale=flow_scale,
    )
    return StepObjectiveEvaluation(
        step_index=int(step_index),
        action_row=action_row.copy(),
        objective=-step_normalized_mse,
        step_mse=step_mse,
        step_mae=step_mae,
        step_normalized_mse=step_normalized_mse,
        simulated_link_flow_row=np.asarray(simulated_link_flow_row, dtype=np.float64).copy(),
    )


def copy_runtime_for_single_step_candidate(locked_runtime: ForwardDUOSimulator):
    if hasattr(locked_runtime, "copy_for_single_step_candidate"):
        return locked_runtime.copy_for_single_step_candidate()
    raise TypeError(
        "DUO locked runtime does not support single-step candidate cloning: "
        f"{type(locked_runtime).__name__}"
    )


def finalize_duo_or_full_result(
    model: DynamicNetworkLoadingModel,
    estimated_od_matrix: np.ndarray,
    locked_runtime: ForwardDUOSimulator | None,
) -> AssignmentResult:
    if model.route_choice_mode == "duo":
        if locked_runtime is None:
            raise RuntimeError("DUO sequential optimization requires a locked runtime state.")
        return model.finalize_duo_runtime(locked_runtime)
    return model.solve(np.asarray(estimated_od_matrix, dtype=np.float64))
