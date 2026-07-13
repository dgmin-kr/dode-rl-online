from __future__ import annotations

import numpy as np

from .config import get_dnl_settings
from .model import AssignmentResult, DynamicNetworkLoadingModel
from .network.registry import build_network_definition, canonical_network_name


def build_default_model(
    network_name: str = "melbourne_scats",
    max_paths_per_od: int | None = None,
    due_max_iterations: int | None = None,
    due_tolerance: float | None = None,
    clearance_steps: int | None = None,
    stochastic_logit_scale: float | None = None,
    route_choice_mode: str | None = None,
    sample_route_choices: bool | None = None,
    route_choice_sampling_unit: float | None = None,
    random_seed: int | None = None,
    use_parallel_kernels: bool | str | None = None,
    numba_threads: int | None = None,
    record_temporal_inflows: bool = True,
    external_time_step_minutes: float | None = None,
    internal_time_step_minutes: float | None = None,
    akcelik_alpha: float | None = None,
    akcelik_j: float | None = None,
    akcelik_period_minutes: float | None = None,
) -> DynamicNetworkLoadingModel:
    canonical_name = canonical_network_name(network_name)
    dnl_settings = get_dnl_settings(canonical_name)
    resolved_external_time_step_minutes = float(
        dnl_settings.get("external_time_step_minutes", 15.0)
        if external_time_step_minutes is None
        else external_time_step_minutes
    )
    resolved_internal_time_step_minutes = float(
        dnl_settings.get("internal_time_step_minutes", resolved_external_time_step_minutes)
        if internal_time_step_minutes is None
        else internal_time_step_minutes
    )
    aggregation_factor = int(round(resolved_external_time_step_minutes / resolved_internal_time_step_minutes))
    if aggregation_factor <= 0:
        raise ValueError("internal_time_step_minutes must not exceed external_time_step_minutes.")
    resolved_clearance_steps = int(dnl_settings["clearance_steps"] if clearance_steps is None else clearance_steps)
    definition = build_network_definition(
        canonical_name,
        time_step_minutes=resolved_internal_time_step_minutes,
    )
    return DynamicNetworkLoadingModel(
        network=definition.network,
        od_pairs=definition.od_pairs,
        max_paths_per_od=int(
            dnl_settings["max_paths_per_od"] if max_paths_per_od is None else max_paths_per_od
        ),
        due_max_iterations=int(
            dnl_settings["due_max_iterations"] if due_max_iterations is None else due_max_iterations
        ),
        due_tolerance=float(dnl_settings["due_tolerance"] if due_tolerance is None else due_tolerance),
        clearance_steps=int(resolved_clearance_steps * aggregation_factor),
        stochastic_logit_scale=float(
            dnl_settings["stochastic_logit_scale"] if stochastic_logit_scale is None else stochastic_logit_scale
        ),
        route_choice_mode=str(dnl_settings["route_choice_mode"] if route_choice_mode is None else route_choice_mode),
        sample_route_choices=bool(
            dnl_settings["sample_route_choices"] if sample_route_choices is None else sample_route_choices
        ),
        route_choice_sampling_unit=float(
            dnl_settings["route_choice_sampling_unit"]
            if route_choice_sampling_unit is None
            else route_choice_sampling_unit
        ),
        random_seed=dnl_settings["random_seed"] if random_seed is None else random_seed,
        use_parallel_kernels=(
            dnl_settings["use_parallel_kernels"] if use_parallel_kernels is None else use_parallel_kernels
        ),
        numba_threads=dnl_settings["numba_threads"] if numba_threads is None else numba_threads,
        record_temporal_inflows=record_temporal_inflows,
        external_time_step_minutes=resolved_external_time_step_minutes,
        internal_time_step_minutes=resolved_internal_time_step_minutes,
        akcelik_alpha=float(dnl_settings.get("akcelik_alpha", 0.0) if akcelik_alpha is None else akcelik_alpha),
        akcelik_j=float(dnl_settings.get("akcelik_j", 0.8) if akcelik_j is None else akcelik_j),
        akcelik_period_minutes=float(
            dnl_settings.get("akcelik_period_minutes", resolved_external_time_step_minutes)
            if akcelik_period_minutes is None
            else akcelik_period_minutes
        ),
    )


def run_dnl_due(
    od_matrix: np.ndarray,
    include_clearance_steps: bool = False,
    return_details: bool = False,
    stochastic_logit_scale: float | None = None,
    route_choice_mode: str | None = None,
    sample_route_choices: bool | None = None,
    route_choice_sampling_unit: float | None = None,
    random_seed: int | None = None,
    network_name: str = "melbourne_scats",
    use_parallel_kernels: bool | str | None = None,
    numba_threads: int | None = None,
    record_temporal_inflows: bool = True,
    external_time_step_minutes: float | None = None,
    internal_time_step_minutes: float | None = None,
    akcelik_alpha: float | None = None,
    akcelik_j: float | None = None,
    akcelik_period_minutes: float | None = None,
) -> np.ndarray | AssignmentResult:
    model = build_default_model(
        network_name=network_name,
        stochastic_logit_scale=stochastic_logit_scale,
        route_choice_mode=route_choice_mode,
        sample_route_choices=sample_route_choices,
        route_choice_sampling_unit=route_choice_sampling_unit,
        random_seed=random_seed,
        use_parallel_kernels=use_parallel_kernels,
        numba_threads=numba_threads,
        record_temporal_inflows=record_temporal_inflows,
        external_time_step_minutes=external_time_step_minutes,
        internal_time_step_minutes=internal_time_step_minutes,
        akcelik_alpha=akcelik_alpha,
        akcelik_j=akcelik_j,
        akcelik_period_minutes=akcelik_period_minutes,
    )
    result = model.solve(od_matrix)
    if return_details:
        return result
    return result.full_link_inflows if include_clearance_steps else result.link_inflows

