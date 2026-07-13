from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinkFlowPropagationGuidance:
    link_flow_propagation_guidance: np.ndarray
    temporal_mass: np.ndarray
    stats: dict[str, float]


def extract_link_flow_propagation_guidance_components(
    temporal_link_inflows: np.ndarray,
    simulated_link_flows: np.ndarray,
    target_observations: np.ndarray,
    flow_scale: np.ndarray,
    gamma: float,
    observed_link_indices: np.ndarray,
    observation_scale: np.ndarray | None = None,
) -> LinkFlowPropagationGuidance:
    """
    Extract per-(time, OD) link-flow propagation guidance from realized propagation.

    temporal_link_inflows is indexed as [future_time, departure_time, od, link].
    For each action a_{t,od}, the returned information accumulates future reward
    sensitivities over all links reached by that exact propagated cohort volume.

    The volume-weighted signal is used by LFP-A.
    """
    temporal_link_inflows = np.asarray(temporal_link_inflows, dtype=np.float32)
    simulated_link_flows = np.asarray(simulated_link_flows, dtype=np.float32)
    target_observations = np.asarray(target_observations, dtype=np.float32)
    observed_link_indices = np.asarray(observed_link_indices, dtype=np.int64).reshape(-1)
    flow_scale = np.asarray(flow_scale, dtype=np.float32).reshape(-1)

    if temporal_link_inflows.ndim != 4:
        raise ValueError("temporal_link_inflows must be shaped as [time, departure_time, od, link].")

    reward_horizon = min(
        int(temporal_link_inflows.shape[0]),
        int(temporal_link_inflows.shape[1]),
        int(simulated_link_flows.shape[0]),
        int(target_observations.shape[0]),
    )
    num_od = int(temporal_link_inflows.shape[2])
    num_links = int(temporal_link_inflows.shape[3])
    if reward_horizon <= 0 or num_od <= 0 or num_links <= 0:
        raise ValueError("Link-flow propagation guidance inputs are empty.")
    if flow_scale.shape[0] < num_links:
        raise ValueError(f"flow_scale must have at least {num_links} entries, got {flow_scale.shape[0]}.")

    flow_scale = flow_scale[:num_links]
    if observed_link_indices.size <= 0:
        raise ValueError("observed_link_indices must include at least one detector link.")
    if int(np.min(observed_link_indices)) < 0 or int(np.max(observed_link_indices)) >= num_links:
        raise ValueError(
            "observed_link_indices out of range: "
            f"valid range is [0, {num_links - 1}], got {observed_link_indices.tolist()}."
        )
    observed_count = int(observed_link_indices.shape[0])
    if target_observations.ndim != 2 or target_observations.shape[1] != observed_count:
        raise ValueError(
            "target_observations must have shape [time, num_observed_links]; "
            f"got {target_observations.shape} for {observed_count} observed links."
        )
    if observation_scale is None:
        observation_scale = flow_scale[observed_link_indices]
    observation_scale = np.maximum(np.asarray(observation_scale, dtype=np.float32).reshape(-1), 1.0)
    if observation_scale.shape != (observed_count,):
        raise ValueError(
            "observation_scale length must match observed_link_indices: "
            f"expected {observed_count}, got {observation_scale.shape[0]}."
        )
    simulated_observations = simulated_link_flows[:reward_horizon, observed_link_indices]
    observation_sensitivity = (
        2.0
        * (target_observations[:reward_horizon] - simulated_observations)
        / np.square(observation_scale)[None, :]
        / float(observed_count)
    ).astype(np.float32)

    link_flow_propagation_guidance = np.zeros((reward_horizon, num_od), dtype=np.float32)
    temporal_mass = np.zeros_like(link_flow_propagation_guidance)

    for departure_time in range(reward_horizon):
        future_volume = np.take(
            temporal_link_inflows[departure_time:reward_horizon, departure_time],
            observed_link_indices,
            axis=2,
        )
        if future_volume.size == 0:
            continue
        discounts = (float(gamma) ** np.arange(future_volume.shape[0], dtype=np.float32)).reshape(-1, 1, 1)
        weighted_sensitivity = observation_sensitivity[departure_time:reward_horizon][:, None, :]
        link_flow_propagation_guidance[departure_time] = np.sum(
            future_volume * weighted_sensitivity * discounts,
            axis=(0, 2),
        )
        temporal_mass[departure_time] = np.sum(future_volume * discounts, axis=(0, 2))

    stats = {
        "lfp_information_abs_mean": float(np.mean(np.abs(link_flow_propagation_guidance))),
        "lfp_information_abs_max": float(np.max(np.abs(link_flow_propagation_guidance))),
        "temporal_mass_mean": float(np.mean(temporal_mass)),
        "temporal_mass_max": float(np.max(temporal_mass)),
        "link_sensitivity_abs_mean": float(np.mean(np.abs(observation_sensitivity))),
    }
    return LinkFlowPropagationGuidance(
        link_flow_propagation_guidance=link_flow_propagation_guidance,
        temporal_mass=temporal_mass,
        stats=stats,
    )


def extract_link_flow_propagation_guidance(
    temporal_link_inflows: np.ndarray,
    simulated_link_flows: np.ndarray,
    target_observations: np.ndarray,
    flow_scale: np.ndarray,
    gamma: float,
    observed_link_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    guidance = extract_link_flow_propagation_guidance_components(
        temporal_link_inflows=temporal_link_inflows,
        simulated_link_flows=simulated_link_flows,
        target_observations=target_observations,
        flow_scale=flow_scale,
        gamma=gamma,
        observed_link_indices=observed_link_indices,
    )
    return guidance.link_flow_propagation_guidance, guidance.stats


# Compatibility alias for local guidance-target imports.
link_flow_propagation_guidance_advantage_targets = extract_link_flow_propagation_guidance
