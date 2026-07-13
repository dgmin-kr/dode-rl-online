from __future__ import annotations

import numpy as np

from utils.assignment_guidance import compute_assignment_gradient


def compute_lfpg_gd_gradient(
    *,
    od_matrix: np.ndarray,
    temporal_link_inflows: np.ndarray,
    simulated_link_flows: np.ndarray,
    target_observations: np.ndarray,
    observed_link_indices: np.ndarray,
    flow_scale: np.ndarray | None = None,
) -> np.ndarray:
    if flow_scale is None:
        flow_scale = np.ones(np.asarray(simulated_link_flows, dtype=np.float64).shape[1], dtype=np.float64)
    return compute_assignment_gradient(
        od_matrix=od_matrix,
        temporal_link_inflows=temporal_link_inflows,
        simulated_link_flows=simulated_link_flows,
        target_observations=target_observations,
        observed_link_indices=observed_link_indices,
        flow_scale=np.asarray(flow_scale, dtype=np.float64),
    )
