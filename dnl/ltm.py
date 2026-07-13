from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ._kernels import (
    accumulate_downstream_demand_queue_kernel,
    apply_moves_queue_kernel,
    configure_numba_threads,
    count_source_loads_queue_kernel,
    departures_from_share_row_kernel,
    downstream_acceptance_kernel,
    duo_logit_shares_row_kernel,
    estimate_link_travel_times_kernel,
    estimate_path_costs_kernel,
    get_active_numba_threads,
    is_network_empty_queue_kernel,
    load_sources_queue_kernel,
    merge_pending_queue_kernel,
    prune_empty_heads_queue_kernel,
    receiving_kernel,
    sampled_departures_from_share_row_kernel,
    sending_kernel,
    snapshot_link_travel_times_kernel,
    snapshot_path_costs_kernel,
)
from .paths import Path
from .network.structures import Network

EPS = 1e-9


def _build_group_index(groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(groups) + 1, dtype=np.int32)
    flat_ids: list[int] = []
    cursor = 0
    for index, group in enumerate(groups):
        cursor += len(group)
        offsets[index + 1] = cursor
        flat_ids.extend(group)
    return offsets, np.asarray(flat_ids, dtype=np.int32)


def _infer_od_pairs_from_paths(
    paths: list[Path],
    paths_by_od: list[list[int]],
) -> tuple[tuple[int, int], ...]:
    od_pairs: list[tuple[int, int]] = []
    for path_ids in paths_by_od:
        if not path_ids:
            raise ValueError("Cannot infer an OD pair from an empty path group.")
        path = paths[int(path_ids[0])]
        od_pairs.append((int(path.origin), int(path.destination)))
    return tuple(od_pairs)


def _resolve_parallel_kernel_setting(
    requested: bool | str | None,
    num_links: int,
    num_paths: int,
) -> tuple[str, bool]:
    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized == "auto":
            return "auto", bool(num_links >= 48 or num_paths >= 512)
        if normalized in {"true", "on", "yes", "1"}:
            return normalized, True
        if normalized in {"false", "off", "no", "0"}:
            return normalized, False
        raise ValueError(
            "use_parallel_kernels must be one of: True, False, None, or 'auto'."
        )
    if requested is None:
        return "auto", bool(num_links >= 48 or num_paths >= 512)
    return ("explicit", bool(requested))


def _sample_departures_and_realized_shares_from_share_row(
    od_row: np.ndarray,
    share_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
    rng: np.random.Generator,
    sampling_unit: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_unit <= EPS:
        raise ValueError("sampling_unit must be positive when stochastic sampling is enabled.")

    od_row = np.asarray(od_row, dtype=np.float64)
    share_row = np.asarray(share_row, dtype=np.float64)
    unit = float(sampling_unit)
    nonnegative_demand = np.maximum(od_row, 0.0)
    whole_units = np.floor((nonnegative_demand / unit) + EPS).astype(np.int64)
    unit_offsets = np.empty(whole_units.shape[0] + 1, dtype=np.int64)
    unit_offsets[0] = 0
    np.cumsum(whole_units, out=unit_offsets[1:])
    total_units = int(unit_offsets[-1])
    unit_draws = rng.random(total_units, dtype=np.float64) if total_units > 0 else np.empty(0, dtype=np.float64)
    remainder_draws = rng.random(od_path_offsets.shape[0] - 1, dtype=np.float64)
    return sampled_departures_from_share_row_kernel(
        od_row=od_row,
        share_row=share_row,
        od_path_offsets=od_path_offsets,
        od_path_ids=od_path_ids,
        unit_offsets=unit_offsets,
        unit_draws=unit_draws,
        remainder_draws=remainder_draws,
        sampling_unit=unit,
        eps=EPS,
    )


def _sample_departures_from_share_row(
    od_row: np.ndarray,
    share_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
    rng: np.random.Generator,
    sampling_unit: float,
) -> np.ndarray:
    departures, _ = _sample_departures_and_realized_shares_from_share_row(
        od_row=od_row,
        share_row=share_row,
        od_path_offsets=od_path_offsets,
        od_path_ids=od_path_ids,
        rng=rng,
        sampling_unit=sampling_unit,
    )
    return departures


def _realized_shares_from_departures(
    od_row: np.ndarray,
    departure_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
    fallback_share_row: np.ndarray,
) -> np.ndarray:
    shares = np.zeros_like(fallback_share_row)
    for od_index in range(od_path_offsets.shape[0] - 1):
        start = int(od_path_offsets[od_index])
        end = int(od_path_offsets[od_index + 1])
        path_ids = od_path_ids[start:end]
        demand = max(float(od_row[od_index]), 0.0)
        if demand <= EPS:
            shares[path_ids] = fallback_share_row[path_ids]
            continue
        shares[path_ids] = departure_row[path_ids] / demand
    return shares


@dataclass(slots=True)
class LinkCohort:
    path_id: int
    path_pos: int
    departure_time: int
    entry_time: int
    amount: float


@dataclass(frozen=True)
class SimulationResult:
    demand_horizon: int
    actual_steps: int
    full_link_inflows: np.ndarray
    full_link_outflows: np.ndarray
    link_occupancies: np.ndarray
    link_travel_times: np.ndarray
    path_costs: np.ndarray
    cumulative_inflows: np.ndarray
    cumulative_outflows: np.ndarray
    arrived_volume: np.ndarray
    temporal_link_inflows: np.ndarray | None


@dataclass
class _SimulationWorkspace:
    demand_horizon: int
    max_steps: int
    link_inflows: np.ndarray
    link_outflows: np.ndarray
    cumulative_inflows: np.ndarray
    cumulative_outflows: np.ndarray
    arrived_volume: np.ndarray
    link_queues: list[deque[LinkCohort]]
    source_buffer: np.ndarray
    source_departure_buffer: np.ndarray
    temporal_link_inflows: np.ndarray | None
    temporal_aggregation_factor: int = 1
    temporal_horizon: int = 0
    temporal_current_offset: int = 0
    temporal_departure_offset: int = 0
    use_array_queues: bool = False
    queue_head: np.ndarray | None = None
    queue_tail: np.ndarray | None = None
    cohort_next: np.ndarray | None = None
    cohort_path_id: np.ndarray | None = None
    cohort_path_pos: np.ndarray | None = None
    cohort_departure_time: np.ndarray | None = None
    cohort_entry_time: np.ndarray | None = None
    cohort_amount: np.ndarray | None = None
    queue_capacity: int = 0
    queue_next_free: int = 0
    queue_active_nodes: int = 0
    pending_link: np.ndarray | None = None
    pending_path_id: np.ndarray | None = None
    pending_path_pos: np.ndarray | None = None
    pending_departure_time: np.ndarray | None = None
    pending_entry_time: np.ndarray | None = None
    pending_amount: np.ndarray | None = None
    pending_capacity: int = 0


@dataclass(frozen=True)
class DUOStepResult:
    time_step: int
    path_departure_row: np.ndarray
    path_share_row: np.ndarray
    route_choice_cost_row: np.ndarray
    link_inflow_row: np.ndarray
    link_occupancy_row: np.ndarray
    snapshot_link_travel_times: np.ndarray


class ForwardDUOSimulator:
    def __init__(
        self,
        loader: "LinkTransmissionModel",
        demand_horizon: int,
        paths_by_od: list[list[int]],
        logit_scale: float,
        od_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
        max_paths_per_od: int | None,
        sample_route_choices: bool = False,
        route_choice_sampling_unit: float = 1.0,
        random_seed: int | None = None,
    ) -> None:
        self.loader = loader
        self.od_pairs = (
            _infer_od_pairs_from_paths(loader.paths, paths_by_od)
            if od_pairs is None
            else tuple((int(origin), int(destination)) for origin, destination in od_pairs)
        )
        if len(self.od_pairs) != len(paths_by_od):
            raise ValueError("od_pairs must have the same length as paths_by_od.")
        self.paths_by_od = paths_by_od
        self.max_paths_per_od = int(
            max(len(path_ids) for path_ids in paths_by_od) if max_paths_per_od is None else max_paths_per_od
        )
        if self.max_paths_per_od <= 0:
            raise ValueError("max_paths_per_od must be positive for DUO route choice.")
        self.logit_scale = float(logit_scale)
        self.sample_route_choices = bool(sample_route_choices)
        self.route_choice_sampling_unit = float(route_choice_sampling_unit)
        if self.sample_route_choices and self.route_choice_sampling_unit <= EPS:
            raise ValueError(
                "route_choice_sampling_unit must be positive when stochastic route-choice "
                "sampling is enabled."
            )
        self.random_seed = random_seed
        self.rng = (
            np.random.default_rng(random_seed) if self.sample_route_choices else None
        )
        self.od_path_offsets, self.od_path_ids = _build_group_index(paths_by_od)
        self.workspace = loader._initialize_workspace(demand_horizon)
        self.path_departures = np.zeros((demand_horizon, len(loader.paths)), dtype=float)
        self.path_shares = np.zeros_like(self.path_departures)
        self.route_choice_costs = np.zeros_like(self.path_departures)
        self.current_step = 0

    def copy_for_single_step_candidate(self, temporal_step_index: int | None = None) -> "ForwardDUOSimulator":
        copied = object.__new__(type(self))
        copied.loader = self.loader
        copied.od_pairs = self.od_pairs
        copied.paths_by_od = self.paths_by_od
        copied.max_paths_per_od = int(self.max_paths_per_od)
        copied.logit_scale = float(self.logit_scale)
        copied.sample_route_choices = bool(self.sample_route_choices)
        copied.route_choice_sampling_unit = float(self.route_choice_sampling_unit)
        copied.random_seed = self.random_seed
        if self.rng is None:
            copied.rng = None
        else:
            copied.rng = np.random.default_rng()
            copied.rng.bit_generator.state = self.rng.bit_generator.state
        copied.od_path_offsets = self.od_path_offsets
        copied.od_path_ids = self.od_path_ids
        copied.workspace = self._copy_workspace_for_single_step_candidate(
            temporal_step_index=temporal_step_index
        )
        copied.path_departures = np.zeros_like(self.path_departures)
        copied.path_shares = np.zeros_like(self.path_shares)
        copied.route_choice_costs = np.zeros_like(self.route_choice_costs)
        copied.current_step = int(self.current_step)
        return copied

    def _copy_workspace_for_single_step_candidate(
        self,
        temporal_step_index: int | None,
    ) -> _SimulationWorkspace:
        workspace = self.workspace
        temporal_horizon = int(workspace.temporal_horizon)
        temporal_current_offset = int(workspace.temporal_current_offset)
        temporal_departure_offset = int(workspace.temporal_departure_offset)
        if workspace.temporal_link_inflows is None:
            temporal_link_inflows = None
        elif temporal_step_index is None:
            temporal_link_inflows = np.zeros_like(workspace.temporal_link_inflows)
        else:
            temporal_horizon = 1
            temporal_current_offset = int(temporal_step_index)
            temporal_departure_offset = int(temporal_step_index)
            temporal_link_inflows = np.zeros(
                (
                    1,
                    1,
                    workspace.temporal_link_inflows.shape[2],
                    workspace.temporal_link_inflows.shape[3],
                ),
                dtype=workspace.temporal_link_inflows.dtype,
            )

        return _SimulationWorkspace(
            demand_horizon=int(workspace.demand_horizon),
            max_steps=int(workspace.max_steps),
            link_inflows=workspace.link_inflows.copy(),
            link_outflows=workspace.link_outflows.copy(),
            cumulative_inflows=workspace.cumulative_inflows.copy(),
            cumulative_outflows=workspace.cumulative_outflows.copy(),
            arrived_volume=np.zeros_like(workspace.arrived_volume),
            link_queues=[deque(queue) for queue in workspace.link_queues],
            source_buffer=workspace.source_buffer.copy(),
            source_departure_buffer=workspace.source_departure_buffer.copy(),
            temporal_link_inflows=temporal_link_inflows,
            temporal_aggregation_factor=int(workspace.temporal_aggregation_factor),
            temporal_horizon=int(temporal_horizon),
            temporal_current_offset=int(temporal_current_offset),
            temporal_departure_offset=int(temporal_departure_offset),
            use_array_queues=bool(workspace.use_array_queues),
            queue_head=None if workspace.queue_head is None else workspace.queue_head.copy(),
            queue_tail=None if workspace.queue_tail is None else workspace.queue_tail.copy(),
            cohort_next=None if workspace.cohort_next is None else workspace.cohort_next.copy(),
            cohort_path_id=None if workspace.cohort_path_id is None else workspace.cohort_path_id.copy(),
            cohort_path_pos=None if workspace.cohort_path_pos is None else workspace.cohort_path_pos.copy(),
            cohort_departure_time=(
                None if workspace.cohort_departure_time is None else workspace.cohort_departure_time.copy()
            ),
            cohort_entry_time=None if workspace.cohort_entry_time is None else workspace.cohort_entry_time.copy(),
            cohort_amount=None if workspace.cohort_amount is None else workspace.cohort_amount.copy(),
            queue_capacity=int(workspace.queue_capacity),
            queue_next_free=int(workspace.queue_next_free),
            queue_active_nodes=int(workspace.queue_active_nodes),
            pending_link=None if workspace.pending_link is None else workspace.pending_link.copy(),
            pending_path_id=None if workspace.pending_path_id is None else workspace.pending_path_id.copy(),
            pending_path_pos=None if workspace.pending_path_pos is None else workspace.pending_path_pos.copy(),
            pending_departure_time=(
                None if workspace.pending_departure_time is None else workspace.pending_departure_time.copy()
            ),
            pending_entry_time=None if workspace.pending_entry_time is None else workspace.pending_entry_time.copy(),
            pending_amount=None if workspace.pending_amount is None else workspace.pending_amount.copy(),
            pending_capacity=int(workspace.pending_capacity),
        )

    @property
    def demand_horizon(self) -> int:
        return self.workspace.demand_horizon

    def step(self, od_row: np.ndarray) -> DUOStepResult:
        if self.current_step >= self.demand_horizon:
            raise RuntimeError("The DUO simulator has already processed all demand steps.")

        t = self.current_step
        od_row = np.asarray(od_row, dtype=float)
        if od_row.ndim != 1 or od_row.shape[0] != len(self.od_pairs):
            raise ValueError("od_row must be a 1D array with one value per OD pair.")
        if np.any(od_row < -EPS):
            raise ValueError("od_row cannot contain negative demand.")

        snapshot_link_costs = self.loader._estimate_snapshot_link_travel_times(
            cumulative_inflows=self.workspace.cumulative_inflows,
            cumulative_outflows=self.workspace.cumulative_outflows,
            t=t,
        )
        snapshot_path_costs = self.loader._estimate_snapshot_path_costs(snapshot_link_costs)
        share_row = self.loader._duo_logit_shares_row(
            od_row=od_row,
            path_costs_row=snapshot_path_costs,
            od_path_offsets=self.od_path_offsets,
            od_path_ids=self.od_path_ids,
            logit_scale=self.logit_scale,
        )
        if self.sample_route_choices:
            assert self.rng is not None
            departure_row, share_row = _sample_departures_and_realized_shares_from_share_row(
                od_row=od_row,
                share_row=share_row,
                od_path_offsets=self.od_path_offsets,
                od_path_ids=self.od_path_ids,
                rng=self.rng,
                sampling_unit=self.route_choice_sampling_unit,
            )
        else:
            departure_row = self.loader._departures_from_share_row(
                od_row=od_row,
                share_row=share_row,
                od_path_offsets=self.od_path_offsets,
                od_path_ids=self.od_path_ids,
            )

        self.route_choice_costs[t] = snapshot_path_costs
        self.path_shares[t] = share_row
        self.path_departures[t] = departure_row
        self.loader._advance_one_step(
            workspace=self.workspace,
            t=t,
            departures_at_t=departure_row,
        )
        self.current_step += 1

        return DUOStepResult(
            time_step=t,
            path_departure_row=departure_row.copy(),
            path_share_row=share_row.copy(),
            route_choice_cost_row=snapshot_path_costs.copy(),
            link_inflow_row=self.workspace.link_inflows[t].copy(),
            link_occupancy_row=(
                self.workspace.cumulative_inflows[t + 1] - self.workspace.cumulative_outflows[t + 1]
            ).copy(),
            snapshot_link_travel_times=snapshot_link_costs.copy(),
        )

    def finalize(self) -> tuple[SimulationResult, np.ndarray, np.ndarray, np.ndarray]:
        actual_steps = self.workspace.max_steps
        for t in range(self.current_step, self.workspace.max_steps):
            self.loader._advance_one_step(
                workspace=self.workspace,
                t=t,
                departures_at_t=None,
            )
            if t >= self.workspace.demand_horizon - 1 and self.loader._is_network_empty(
                self.workspace.link_queues,
                self.workspace.source_buffer,
                queue_head=self.workspace.queue_head,
                use_array_queues=self.workspace.use_array_queues,
            ):
                actual_steps = t + 1
                break

        simulation = self.loader._finalize_workspace(self.workspace, actual_steps)
        return (
            simulation,
            self.path_departures.copy(),
            self.path_shares.copy(),
            self.route_choice_costs.copy(),
        )


class LinkTransmissionModel:
    def __init__(
        self,
        network: Network,
        paths: list[Path],
        clearance_steps: int = 60,
        use_parallel_kernels: bool | str | None = None,
        numba_threads: int | None = None,
        record_temporal_inflows: bool = True,
        temporal_aggregation_factor: int = 1,
        akcelik_alpha: float = 0.0,
        akcelik_j: float = 0.8,
        akcelik_period_steps: float = 1.0,
    ) -> None:
        self.network = network
        self.paths = paths
        self.clearance_steps = clearance_steps
        self.record_temporal_inflows = bool(record_temporal_inflows)
        self.temporal_aggregation_factor = max(int(temporal_aggregation_factor), 1)
        self.akcelik_alpha = float(akcelik_alpha)
        self.akcelik_j = float(akcelik_j)
        self.akcelik_period_steps = float(akcelik_period_steps)
        if self.akcelik_alpha < 0.0:
            raise ValueError("akcelik_alpha must be non-negative.")
        if self.akcelik_j <= 0.0:
            raise ValueError("akcelik_j must be positive.")
        if self.akcelik_period_steps <= 0.0:
            raise ValueError("akcelik_period_steps must be positive.")

        self.free_flow_steps = np.array([link.free_flow_steps for link in network.links], dtype=int)
        self.backward_wave_steps = np.array([link.backward_wave_steps for link in network.links], dtype=int)
        self.capacity = np.array([link.capacity for link in network.links], dtype=float)
        self.jam_storage = np.array([link.jam_storage for link in network.links], dtype=float)
        self._rebuild_path_indexes()
        self._empty_temporal_link_inflows = np.zeros((0, 0, 0, 0), dtype=np.float32)
        self.parallel_kernel_mode, self.use_parallel_kernels = _resolve_parallel_kernel_setting(
            requested=use_parallel_kernels,
            num_links=self.network.num_links,
            num_paths=len(self.paths),
        )
        self.numba_threads = configure_numba_threads(numba_threads)
        self.active_numba_threads = get_active_numba_threads()

    def _akcelik_effective_delay(self, flow: float, capacity: float) -> float:
        if self.akcelik_alpha <= 0.0 or flow <= EPS:
            return 0.0
        saturation = max(float(flow), 0.0) / max(float(capacity), EPS)
        period_capacity = max(float(capacity) * self.akcelik_period_steps, EPS)
        z = saturation - 1.0
        radical = z * z + (8.0 * self.akcelik_j * saturation / period_capacity)
        return self.akcelik_alpha * self.akcelik_period_steps * (z + math.sqrt(max(radical, 0.0)))

    def _rebuild_path_indexes(self) -> None:
        self.num_od = max((path.od_index for path in self.paths), default=-1) + 1
        self.path_od_index = np.asarray([path.od_index for path in self.paths], dtype=np.int32)

        self.first_link_to_paths: dict[int, list[int]] = {}
        self.path_free_flow_cost = []
        for path in self.paths:
            self.first_link_to_paths.setdefault(path.links[0], []).append(path.path_id)
            self.path_free_flow_cost.append(
                sum(self.network.links[link_id].free_flow_steps for link_id in path.links)
            )
        self.max_path_free_flow = max(self.path_free_flow_cost, default=1)
        self.max_path_length = max((len(path.links) for path in self.paths), default=1)
        self.path_link_ids = np.full((len(self.paths), self.max_path_length), -1, dtype=np.int32)
        self.path_link_lengths = np.zeros(len(self.paths), dtype=np.int32)
        self.path_next_links = np.full((len(self.paths), self.max_path_length), -1, dtype=np.int32)
        for path in self.paths:
            path_len = len(path.links)
            self.path_link_lengths[path.path_id] = path_len
            self.path_link_ids[path.path_id, :path_len] = np.asarray(path.links, dtype=np.int32)
            if path_len > 1:
                self.path_next_links[path.path_id, : path_len - 1] = np.asarray(
                    path.links[1:],
                    dtype=np.int32,
                )

        first_link_groups = [self.first_link_to_paths.get(link_id, []) for link_id in range(self.network.num_links)]
        self.first_link_offsets, self.first_link_path_ids = _build_group_index(first_link_groups)

    def _initial_queue_capacity(self, demand_horizon: int) -> int:
        baseline = max(1024, self.network.num_links * 64, len(self.paths) * 4)
        return int(max(baseline, demand_horizon * max(8, len(self.paths) // 4)))

    def _ensure_queue_capacity(self, workspace: _SimulationWorkspace, additional_slots: int) -> None:
        if not workspace.use_array_queues:
            return
        required = int(workspace.queue_next_free + max(additional_slots, 0))
        if required <= workspace.queue_capacity:
            return

        new_capacity = max(required, workspace.queue_capacity * 2 if workspace.queue_capacity else 1024)
        def _grow_int_array(array: np.ndarray | None, fill_value: int = -1) -> np.ndarray:
            if array is None:
                grown = np.full(new_capacity, fill_value, dtype=np.int32)
            else:
                grown = np.full(new_capacity, fill_value, dtype=array.dtype)
                grown[: array.shape[0]] = array
            return grown

        def _grow_float_array(array: np.ndarray | None) -> np.ndarray:
            if array is None:
                grown = np.zeros(new_capacity, dtype=np.float64)
            else:
                grown = np.zeros(new_capacity, dtype=array.dtype)
                grown[: array.shape[0]] = array
            return grown

        workspace.cohort_next = _grow_int_array(workspace.cohort_next, fill_value=-1)
        workspace.cohort_path_id = _grow_int_array(workspace.cohort_path_id, fill_value=-1)
        workspace.cohort_path_pos = _grow_int_array(workspace.cohort_path_pos, fill_value=-1)
        workspace.cohort_departure_time = _grow_int_array(workspace.cohort_departure_time, fill_value=-1)
        workspace.cohort_entry_time = _grow_int_array(workspace.cohort_entry_time, fill_value=-1)
        workspace.cohort_amount = _grow_float_array(workspace.cohort_amount)
        workspace.queue_capacity = int(new_capacity)

    def _ensure_pending_capacity(self, workspace: _SimulationWorkspace, required_slots: int) -> None:
        if not workspace.use_array_queues:
            return
        required = int(max(required_slots, 0))
        if required <= workspace.pending_capacity:
            return

        new_capacity = max(required, workspace.pending_capacity * 2 if workspace.pending_capacity else 1024)
        def _grow_int_array(array: np.ndarray | None) -> np.ndarray:
            if array is None:
                grown = np.full(new_capacity, -1, dtype=np.int32)
            else:
                grown = np.full(new_capacity, -1, dtype=array.dtype)
                grown[: array.shape[0]] = array
            return grown

        def _grow_float_array(array: np.ndarray | None) -> np.ndarray:
            if array is None:
                grown = np.zeros(new_capacity, dtype=np.float64)
            else:
                grown = np.zeros(new_capacity, dtype=array.dtype)
                grown[: array.shape[0]] = array
            return grown

        workspace.pending_link = _grow_int_array(workspace.pending_link)
        workspace.pending_path_id = _grow_int_array(workspace.pending_path_id)
        workspace.pending_path_pos = _grow_int_array(workspace.pending_path_pos)
        workspace.pending_departure_time = _grow_int_array(workspace.pending_departure_time)
        workspace.pending_entry_time = _grow_int_array(workspace.pending_entry_time)
        workspace.pending_amount = _grow_float_array(workspace.pending_amount)
        workspace.pending_capacity = int(new_capacity)

    def simulate(self, path_departures: np.ndarray) -> SimulationResult:
        path_departures = np.asarray(path_departures, dtype=float)
        if path_departures.ndim != 2:
            raise ValueError("path_departures must be a 2D array shaped as [time, path].")
        if path_departures.shape[1] != len(self.paths):
            raise ValueError("The number of path columns must match the number of candidate paths.")
        if np.any(path_departures < -EPS):
            raise ValueError("path_departures cannot contain negative demand.")

        demand_horizon = path_departures.shape[0]
        workspace = self._initialize_workspace(demand_horizon)
        actual_steps = workspace.max_steps

        for t in range(workspace.max_steps):
            departures_at_t = path_departures[t] if t < demand_horizon else None
            self._advance_one_step(
                workspace=workspace,
                t=t,
                departures_at_t=departures_at_t,
            )
            if t >= demand_horizon - 1 and self._is_network_empty(
                workspace.link_queues,
                workspace.source_buffer,
                queue_head=workspace.queue_head,
                use_array_queues=workspace.use_array_queues,
            ):
                actual_steps = t + 1
                break

        return self._finalize_workspace(workspace, actual_steps)

    def simulate_duo(
        self,
        od_matrix: np.ndarray,
        paths_by_od: list[list[int]],
        logit_scale: float,
        od_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        max_paths_per_od: int | None = None,
        sample_route_choices: bool = False,
        route_choice_sampling_unit: float = 1.0,
        random_seed: int | None = None,
    ) -> tuple[SimulationResult, np.ndarray, np.ndarray, np.ndarray]:
        od_matrix = np.asarray(od_matrix, dtype=float)
        if od_matrix.ndim != 2:
            raise ValueError("od_matrix must be a 2D array shaped as [time, od_pair].")
        if od_matrix.shape[1] != len(paths_by_od):
            raise ValueError("The number of OD columns does not match the configured OD pairs.")
        if np.any(od_matrix < -EPS):
            raise ValueError("od_matrix cannot contain negative demand.")
        if logit_scale <= EPS:
            raise ValueError("logit_scale must be positive for DUO route choice.")

        runtime = self.make_duo_runtime(
            demand_horizon=od_matrix.shape[0],
            paths_by_od=paths_by_od,
            logit_scale=logit_scale,
            od_pairs=od_pairs,
            max_paths_per_od=max_paths_per_od,
            sample_route_choices=sample_route_choices,
            route_choice_sampling_unit=route_choice_sampling_unit,
            random_seed=random_seed,
        )
        for t in range(od_matrix.shape[0]):
            runtime.step(od_matrix[t])
        return runtime.finalize()

    def make_duo_runtime(
        self,
        demand_horizon: int,
        paths_by_od: list[list[int]],
        logit_scale: float,
        od_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        max_paths_per_od: int | None = None,
        sample_route_choices: bool = False,
        route_choice_sampling_unit: float = 1.0,
        random_seed: int | None = None,
    ) -> ForwardDUOSimulator:
        return ForwardDUOSimulator(
            loader=self,
            demand_horizon=demand_horizon,
            paths_by_od=paths_by_od,
            logit_scale=logit_scale,
            od_pairs=od_pairs,
            max_paths_per_od=max_paths_per_od,
            sample_route_choices=sample_route_choices,
            route_choice_sampling_unit=route_choice_sampling_unit,
            random_seed=random_seed,
        )

    def _initialize_workspace(self, demand_horizon: int) -> _SimulationWorkspace:
        max_steps = demand_horizon + self.clearance_steps + self.max_path_free_flow
        num_links = self.network.num_links
        use_array_queues = bool(self.use_parallel_kernels)
        queue_capacity = self._initial_queue_capacity(demand_horizon) if use_array_queues else 0
        pending_capacity = max(1024, queue_capacity // 2) if use_array_queues else 0
        temporal_horizon = int(np.ceil(float(demand_horizon) / float(self.temporal_aggregation_factor)))
        return _SimulationWorkspace(
            demand_horizon=demand_horizon,
            max_steps=max_steps,
            link_inflows=np.zeros((max_steps, num_links), dtype=float),
            link_outflows=np.zeros((max_steps, num_links), dtype=float),
            cumulative_inflows=np.zeros((max_steps + 1, num_links), dtype=float),
            cumulative_outflows=np.zeros((max_steps + 1, num_links), dtype=float),
            arrived_volume=np.zeros((demand_horizon, len(self.paths)), dtype=float),
            link_queues=[deque() for _ in range(num_links)],
            source_buffer=np.zeros(len(self.paths), dtype=float),
            source_departure_buffer=np.zeros((demand_horizon, len(self.paths)), dtype=float),
            temporal_link_inflows=(
                np.zeros((temporal_horizon, temporal_horizon, self.num_od, num_links), dtype=np.float32)
                if self.record_temporal_inflows
                else None
            ),
            temporal_aggregation_factor=int(self.temporal_aggregation_factor),
            temporal_horizon=int(temporal_horizon),
            temporal_current_offset=0,
            temporal_departure_offset=0,
            use_array_queues=use_array_queues,
            queue_head=np.full(num_links, -1, dtype=np.int32) if use_array_queues else None,
            queue_tail=np.full(num_links, -1, dtype=np.int32) if use_array_queues else None,
            cohort_next=np.full(queue_capacity, -1, dtype=np.int32) if use_array_queues else None,
            cohort_path_id=np.full(queue_capacity, -1, dtype=np.int32) if use_array_queues else None,
            cohort_path_pos=np.full(queue_capacity, -1, dtype=np.int32) if use_array_queues else None,
            cohort_departure_time=np.full(queue_capacity, -1, dtype=np.int32) if use_array_queues else None,
            cohort_entry_time=np.full(queue_capacity, -1, dtype=np.int32) if use_array_queues else None,
            cohort_amount=np.zeros(queue_capacity, dtype=np.float64) if use_array_queues else None,
            queue_capacity=queue_capacity,
            queue_next_free=0,
            queue_active_nodes=0,
            pending_link=np.full(pending_capacity, -1, dtype=np.int32) if use_array_queues else None,
            pending_path_id=np.full(pending_capacity, -1, dtype=np.int32) if use_array_queues else None,
            pending_path_pos=np.full(pending_capacity, -1, dtype=np.int32) if use_array_queues else None,
            pending_departure_time=np.full(pending_capacity, -1, dtype=np.int32) if use_array_queues else None,
            pending_entry_time=np.full(pending_capacity, -1, dtype=np.int32) if use_array_queues else None,
            pending_amount=np.zeros(pending_capacity, dtype=np.float64) if use_array_queues else None,
            pending_capacity=pending_capacity,
        )

    def _advance_one_step(
        self,
        workspace: _SimulationWorkspace,
        t: int,
        departures_at_t: np.ndarray | None = None,
    ) -> None:
        if workspace.use_array_queues:
            self._advance_one_step_array(workspace=workspace, t=t, departures_at_t=departures_at_t)
            return

        if departures_at_t is not None:
            departures_row = np.asarray(departures_at_t, dtype=float)
            workspace.source_buffer += departures_row
            workspace.source_departure_buffer[t] += departures_row

        receiving = self._compute_receiving(workspace.cumulative_inflows, workspace.cumulative_outflows, t)
        self._load_from_sources_with_departure_tracking(workspace=workspace, t=t, receiving=receiving)

        sending = self._compute_sending(workspace.cumulative_inflows, workspace.cumulative_outflows, t)
        demand_by_downstream = self._accumulate_downstream_demand(
            t=t,
            sending=sending,
            link_queues=workspace.link_queues,
        )
        accepted_ratio = self._compute_downstream_acceptance(receiving, demand_by_downstream)
        pending_entries: list[list[LinkCohort]] = [[] for _ in range(self.network.num_links)]

        self._apply_moves(
            t=t,
            sending=sending,
            accepted_ratio=accepted_ratio,
            link_queues=workspace.link_queues,
            pending_entries=pending_entries,
            link_inflows=workspace.link_inflows,
            link_outflows=workspace.link_outflows,
            arrived_volume=workspace.arrived_volume,
            demand_horizon=workspace.demand_horizon,
            workspace=workspace,
        )

        for queue in workspace.link_queues:
            while queue and queue[0].amount <= EPS:
                queue.popleft()

        for link_id, entries in enumerate(pending_entries):
            for cohort in entries:
                self._append_cohort(workspace.link_queues[link_id], cohort)

        workspace.cumulative_inflows[t + 1] = workspace.cumulative_inflows[t] + workspace.link_inflows[t]
        workspace.cumulative_outflows[t + 1] = workspace.cumulative_outflows[t] + workspace.link_outflows[t]

    def _advance_one_step_array(
        self,
        workspace: _SimulationWorkspace,
        t: int,
        departures_at_t: np.ndarray | None = None,
    ) -> None:
        if departures_at_t is not None:
            departures_row = np.asarray(departures_at_t, dtype=float)
            workspace.source_buffer += departures_row
            workspace.source_departure_buffer[t] += departures_row

        link_inflows_t = workspace.link_inflows[t]
        link_outflows_t = workspace.link_outflows[t]
        link_inflows_t.fill(0.0)
        link_outflows_t.fill(0.0)

        receiving = self._compute_receiving(workspace.cumulative_inflows, workspace.cumulative_outflows, t)
        self._load_from_sources_with_departure_tracking(workspace=workspace, t=t, receiving=receiving)

        sending = self._compute_sending(workspace.cumulative_inflows, workspace.cumulative_outflows, t)
        demand_by_downstream = accumulate_downstream_demand_queue_kernel(
            t=t,
            sending=sending,
            queue_head=workspace.queue_head,
            cohort_next=workspace.cohort_next,
            cohort_path_id=workspace.cohort_path_id,
            cohort_path_pos=workspace.cohort_path_pos,
            cohort_entry_time=workspace.cohort_entry_time,
            cohort_amount=workspace.cohort_amount,
            free_flow_steps=self.free_flow_steps,
            path_next_links=self.path_next_links,
            eps=EPS,
        )
        accepted_ratio = self._compute_downstream_acceptance(receiving, demand_by_downstream)

        self._ensure_pending_capacity(workspace, workspace.queue_active_nodes + len(self.paths))
        temporal_link_inflows = (
            workspace.temporal_link_inflows
            if workspace.temporal_link_inflows is not None
            else self._empty_temporal_link_inflows
        )
        pending_count = apply_moves_queue_kernel(
            t=t,
            sending=sending,
            accepted_ratio=accepted_ratio,
            queue_head=workspace.queue_head,
            cohort_next=workspace.cohort_next,
            cohort_path_id=workspace.cohort_path_id,
            cohort_path_pos=workspace.cohort_path_pos,
            cohort_departure_time=workspace.cohort_departure_time,
            cohort_entry_time=workspace.cohort_entry_time,
            cohort_amount=workspace.cohort_amount,
            free_flow_steps=self.free_flow_steps,
            path_next_links=self.path_next_links,
            path_od_index=self.path_od_index,
            pending_link=workspace.pending_link,
            pending_path_id=workspace.pending_path_id,
            pending_path_pos=workspace.pending_path_pos,
            pending_departure_time=workspace.pending_departure_time,
            pending_entry_time=workspace.pending_entry_time,
            pending_amount=workspace.pending_amount,
            link_inflows_t=link_inflows_t,
            link_outflows_t=link_outflows_t,
            arrived_volume=workspace.arrived_volume,
            temporal_link_inflows=temporal_link_inflows,
            temporal_aggregation_factor=int(workspace.temporal_aggregation_factor),
            temporal_horizon=int(workspace.temporal_horizon),
            temporal_current_offset=int(workspace.temporal_current_offset),
            temporal_departure_offset=int(workspace.temporal_departure_offset),
            record_temporal=workspace.temporal_link_inflows is not None,
            demand_horizon=workspace.demand_horizon,
            eps=EPS,
        )

        removed_nodes = prune_empty_heads_queue_kernel(
            queue_head=workspace.queue_head,
            queue_tail=workspace.queue_tail,
            cohort_next=workspace.cohort_next,
            cohort_amount=workspace.cohort_amount,
            eps=EPS,
        )
        workspace.queue_active_nodes -= int(removed_nodes)

        self._ensure_queue_capacity(workspace, int(pending_count))
        queue_next_free, added_pending_nodes = merge_pending_queue_kernel(
            pending_count=int(pending_count),
            pending_link=workspace.pending_link,
            pending_path_id=workspace.pending_path_id,
            pending_path_pos=workspace.pending_path_pos,
            pending_departure_time=workspace.pending_departure_time,
            pending_entry_time=workspace.pending_entry_time,
            pending_amount=workspace.pending_amount,
            queue_head=workspace.queue_head,
            queue_tail=workspace.queue_tail,
            cohort_next=workspace.cohort_next,
            cohort_path_id=workspace.cohort_path_id,
            cohort_path_pos=workspace.cohort_path_pos,
            cohort_departure_time=workspace.cohort_departure_time,
            cohort_entry_time=workspace.cohort_entry_time,
            cohort_amount=workspace.cohort_amount,
            link_inflows_t=link_inflows_t,
            next_free=workspace.queue_next_free,
            eps=EPS,
        )
        workspace.queue_next_free = int(queue_next_free)
        workspace.queue_active_nodes += int(added_pending_nodes)

        workspace.cumulative_inflows[t + 1] = workspace.cumulative_inflows[t] + link_inflows_t
        workspace.cumulative_outflows[t + 1] = workspace.cumulative_outflows[t] + link_outflows_t

    def _record_temporal_link_inflow(
        self,
        workspace: _SimulationWorkspace,
        current_time: int,
        departure_time: int,
        path_id: int,
        link_id: int,
        amount: float,
    ) -> None:
        if workspace.temporal_link_inflows is None:
            return
        if amount <= EPS or current_time < 0 or departure_time < 0:
            return
        if departure_time >= workspace.demand_horizon or link_id < 0:
            return
        aggregation_factor = max(int(workspace.temporal_aggregation_factor), 1)
        current_index = int(current_time) // aggregation_factor - int(workspace.temporal_current_offset)
        departure_index = int(departure_time) // aggregation_factor - int(workspace.temporal_departure_offset)
        if current_index < 0 or departure_index < 0:
            return
        if current_index >= int(workspace.temporal_horizon) or departure_index >= int(workspace.temporal_horizon):
            return
        od_index = int(self.path_od_index[path_id])
        workspace.temporal_link_inflows[current_index, departure_index, od_index, link_id] += np.float32(amount)

    def _append_array_cohort(
        self,
        workspace: _SimulationWorkspace,
        link_id: int,
        path_id: int,
        path_pos: int,
        departure_time: int,
        entry_time: int,
        amount: float,
    ) -> int:
        if amount <= EPS:
            return 0
        if workspace.queue_head is None or workspace.queue_tail is None:
            raise RuntimeError("Array queue buffers are missing.")
        if (
            workspace.cohort_next is None
            or workspace.cohort_path_id is None
            or workspace.cohort_path_pos is None
            or workspace.cohort_departure_time is None
            or workspace.cohort_entry_time is None
            or workspace.cohort_amount is None
        ):
            raise RuntimeError("Array queue cohort buffers are missing.")

        tail_idx = int(workspace.queue_tail[link_id])
        if tail_idx != -1:
            if (
                int(workspace.cohort_path_id[tail_idx]) == int(path_id)
                and int(workspace.cohort_path_pos[tail_idx]) == int(path_pos)
                and int(workspace.cohort_departure_time[tail_idx]) == int(departure_time)
                and int(workspace.cohort_entry_time[tail_idx]) == int(entry_time)
            ):
                workspace.cohort_amount[tail_idx] += float(amount)
                return 0

        self._ensure_queue_capacity(workspace, 1)
        idx = int(workspace.queue_next_free)
        workspace.cohort_next[idx] = -1
        workspace.cohort_path_id[idx] = int(path_id)
        workspace.cohort_path_pos[idx] = int(path_pos)
        workspace.cohort_departure_time[idx] = int(departure_time)
        workspace.cohort_entry_time[idx] = int(entry_time)
        workspace.cohort_amount[idx] = float(amount)

        if tail_idx == -1:
            workspace.queue_head[link_id] = idx
            workspace.queue_tail[link_id] = idx
        else:
            workspace.cohort_next[tail_idx] = idx
            workspace.queue_tail[link_id] = idx

        workspace.queue_next_free = idx + 1
        return 1

    def _load_from_sources_with_departure_tracking(
        self,
        workspace: _SimulationWorkspace,
        t: int,
        receiving: np.ndarray,
    ) -> None:
        link_inflows_t = workspace.link_inflows[t]
        if workspace.use_array_queues:
            if (
                workspace.queue_head is None
                or workspace.queue_tail is None
                or workspace.cohort_next is None
                or workspace.cohort_path_id is None
                or workspace.cohort_path_pos is None
                or workspace.cohort_departure_time is None
                or workspace.cohort_entry_time is None
                or workspace.cohort_amount is None
            ):
                raise RuntimeError("Array queue buffers are missing.")
            added_slots = count_source_loads_queue_kernel(
                receiving=receiving,
                first_link_offsets=self.first_link_offsets,
                first_link_path_ids=self.first_link_path_ids,
                source_buffer=workspace.source_buffer,
                source_departure_buffer=workspace.source_departure_buffer,
                eps=EPS,
            )
            self._ensure_queue_capacity(workspace, int(added_slots))
            temporal_link_inflows = (
                workspace.temporal_link_inflows
                if workspace.temporal_link_inflows is not None
                else self._empty_temporal_link_inflows
            )
            queue_next_free, added_nodes = load_sources_queue_kernel(
                t=t,
                receiving=receiving,
                first_link_offsets=self.first_link_offsets,
                first_link_path_ids=self.first_link_path_ids,
                path_od_index=self.path_od_index,
                source_buffer=workspace.source_buffer,
                source_departure_buffer=workspace.source_departure_buffer,
                queue_head=workspace.queue_head,
                queue_tail=workspace.queue_tail,
                cohort_next=workspace.cohort_next,
                cohort_path_id=workspace.cohort_path_id,
                cohort_path_pos=workspace.cohort_path_pos,
                cohort_departure_time=workspace.cohort_departure_time,
                cohort_entry_time=workspace.cohort_entry_time,
                cohort_amount=workspace.cohort_amount,
                link_inflows_t=link_inflows_t,
                temporal_link_inflows=temporal_link_inflows,
                temporal_aggregation_factor=int(workspace.temporal_aggregation_factor),
                temporal_horizon=int(workspace.temporal_horizon),
                temporal_current_offset=int(workspace.temporal_current_offset),
                temporal_departure_offset=int(workspace.temporal_departure_offset),
                next_free=int(workspace.queue_next_free),
                record_temporal=workspace.temporal_link_inflows is not None,
                eps=EPS,
            )
            workspace.queue_next_free = int(queue_next_free)
            workspace.queue_active_nodes += int(added_nodes)
            return

        added_nodes = 0

        for first_link in range(self.network.num_links):
            start = int(self.first_link_offsets[first_link])
            end = int(self.first_link_offsets[first_link + 1])
            if start == end:
                continue

            pending = 0.0
            for cursor in range(start, end):
                pending += float(workspace.source_buffer[int(self.first_link_path_ids[cursor])])
            if pending <= EPS or receiving[first_link] <= EPS:
                continue

            accepted_total = min(float(receiving[first_link]), pending)
            scale = accepted_total / pending

            for cursor in range(start, end):
                path_id = int(self.first_link_path_ids[cursor])
                path_pending = float(workspace.source_buffer[path_id])
                if path_pending <= EPS:
                    continue
                remaining = path_pending * scale
                if remaining <= EPS:
                    continue

                for departure_time in range(workspace.demand_horizon):
                    available = float(workspace.source_departure_buffer[departure_time, path_id])
                    if available <= EPS:
                        continue
                    moved = min(available, remaining)
                    if moved <= EPS:
                        continue

                    workspace.source_departure_buffer[departure_time, path_id] -= moved
                    workspace.source_buffer[path_id] -= moved

                    if workspace.use_array_queues:
                        added_nodes += self._append_array_cohort(
                            workspace=workspace,
                            link_id=first_link,
                            path_id=path_id,
                            path_pos=0,
                            departure_time=departure_time,
                            entry_time=t,
                            amount=moved,
                        )
                    else:
                        self._append_cohort(
                            workspace.link_queues[first_link],
                            LinkCohort(
                                path_id=path_id,
                                path_pos=0,
                                departure_time=departure_time,
                                entry_time=t,
                                amount=moved,
                            ),
                        )

                    link_inflows_t[first_link] += moved
                    self._record_temporal_link_inflow(
                        workspace=workspace,
                        current_time=t,
                        departure_time=departure_time,
                        path_id=path_id,
                        link_id=first_link,
                        amount=moved,
                    )
                    remaining -= moved
                    if remaining <= EPS:
                        break

                if workspace.source_buffer[path_id] < EPS:
                    workspace.source_buffer[path_id] = 0.0

            receiving[first_link] -= accepted_total

        if workspace.use_array_queues:
            workspace.queue_active_nodes += int(added_nodes)

    def _finalize_workspace(self, workspace: _SimulationWorkspace, actual_steps: int) -> SimulationResult:
        cumulative_inflows = workspace.cumulative_inflows[: actual_steps + 1]
        cumulative_outflows = workspace.cumulative_outflows[: actual_steps + 1]
        link_inflows = workspace.link_inflows[:actual_steps]
        link_outflows = workspace.link_outflows[:actual_steps]
        link_occupancies = cumulative_inflows[1:] - cumulative_outflows[1:]

        link_travel_times = self._estimate_link_travel_times(
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            actual_steps=actual_steps,
        )
        path_costs = self._estimate_path_costs(
            link_travel_times=link_travel_times,
            demand_horizon=workspace.demand_horizon,
        )

        return SimulationResult(
            demand_horizon=workspace.demand_horizon,
            actual_steps=actual_steps,
            full_link_inflows=link_inflows,
            full_link_outflows=link_outflows,
            link_occupancies=link_occupancies,
            link_travel_times=link_travel_times,
            path_costs=path_costs,
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            arrived_volume=workspace.arrived_volume,
            temporal_link_inflows=(
                workspace.temporal_link_inflows
                if workspace.temporal_link_inflows is not None
                else np.zeros((0, 0, 0, 0), dtype=np.float32)
            ),
        )

    def _estimate_snapshot_link_travel_times(
        self,
        cumulative_inflows: np.ndarray,
        cumulative_outflows: np.ndarray,
        t: int,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            snapshot = self.free_flow_steps.astype(float).copy()
            for link_id in range(self.network.num_links):
                ready_index = max(t + 1 - self.free_flow_steps[link_id], 0)
                ready_volume = cumulative_inflows[ready_index, link_id]
                exited_volume = cumulative_outflows[t, link_id]
                queue_backlog = max(0.0, ready_volume - exited_volume)
                recent_flow = 0.0
                if t > 0:
                    recent_flow = max(0.0, cumulative_inflows[t, link_id] - cumulative_inflows[t - 1, link_id])
                snapshot[link_id] += (
                    queue_backlog / max(self.capacity[link_id], EPS)
                    + self._akcelik_effective_delay(
                        recent_flow,
                        float(self.capacity[link_id]),
                    )
                )
            return snapshot
        return snapshot_link_travel_times_kernel(
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            free_flow_steps=self.free_flow_steps,
            capacity=self.capacity,
            t=t,
            akcelik_alpha=self.akcelik_alpha,
            akcelik_j=self.akcelik_j,
            akcelik_period_steps=self.akcelik_period_steps,
            eps=EPS,
        )

    def _estimate_snapshot_path_costs(self, snapshot_link_costs: np.ndarray) -> np.ndarray:
        if not self.use_parallel_kernels:
            path_costs = np.zeros(len(self.paths), dtype=float)
            for path in self.paths:
                path_costs[path.path_id] = float(np.sum(snapshot_link_costs[list(path.links)]))
            return path_costs
        return snapshot_path_costs_kernel(
            snapshot_link_costs=snapshot_link_costs,
            path_link_ids=self.path_link_ids,
            path_link_lengths=self.path_link_lengths,
        )

    def _duo_logit_shares_row(
        self,
        od_row: np.ndarray,
        path_costs_row: np.ndarray,
        od_path_offsets: np.ndarray,
        od_path_ids: np.ndarray,
        logit_scale: float,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            shares = np.zeros(len(self.paths), dtype=float)
            for od_index in range(od_path_offsets.shape[0] - 1):
                start = int(od_path_offsets[od_index])
                end = int(od_path_offsets[od_index + 1])
                path_ids = od_path_ids[start:end]
                if od_row[od_index] <= EPS:
                    shares[path_ids] = 1.0 / len(path_ids)
                    continue
                costs = path_costs_row[path_ids]
                logit_utility = -logit_scale * costs
                logit_utility -= np.max(logit_utility)
                exp_utility = np.exp(logit_utility)
                shares[path_ids] = exp_utility / max(float(np.sum(exp_utility)), EPS)
            return shares
        return duo_logit_shares_row_kernel(
            od_row=od_row,
            path_costs_row=path_costs_row,
            od_path_offsets=od_path_offsets,
            od_path_ids=od_path_ids,
            logit_scale=logit_scale,
            eps=EPS,
        )

    def _departures_from_share_row(
        self,
        od_row: np.ndarray,
        share_row: np.ndarray,
        od_path_offsets: np.ndarray,
        od_path_ids: np.ndarray,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            departures = np.zeros(len(self.paths), dtype=float)
            for od_index in range(od_path_offsets.shape[0] - 1):
                start = int(od_path_offsets[od_index])
                end = int(od_path_offsets[od_index + 1])
                path_ids = od_path_ids[start:end]
                departures[path_ids] = od_row[od_index] * share_row[path_ids]
            return departures
        return departures_from_share_row_kernel(
            od_row=od_row,
            share_row=share_row,
            od_path_offsets=od_path_offsets,
            od_path_ids=od_path_ids,
        )

    def _compute_sending(
        self,
        cumulative_inflows: np.ndarray,
        cumulative_outflows: np.ndarray,
        t: int,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            sending = np.zeros(self.network.num_links, dtype=float)
            current_outflows = cumulative_outflows[t]
            for link_id in range(self.network.num_links):
                ready_index = t + 1 - self.free_flow_steps[link_id]
                ready_inflow = cumulative_inflows[ready_index, link_id] if ready_index > 0 else 0.0
                sending[link_id] = min(
                    self.capacity[link_id],
                    max(0.0, ready_inflow - current_outflows[link_id]),
                )
            return sending
        return sending_kernel(
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            free_flow_steps=self.free_flow_steps,
            capacity=self.capacity,
            t=t,
        )

    def _compute_receiving(
        self,
        cumulative_inflows: np.ndarray,
        cumulative_outflows: np.ndarray,
        t: int,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            receiving = np.zeros(self.network.num_links, dtype=float)
            current_inflows = cumulative_inflows[t]
            for link_id in range(self.network.num_links):
                lag_index = max(t + 1 - self.backward_wave_steps[link_id], 0)
                remaining_storage = self.jam_storage[link_id] - (
                    current_inflows[link_id] - cumulative_outflows[lag_index, link_id]
                )
                receiving[link_id] = min(self.capacity[link_id], max(0.0, remaining_storage))
            return receiving
        return receiving_kernel(
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            backward_wave_steps=self.backward_wave_steps,
            capacity=self.capacity,
            jam_storage=self.jam_storage,
            t=t,
        )

    def _accumulate_downstream_demand(
        self,
        t: int,
        sending: np.ndarray,
        link_queues: list[deque[LinkCohort]],
    ) -> np.ndarray:
        demand_by_downstream = np.zeros(self.network.num_links, dtype=float)

        for link_id, queue in enumerate(link_queues):
            remaining_sending = sending[link_id]
            if remaining_sending <= EPS:
                continue

            for cohort in queue:
                if remaining_sending <= EPS:
                    break
                if t - cohort.entry_time < self.free_flow_steps[link_id]:
                    break

                candidate_amount = min(cohort.amount, remaining_sending)
                next_link = int(self.path_next_links[cohort.path_id, cohort.path_pos])
                if next_link >= 0:
                    demand_by_downstream[next_link] += candidate_amount
                remaining_sending -= candidate_amount

        return demand_by_downstream

    def _apply_moves(
        self,
        t: int,
        sending: np.ndarray,
        accepted_ratio: np.ndarray,
        link_queues: list[deque[LinkCohort]],
        pending_entries: list[list[LinkCohort]],
        link_inflows: np.ndarray,
        link_outflows: np.ndarray,
        arrived_volume: np.ndarray,
        demand_horizon: int,
        workspace: _SimulationWorkspace,
    ) -> None:
        for link_id, queue in enumerate(link_queues):
            remaining_sending = sending[link_id]
            if remaining_sending <= EPS:
                continue

            free_flow = self.free_flow_steps[link_id]
            for cohort in queue:
                if remaining_sending <= EPS:
                    break
                if t - cohort.entry_time < free_flow:
                    break

                candidate_amount = min(cohort.amount, remaining_sending)
                remaining_sending -= candidate_amount
                next_link = int(self.path_next_links[cohort.path_id, cohort.path_pos])
                moved = candidate_amount
                if next_link >= 0:
                    moved *= accepted_ratio[next_link]
                if moved <= EPS:
                    continue

                cohort.amount -= moved
                link_outflows[t, link_id] += moved

                if next_link < 0:
                    departure_time = cohort.departure_time
                    if departure_time < demand_horizon:
                        arrived_volume[departure_time, cohort.path_id] += moved
                    continue

                pending_entries[next_link].append(
                    LinkCohort(
                        path_id=cohort.path_id,
                        path_pos=cohort.path_pos + 1,
                        departure_time=cohort.departure_time,
                        entry_time=t,
                        amount=moved,
                    )
                )
                link_inflows[t, next_link] += moved
                self._record_temporal_link_inflow(
                    workspace=workspace,
                    current_time=t,
                    departure_time=cohort.departure_time,
                    path_id=cohort.path_id,
                    link_id=next_link,
                    amount=moved,
                )

    def _compute_downstream_acceptance(
        self,
        receiving: np.ndarray,
        demand_by_downstream: np.ndarray,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            accepted_ratio = np.ones(self.network.num_links, dtype=float)
            for link_id in range(self.network.num_links):
                demand = demand_by_downstream[link_id]
                if demand <= EPS:
                    continue
                accepted_ratio[link_id] = min(receiving[link_id], demand) / demand
            return accepted_ratio
        return downstream_acceptance_kernel(
            receiving=receiving,
            demand_by_downstream=demand_by_downstream,
            eps=EPS,
        )

    def _append_cohort(self, queue: deque[LinkCohort], new_cohort: LinkCohort) -> None:
        if new_cohort.amount <= EPS:
            return
        if queue:
            last = queue[-1]
            if (
                last.path_id == new_cohort.path_id
                and last.path_pos == new_cohort.path_pos
                and last.departure_time == new_cohort.departure_time
                and last.entry_time == new_cohort.entry_time
            ):
                last.amount += new_cohort.amount
                return
        queue.append(new_cohort)

    def _is_network_empty(
        self,
        link_queues: list[deque[LinkCohort]],
        source_buffer: np.ndarray,
        queue_head: np.ndarray | None = None,
        use_array_queues: bool = False,
    ) -> bool:
        if use_array_queues:
            if queue_head is None:
                raise ValueError("queue_head must be provided when use_array_queues=True.")
            return bool(is_network_empty_queue_kernel(queue_head=queue_head, source_buffer=source_buffer, eps=EPS))
        if np.any(source_buffer > EPS):
            return False
        return not any(queue for queue in link_queues)

    def _estimate_link_travel_times(
        self,
        cumulative_inflows: np.ndarray,
        cumulative_outflows: np.ndarray,
        actual_steps: int,
    ) -> np.ndarray:
        if not self.use_parallel_kernels:
            link_travel_times = np.zeros((actual_steps, self.network.num_links), dtype=float)

            for link_id in range(self.network.num_links):
                for t in range(actual_steps):
                    probe_rank = cumulative_inflows[t + 1, link_id]
                    earliest_exit = min(actual_steps - 1, t + self.free_flow_steps[link_id])
                    exit_step = earliest_exit
                    while (
                        exit_step < actual_steps - 1
                        and cumulative_outflows[exit_step + 1, link_id] + EPS < probe_rank
                    ):
                        exit_step += 1
                    travel_time = max(
                        float(self.free_flow_steps[link_id]),
                        float(exit_step - t),
                    )
                    entry_flow = max(0.0, cumulative_inflows[t + 1, link_id] - cumulative_inflows[t, link_id])
                    link_travel_times[t, link_id] = travel_time + self._akcelik_effective_delay(
                        entry_flow,
                        float(self.capacity[link_id]),
                    )

            return link_travel_times
        return estimate_link_travel_times_kernel(
            cumulative_inflows=cumulative_inflows,
            cumulative_outflows=cumulative_outflows,
            free_flow_steps=self.free_flow_steps,
            capacity=self.capacity,
            actual_steps=actual_steps,
            akcelik_alpha=self.akcelik_alpha,
            akcelik_j=self.akcelik_j,
            akcelik_period_steps=self.akcelik_period_steps,
            eps=EPS,
        )

    def _estimate_path_costs(self, link_travel_times: np.ndarray, demand_horizon: int) -> np.ndarray:
        if not self.use_parallel_kernels:
            path_costs = np.zeros((demand_horizon, len(self.paths)), dtype=float)
            final_row = link_travel_times.shape[0] - 1

            for path in self.paths:
                for departure_time in range(demand_horizon):
                    current_time = departure_time
                    total_cost = 0.0
                    for link_id in path.links:
                        row = min(current_time, final_row)
                        travel_time = link_travel_times[row, link_id]
                        total_cost += travel_time
                        current_time += int(np.ceil(travel_time))
                    path_costs[departure_time, path.path_id] = total_cost

            return path_costs
        return estimate_path_costs_kernel(
            link_travel_times=link_travel_times,
            path_link_ids=self.path_link_ids,
            path_link_lengths=self.path_link_lengths,
            demand_horizon=demand_horizon,
        )

