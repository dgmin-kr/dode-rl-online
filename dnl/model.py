from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .network.structures import Network

from .due import StochasticRouteChoiceSolver
from .ltm import DUOStepResult, ForwardDUOSimulator, LinkTransmissionModel, SimulationResult
from .paths import build_candidate_paths


@dataclass(frozen=True)
class AssignmentResult:
    link_inflows: np.ndarray
    full_link_inflows: np.ndarray
    full_link_outflows: np.ndarray
    link_occupancies: np.ndarray
    link_travel_times: np.ndarray
    temporal_link_inflows: np.ndarray
    path_costs: np.ndarray
    route_choice_costs: np.ndarray
    path_shares: np.ndarray
    gap_history: tuple[float, ...]
    iterations: int
    od_pairs: tuple[tuple[int, int], ...]
    path_labels: tuple[str, ...]
    link_labels: tuple[str, ...]
    route_choice_model: str
    logit_scale: float
    sample_route_choices: bool
    route_choice_sampling_unit: float
    random_seed: int | None


class ExternalTimeStepDUORuntime:
    def __init__(
        self,
        *,
        model: "DynamicNetworkLoadingModel",
        internal_runtime: ForwardDUOSimulator,
        external_demand_horizon: int,
    ) -> None:
        self.model = model
        self.internal_runtime = internal_runtime
        self.external_demand_horizon = int(external_demand_horizon)
        self.current_step = 0
        self.link_inflows = np.zeros((self.external_demand_horizon, model.network.num_links), dtype=float)
        self.link_occupancies = np.zeros_like(self.link_inflows)
        self.snapshot_link_travel_times = np.tile(
            model.loader.free_flow_steps.astype(float),
            (self.external_demand_horizon, 1),
        )

    @property
    def demand_horizon(self) -> int:
        return self.external_demand_horizon

    @property
    def workspace(self):
        return self.internal_runtime.workspace

    def copy_for_single_step_candidate(self) -> "ExternalTimeStepDUORuntime":
        copied = object.__new__(type(self))
        copied.model = self.model
        copied.internal_runtime = self.internal_runtime.copy_for_single_step_candidate(
            temporal_step_index=int(self.current_step)
        )
        copied.external_demand_horizon = int(self.external_demand_horizon)
        copied.current_step = int(self.current_step)
        copied.link_inflows = np.zeros_like(self.link_inflows)
        copied.link_occupancies = np.zeros_like(self.link_occupancies)
        copied.snapshot_link_travel_times = np.zeros_like(self.snapshot_link_travel_times)
        return copied

    def step(self, od_row: np.ndarray) -> DUOStepResult:
        if self.current_step >= self.external_demand_horizon:
            raise RuntimeError("The DUO simulator has already processed all external demand steps.")

        od_row = np.asarray(od_row, dtype=float)
        factor = int(self.model.aggregation_factor)
        internal_od_row = od_row / float(factor)
        path_departure_rows: list[np.ndarray] = []
        path_share_rows: list[np.ndarray] = []
        route_choice_cost_rows: list[np.ndarray] = []
        link_inflow_row = np.zeros(self.model.network.num_links, dtype=float)
        link_occupancy_row = np.zeros(self.model.network.num_links, dtype=float)
        snapshot_link_travel_times = self.model.loader.free_flow_steps.astype(float).copy()

        for _ in range(factor):
            step_result = self.internal_runtime.step(internal_od_row)
            path_departure_rows.append(step_result.path_departure_row)
            path_share_rows.append(step_result.path_share_row)
            route_choice_cost_rows.append(step_result.route_choice_cost_row)
            link_inflow_row += step_result.link_inflow_row
            link_occupancy_row = step_result.link_occupancy_row
            snapshot_link_travel_times = step_result.snapshot_link_travel_times

        external_step = self.current_step
        self.link_inflows[external_step] = link_inflow_row
        self.link_occupancies[external_step] = link_occupancy_row
        self.snapshot_link_travel_times[external_step] = snapshot_link_travel_times
        self.current_step += 1

        return DUOStepResult(
            time_step=external_step,
            path_departure_row=np.sum(np.asarray(path_departure_rows, dtype=float), axis=0),
            path_share_row=np.mean(np.asarray(path_share_rows, dtype=float), axis=0),
            route_choice_cost_row=np.mean(np.asarray(route_choice_cost_rows, dtype=float), axis=0),
            link_inflow_row=link_inflow_row.copy(),
            link_occupancy_row=link_occupancy_row.copy(),
            snapshot_link_travel_times=snapshot_link_travel_times.copy(),
        )

    def current_state_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.current_step <= 0:
            return (
                np.zeros(self.model.network.num_links, dtype=float),
                np.zeros(self.model.network.num_links, dtype=float),
                np.ones(self.model.network.num_links, dtype=float),
            )
        row_index = min(self.current_step - 1, self.external_demand_horizon - 1)
        speed_index = np.clip(
            self.model.loader.free_flow_steps.astype(float)
            / np.maximum(self.snapshot_link_travel_times[row_index], self.model.loader.free_flow_steps),
            0.0,
            1.0,
        )
        return (
            self.link_inflows[row_index].copy(),
            self.link_occupancies[row_index].copy(),
            speed_index.astype(float, copy=True),
        )

    def finalize(self) -> tuple[SimulationResult, np.ndarray, np.ndarray, np.ndarray]:
        simulation, path_departures, path_shares, route_choice_costs = self.internal_runtime.finalize()
        return (
            simulation,
            self.model._aggregate_sum_rows(path_departures, self.external_demand_horizon),
            self.model._aggregate_mean_rows(path_shares, self.external_demand_horizon),
            self.model._aggregate_mean_rows(route_choice_costs, self.external_demand_horizon),
        )


class DynamicNetworkLoadingModel:
    def __init__(
        self,
        network: Network,
        od_pairs: list[tuple[int, int]],
        max_paths_per_od: int = 4,
        due_max_iterations: int = 30,
        due_tolerance: float = 1e-3,
        clearance_steps: int = 60,
        stochastic_logit_scale: float = 0.35,
        route_choice_mode: str = "due",
        sample_route_choices: bool = False,
        route_choice_sampling_unit: float = 1.0,
        random_seed: int | None = None,
        use_parallel_kernels: bool | str | None = None,
        numba_threads: int | None = None,
        record_temporal_inflows: bool = True,
        external_time_step_minutes: float = 15.0,
        internal_time_step_minutes: float = 15.0,
        akcelik_alpha: float = 0.0,
        akcelik_j: float = 0.8,
        akcelik_period_minutes: float | None = None,
    ) -> None:
        self.network = network
        self.od_pairs = od_pairs
        self.external_time_step_minutes = float(external_time_step_minutes)
        self.internal_time_step_minutes = float(internal_time_step_minutes)
        if self.external_time_step_minutes <= 0.0 or self.internal_time_step_minutes <= 0.0:
            raise ValueError("DNL time-step minutes must be positive.")
        ratio = self.external_time_step_minutes / self.internal_time_step_minutes
        self.aggregation_factor = int(round(ratio))
        if self.aggregation_factor <= 0 or not np.isclose(ratio, float(self.aggregation_factor), rtol=1e-8, atol=1e-8):
            raise ValueError(
                "external_time_step_minutes must be an integer multiple of internal_time_step_minutes: "
                f"external={self.external_time_step_minutes}, internal={self.internal_time_step_minutes}."
            )
        self.route_choice_mode = route_choice_mode
        self.sample_route_choices = bool(sample_route_choices)
        self.route_choice_sampling_unit = float(route_choice_sampling_unit)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.record_temporal_inflows = bool(record_temporal_inflows)
        self.akcelik_alpha = float(akcelik_alpha)
        self.akcelik_j = float(akcelik_j)
        self.akcelik_period_minutes = float(
            self.external_time_step_minutes if akcelik_period_minutes is None else akcelik_period_minutes
        )
        self.akcelik_period_steps = self.akcelik_period_minutes / self.internal_time_step_minutes
        sampling_suffix = "_sampled" if self.sample_route_choices else "_expected"
        self.route_choice_model = f"stochastic_logit_{route_choice_mode}{sampling_suffix}"
        self.logit_scale = float(stochastic_logit_scale)
        self.paths, self.paths_by_od = build_candidate_paths(
            network=network,
            od_pairs=od_pairs,
            max_paths_per_od=max_paths_per_od,
        )
        self.path_labels = tuple(path.label for path in self.paths)
        self.link_labels = tuple(link.label for link in self.network.links)

        self.loader = LinkTransmissionModel(
            network=network,
            paths=self.paths,
            clearance_steps=clearance_steps,
            use_parallel_kernels=use_parallel_kernels,
            numba_threads=numba_threads,
            record_temporal_inflows=self.record_temporal_inflows,
            temporal_aggregation_factor=self.aggregation_factor,
            akcelik_alpha=self.akcelik_alpha,
            akcelik_j=self.akcelik_j,
            akcelik_period_steps=self.akcelik_period_steps,
        )
        self.max_paths_per_od = int(max_paths_per_od)
        self.due_max_iterations = int(due_max_iterations)
        self.due_tolerance = float(due_tolerance)
        self.clearance_steps = int(clearance_steps)
        self.use_parallel_kernels = bool(self.loader.use_parallel_kernels)
        self.parallel_kernel_mode = str(self.loader.parallel_kernel_mode)
        self.numba_threads = self.loader.numba_threads
        self.active_numba_threads = int(self.loader.active_numba_threads)
        self.due_solver = StochasticRouteChoiceSolver(
            paths=self.paths,
            paths_by_od=self.paths_by_od,
            loader=self.loader,
            max_iterations=due_max_iterations,
            tolerance=due_tolerance,
            logit_scale=self.logit_scale,
            route_choice_mode=self.route_choice_mode,
            sample_route_choices=self.sample_route_choices,
            route_choice_sampling_unit=self.route_choice_sampling_unit,
            random_seed=self.random_seed,
        )

    def set_random_seed(self, random_seed: int | None) -> None:
        self.random_seed = None if random_seed is None else int(random_seed)
        self.due_solver.set_random_seed(self.random_seed)

    def run(self, od_matrix: np.ndarray, include_clearance_steps: bool = False) -> np.ndarray:
        result = self.solve(od_matrix)
        return result.full_link_inflows if include_clearance_steps else result.link_inflows

    def _expand_external_od_matrix(self, od_matrix: np.ndarray) -> np.ndarray:
        od_matrix = np.asarray(od_matrix, dtype=float)
        if self.aggregation_factor <= 1:
            return od_matrix.copy()
        return np.repeat(od_matrix / float(self.aggregation_factor), self.aggregation_factor, axis=0)

    def _aggregate_sum_rows(self, matrix: np.ndarray, external_horizon: int | None = None) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if self.aggregation_factor <= 1:
            if external_horizon is None:
                return matrix.copy()
            return matrix[: int(external_horizon)].copy()
        if matrix.shape[0] == 0:
            return matrix.copy()
        factor = int(self.aggregation_factor)
        horizon = int(np.ceil(matrix.shape[0] / float(factor))) if external_horizon is None else int(external_horizon)
        usable = min(matrix.shape[0], horizon * factor)
        padded_shape = (horizon * factor, *matrix.shape[1:])
        padded = np.zeros(padded_shape, dtype=matrix.dtype)
        padded[:usable] = matrix[:usable]
        return padded.reshape(horizon, factor, *matrix.shape[1:]).sum(axis=1)

    def _aggregate_mean_rows(self, matrix: np.ndarray, external_horizon: int | None = None) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if self.aggregation_factor <= 1:
            if external_horizon is None:
                return matrix.copy()
            return matrix[: int(external_horizon)].copy()
        if matrix.shape[0] == 0:
            return matrix.copy()
        factor = int(self.aggregation_factor)
        horizon = int(np.ceil(matrix.shape[0] / float(factor))) if external_horizon is None else int(external_horizon)
        rows: list[np.ndarray] = []
        for external_index in range(horizon):
            start = external_index * factor
            end = min(start + factor, matrix.shape[0])
            if start >= matrix.shape[0]:
                rows.append(np.zeros(matrix.shape[1:], dtype=float))
            else:
                rows.append(np.mean(matrix[start:end], axis=0))
        return np.asarray(rows, dtype=float)

    def _aggregate_last_rows(self, matrix: np.ndarray, external_horizon: int | None = None) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if self.aggregation_factor <= 1:
            if external_horizon is None:
                return matrix.copy()
            return matrix[: int(external_horizon)].copy()
        if matrix.shape[0] == 0:
            return matrix.copy()
        factor = int(self.aggregation_factor)
        horizon = int(np.ceil(matrix.shape[0] / float(factor))) if external_horizon is None else int(external_horizon)
        rows: list[np.ndarray] = []
        for external_index in range(horizon):
            end = min((external_index + 1) * factor, matrix.shape[0]) - 1
            if end < 0:
                rows.append(np.zeros(matrix.shape[1:], dtype=float))
            else:
                rows.append(matrix[end])
        return np.asarray(rows, dtype=float)

    def solve(self, od_matrix: np.ndarray) -> AssignmentResult:
        od_matrix = np.asarray(od_matrix, dtype=float)
        if od_matrix.ndim != 2:
            raise ValueError("od_matrix must be a 2D array shaped as [time, od_pair].")
        if od_matrix.shape[1] != len(self.od_pairs):
            raise ValueError(
                f"Expected {len(self.od_pairs)} OD columns in the order {self.od_pairs}, "
                f"but received {od_matrix.shape[1]} columns."
            )

        internal_od_matrix = self._expand_external_od_matrix(od_matrix)
        due_result = self.due_solver.solve(internal_od_matrix)
        return self._build_assignment_result(
            simulation=due_result.simulation,
            route_choice_costs=self._aggregate_mean_rows(due_result.route_choice_costs, od_matrix.shape[0]),
            path_shares=self._aggregate_mean_rows(due_result.path_shares, od_matrix.shape[0]),
            gap_history=due_result.gap_history,
            iterations=due_result.iterations,
            demand_horizon=od_matrix.shape[0],
        )

    def make_duo_runtime(self, demand_horizon: int) -> ForwardDUOSimulator | ExternalTimeStepDUORuntime:
        if self.route_choice_mode != "duo":
            raise RuntimeError("make_duo_runtime() is only available when route_choice_mode='duo'.")
        internal_runtime = self.loader.make_duo_runtime(
            demand_horizon=int(demand_horizon) * int(self.aggregation_factor),
            paths_by_od=self.paths_by_od,
            logit_scale=self.logit_scale,
            od_pairs=self.od_pairs,
            max_paths_per_od=self.max_paths_per_od,
            sample_route_choices=self.sample_route_choices,
            route_choice_sampling_unit=self.route_choice_sampling_unit,
            random_seed=self.random_seed,
        )
        if self.aggregation_factor <= 1:
            return internal_runtime
        return ExternalTimeStepDUORuntime(
            model=self,
            internal_runtime=internal_runtime,
            external_demand_horizon=int(demand_horizon),
        )

    def finalize_duo_runtime(self, runtime: ForwardDUOSimulator | ExternalTimeStepDUORuntime) -> AssignmentResult:
        if isinstance(runtime, ExternalTimeStepDUORuntime):
            simulation, _, path_shares, route_choice_costs = runtime.finalize()
            demand_horizon = runtime.demand_horizon
        else:
            simulation, _, path_shares, route_choice_costs = runtime.finalize()
            demand_horizon = runtime.demand_horizon
        return self._build_assignment_result(
            simulation=simulation,
            route_choice_costs=route_choice_costs,
            path_shares=path_shares,
            gap_history=(0.0,),
            iterations=1,
            demand_horizon=demand_horizon,
        )

    def _build_assignment_result(
        self,
        simulation,
        route_choice_costs: np.ndarray,
        path_shares: np.ndarray,
        gap_history: tuple[float, ...],
        iterations: int,
        demand_horizon: int,
    ) -> AssignmentResult:
        full_link_inflows = self._aggregate_sum_rows(simulation.full_link_inflows)
        full_link_outflows = self._aggregate_sum_rows(simulation.full_link_outflows)
        link_occupancies = self._aggregate_last_rows(simulation.link_occupancies, full_link_inflows.shape[0])
        link_travel_times = self._aggregate_last_rows(simulation.link_travel_times, full_link_inflows.shape[0])
        path_costs = self._aggregate_mean_rows(simulation.path_costs, demand_horizon)
        return AssignmentResult(
            link_inflows=full_link_inflows[:demand_horizon].copy(),
            full_link_inflows=full_link_inflows.copy(),
            full_link_outflows=full_link_outflows.copy(),
            link_occupancies=link_occupancies.copy(),
            link_travel_times=link_travel_times.copy(),
            temporal_link_inflows=simulation.temporal_link_inflows[:demand_horizon, :demand_horizon].copy(),
            path_costs=path_costs.copy(),
            route_choice_costs=route_choice_costs,
            path_shares=path_shares,
            gap_history=gap_history,
            iterations=iterations,
            od_pairs=tuple(self.od_pairs),
            path_labels=self.path_labels,
            link_labels=self.link_labels,
            route_choice_model=self.route_choice_model,
            logit_scale=self.logit_scale,
            sample_route_choices=self.sample_route_choices,
            route_choice_sampling_unit=self.route_choice_sampling_unit,
            random_seed=self.random_seed,
        )

