from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ltm import (
    LinkTransmissionModel,
    SimulationResult,
    _build_group_index,
    _realized_shares_from_departures,
    _sample_departures_from_share_row,
)
from .paths import Path

EPS = 1e-9
VALID_ROUTE_CHOICE_MODES = {"due", "duo"}


@dataclass(frozen=True)
class DUESolveResult:
    path_shares: np.ndarray
    path_departures: np.ndarray
    simulation: SimulationResult
    route_choice_costs: np.ndarray
    gap_history: tuple[float, ...]
    iterations: int


class StochasticRouteChoiceSolver:
    def __init__(
        self,
        paths: list[Path],
        paths_by_od: list[list[int]],
        loader: LinkTransmissionModel,
        max_iterations: int = 30,
        tolerance: float = 1e-3,
        min_iterations: int = 3,
        logit_scale: float = 0.35,
        route_choice_mode: str = "due",
        sample_route_choices: bool = False,
        route_choice_sampling_unit: float = 1.0,
        random_seed: int | None = None,
    ) -> None:
        if logit_scale <= EPS:
            raise ValueError("logit_scale must be positive for stochastic route choice.")
        if route_choice_mode not in VALID_ROUTE_CHOICE_MODES:
            raise ValueError(
                f"route_choice_mode must be one of {sorted(VALID_ROUTE_CHOICE_MODES)}, "
                f"but received {route_choice_mode!r}."
            )
        if sample_route_choices and route_choice_sampling_unit <= EPS:
            raise ValueError(
                "route_choice_sampling_unit must be positive when stochastic route-choice "
                "sampling is enabled."
            )
        self.paths = paths
        self.paths_by_od = paths_by_od
        self.od_path_offsets, self.od_path_ids = _build_group_index(paths_by_od)
        self.loader = loader
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_iterations = min_iterations
        self.logit_scale = float(logit_scale)
        self.route_choice_mode = route_choice_mode
        self.sample_route_choices = bool(sample_route_choices)
        self.route_choice_sampling_unit = float(route_choice_sampling_unit)
        self.set_random_seed(random_seed)

    def set_random_seed(self, random_seed: int | None) -> None:
        self.random_seed = None if random_seed is None else int(random_seed)
        self.rng = np.random.default_rng(self.random_seed) if self.sample_route_choices else None

    def solve(self, od_matrix: np.ndarray) -> DUESolveResult:
        od_matrix = np.asarray(od_matrix, dtype=float)
        if od_matrix.ndim != 2:
            raise ValueError("od_matrix must be a 2D array shaped as [time, od_pair].")
        if np.any(od_matrix < -EPS):
            raise ValueError("od_matrix cannot contain negative demand.")
        if od_matrix.shape[1] != len(self.paths_by_od):
            raise ValueError("The number of OD columns does not match the configured OD pairs.")
        if self.sample_route_choices:
            self.rng = np.random.default_rng(self.random_seed)
        if self.route_choice_mode == "duo":
            return self._solve_duo(od_matrix)

        shares = self._initial_shares(od_matrix.shape[0])
        gap_history: list[float] = []
        simulation: SimulationResult | None = None
        route_choice_costs = np.zeros((od_matrix.shape[0], len(self.paths)), dtype=float)
        path_departures = self._path_departures_from_shares(od_matrix, shares)
        reported_shares = shares.copy()
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            path_departures = self._path_departures_from_shares(od_matrix, shares)
            simulation = self.loader.simulate(path_departures)
            route_choice_costs = self._route_choice_costs(
                simulation=simulation,
                demand_horizon=od_matrix.shape[0],
            )

            target_shares = self._logit_target_shares(
                od_matrix=od_matrix,
                path_costs=route_choice_costs,
                current_shares=shares,
            )
            gap = self._fixed_point_gap(
                od_matrix=od_matrix,
                shares=shares,
                target_shares=target_shares,
            )
            gap_history.append(gap)

            if iteration >= self.min_iterations and gap <= self.tolerance:
                shares = target_shares
                converged = True
                break

            step_size = 1.0 / iteration
            shares = (1.0 - step_size) * shares + step_size * target_shares

        if simulation is None:
            raise RuntimeError("Route-choice solver failed to run the network loading step.")

        if converged or gap_history:
            path_departures = self._path_departures_from_shares(od_matrix, shares)
            simulation = self.loader.simulate(path_departures)
            route_choice_costs = self._route_choice_costs(
                simulation=simulation,
                demand_horizon=od_matrix.shape[0],
            )
            reported_shares = self._reported_shares_from_departures(
                od_matrix=od_matrix,
                expected_shares=shares,
                path_departures=path_departures,
            )

        return DUESolveResult(
            path_shares=reported_shares,
            path_departures=path_departures,
            simulation=simulation,
            route_choice_costs=route_choice_costs,
            gap_history=tuple(gap_history),
            iterations=len(gap_history),
        )

    def _solve_duo(self, od_matrix: np.ndarray) -> DUESolveResult:
        simulation, path_departures, path_shares, route_choice_costs = self.loader.simulate_duo(
            od_matrix=od_matrix,
            paths_by_od=self.paths_by_od,
            logit_scale=self.logit_scale,
            sample_route_choices=self.sample_route_choices,
            route_choice_sampling_unit=self.route_choice_sampling_unit,
            random_seed=self.random_seed,
        )
        return DUESolveResult(
            path_shares=path_shares,
            path_departures=path_departures,
            simulation=simulation,
            route_choice_costs=route_choice_costs,
            gap_history=(0.0,),
            iterations=1,
        )

    def _initial_shares(self, num_time_steps: int) -> np.ndarray:
        shares = np.zeros((num_time_steps, len(self.paths)), dtype=float)
        for path_ids in self.paths_by_od:
            shares[:, path_ids] = 1.0 / len(path_ids)
        return shares

    def _path_departures_from_shares(self, od_matrix: np.ndarray, shares: np.ndarray) -> np.ndarray:
        path_departures = np.zeros((od_matrix.shape[0], len(self.paths)), dtype=float)
        if self.sample_route_choices:
            assert self.rng is not None
            for time_index in range(od_matrix.shape[0]):
                path_departures[time_index] = _sample_departures_from_share_row(
                    od_row=od_matrix[time_index],
                    share_row=shares[time_index],
                    od_path_offsets=self.od_path_offsets,
                    od_path_ids=self.od_path_ids,
                    rng=self.rng,
                    sampling_unit=self.route_choice_sampling_unit,
                )
            return path_departures
        for od_index, path_ids in enumerate(self.paths_by_od):
            path_departures[:, path_ids] = od_matrix[:, [od_index]] * shares[:, path_ids]
        return path_departures

    def _reported_shares_from_departures(
        self,
        od_matrix: np.ndarray,
        expected_shares: np.ndarray,
        path_departures: np.ndarray,
    ) -> np.ndarray:
        if not self.sample_route_choices:
            return expected_shares

        reported_shares = np.zeros_like(expected_shares)
        for time_index in range(od_matrix.shape[0]):
            reported_shares[time_index] = _realized_shares_from_departures(
                od_row=od_matrix[time_index],
                departure_row=path_departures[time_index],
                od_path_offsets=self.od_path_offsets,
                od_path_ids=self.od_path_ids,
                fallback_share_row=expected_shares[time_index],
            )
        return reported_shares

    def _route_choice_costs(
        self,
        simulation: SimulationResult,
        demand_horizon: int,
    ) -> np.ndarray:
        if self.route_choice_mode == "due":
            return simulation.path_costs.copy()
        return self._instantaneous_path_costs(
            link_travel_times=simulation.link_travel_times,
            demand_horizon=demand_horizon,
        )

    def _instantaneous_path_costs(
        self,
        link_travel_times: np.ndarray,
        demand_horizon: int,
    ) -> np.ndarray:
        path_costs = np.zeros((demand_horizon, len(self.paths)), dtype=float)
        final_row = link_travel_times.shape[0] - 1
        for path in self.paths:
            for departure_time in range(demand_horizon):
                row = min(departure_time, final_row)
                path_costs[departure_time, path.path_id] = float(
                    np.sum(link_travel_times[row, list(path.links)])
                )
        return path_costs

    def _logit_target_shares(
        self,
        od_matrix: np.ndarray,
        path_costs: np.ndarray,
        current_shares: np.ndarray,
    ) -> np.ndarray:
        target = np.zeros_like(current_shares)
        for od_index, path_ids in enumerate(self.paths_by_od):
            od_demand = od_matrix[:, od_index]
            costs = path_costs[:, path_ids]
            logit_utility = -self.logit_scale * costs
            logit_utility -= np.max(logit_utility, axis=1, keepdims=True)
            exp_utility = np.exp(logit_utility)
            target[:, path_ids] = exp_utility / np.maximum(
                np.sum(exp_utility, axis=1, keepdims=True),
                EPS,
            )

            zero_demand_rows = np.where(od_demand <= EPS)[0]
            if zero_demand_rows.size:
                target[np.ix_(zero_demand_rows, path_ids)] = current_shares[np.ix_(zero_demand_rows, path_ids)]

        return target

    def _fixed_point_gap(
        self,
        od_matrix: np.ndarray,
        shares: np.ndarray,
        target_shares: np.ndarray,
    ) -> float:
        numerator = 0.0
        denominator = 0.0
        for od_index, path_ids in enumerate(self.paths_by_od):
            od_demand = od_matrix[:, od_index]
            if np.all(od_demand <= EPS):
                continue

            share_shift = 0.5 * np.sum(
                np.abs(target_shares[:, path_ids] - shares[:, path_ids]),
                axis=1,
            )
            numerator += float(np.sum(od_demand * share_shift))
            denominator += float(np.sum(od_demand))

        return numerator / max(denominator, EPS)


StochasticDynamicUserEquilibriumSolver = StochasticRouteChoiceSolver
DynamicUserEquilibriumSolver = StochasticRouteChoiceSolver
