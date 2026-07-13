from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - runtime dependency check
    from numba import get_num_threads, njit, prange, set_num_threads
except Exception:  # pragma: no cover - fallback keeps the API working without numba.
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args):
        return range(*args)

    def set_num_threads(*args, **kwargs):
        return None

    def get_num_threads():
        return 1


def configure_numba_threads(num_threads: int | None) -> int | None:
    if num_threads is None:
        return int(get_num_threads())
    value = int(num_threads)
    if value <= 0:
        raise ValueError("numba thread count must be a positive integer.")
    set_num_threads(value)
    return int(get_num_threads())


def get_active_numba_threads() -> int:
    return int(get_num_threads())


@njit(cache=True)
def _akcelik_effective_delay(
    flow: float,
    capacity: float,
    alpha: float,
    j_parameter: float,
    period_steps: float,
    eps: float,
) -> float:
    if alpha <= 0.0 or j_parameter <= 0.0 or flow <= eps:
        return 0.0
    saturation = flow / max(capacity, eps)
    if saturation < 0.0:
        saturation = 0.0
    period_capacity = max(capacity * max(period_steps, eps), eps)
    z = saturation - 1.0
    radical = z * z + (8.0 * j_parameter * saturation / period_capacity)
    return alpha * period_steps * (z + math.sqrt(max(radical, 0.0)))


@njit(cache=True)
def _append_queue_record(
    queue_head: np.ndarray,
    queue_tail: np.ndarray,
    cohort_next: np.ndarray,
    cohort_path_id: np.ndarray,
    cohort_path_pos: np.ndarray,
    cohort_departure_time: np.ndarray,
    cohort_entry_time: np.ndarray,
    cohort_amount: np.ndarray,
    link_id: int,
    path_id: int,
    path_pos: int,
    departure_time: int,
    entry_time: int,
    amount: float,
    next_free: int,
    eps: float,
) -> tuple[int, int]:
    if amount <= eps:
        return next_free, 0

    tail_idx = queue_tail[link_id]
    if tail_idx != -1:
        if (
            cohort_path_id[tail_idx] == path_id
            and cohort_path_pos[tail_idx] == path_pos
            and cohort_departure_time[tail_idx] == departure_time
            and cohort_entry_time[tail_idx] == entry_time
        ):
            cohort_amount[tail_idx] += amount
            return next_free, 0

    idx = next_free
    cohort_next[idx] = -1
    cohort_path_id[idx] = path_id
    cohort_path_pos[idx] = path_pos
    cohort_departure_time[idx] = departure_time
    cohort_entry_time[idx] = entry_time
    cohort_amount[idx] = amount

    if tail_idx == -1:
        queue_head[link_id] = idx
        queue_tail[link_id] = idx
    else:
        cohort_next[tail_idx] = idx
        queue_tail[link_id] = idx
    return next_free + 1, 1


@njit(cache=True, parallel=True)
def snapshot_link_travel_times_kernel(
    cumulative_inflows: np.ndarray,
    cumulative_outflows: np.ndarray,
    free_flow_steps: np.ndarray,
    capacity: np.ndarray,
    t: int,
    akcelik_alpha: float,
    akcelik_j: float,
    akcelik_period_steps: float,
    eps: float,
) -> np.ndarray:
    num_links = free_flow_steps.shape[0]
    snapshot = np.empty(num_links, dtype=np.float64)
    for link_id in prange(num_links):
        ready_index = t + 1 - free_flow_steps[link_id]
        if ready_index < 0:
            ready_index = 0
        ready_volume = cumulative_inflows[ready_index, link_id]
        exited_volume = cumulative_outflows[t, link_id]
        queue_backlog = ready_volume - exited_volume
        if queue_backlog < 0.0:
            queue_backlog = 0.0
        recent_flow = 0.0
        if t > 0:
            recent_flow = cumulative_inflows[t, link_id] - cumulative_inflows[t - 1, link_id]
            if recent_flow < 0.0:
                recent_flow = 0.0
        free_flow = float(free_flow_steps[link_id])
        snapshot[link_id] = (
            free_flow
            + queue_backlog / max(capacity[link_id], eps)
            + _akcelik_effective_delay(
                recent_flow,
                capacity[link_id],
                akcelik_alpha,
                akcelik_j,
                akcelik_period_steps,
                eps,
            )
        )
    return snapshot


@njit(cache=True, parallel=True)
def snapshot_path_costs_kernel(
    snapshot_link_costs: np.ndarray,
    path_link_ids: np.ndarray,
    path_link_lengths: np.ndarray,
) -> np.ndarray:
    num_paths = path_link_lengths.shape[0]
    path_costs = np.zeros(num_paths, dtype=np.float64)
    for path_id in prange(num_paths):
        total_cost = 0.0
        path_len = path_link_lengths[path_id]
        for pos in range(path_len):
            total_cost += snapshot_link_costs[path_link_ids[path_id, pos]]
        path_costs[path_id] = total_cost
    return path_costs


@njit(cache=True, parallel=True)
def duo_logit_shares_row_kernel(
    od_row: np.ndarray,
    path_costs_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
    logit_scale: float,
    eps: float,
) -> np.ndarray:
    shares = np.zeros(path_costs_row.shape[0], dtype=np.float64)
    num_ods = od_path_offsets.shape[0] - 1
    for od_index in prange(num_ods):
        start = od_path_offsets[od_index]
        end = od_path_offsets[od_index + 1]
        count = end - start
        if count <= 0:
            continue

        demand = od_row[od_index]
        if demand <= eps:
            uniform = 1.0 / count
            for cursor in range(start, end):
                shares[od_path_ids[cursor]] = uniform
            continue

        max_utility = -1e300
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            utility = -logit_scale * path_costs_row[path_id]
            if utility > max_utility:
                max_utility = utility

        denominator = 0.0
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            exp_utility = math.exp(-logit_scale * path_costs_row[path_id] - max_utility)
            shares[path_id] = exp_utility
            denominator += exp_utility

        denominator = max(denominator, eps)
        inv_denominator = 1.0 / denominator
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            shares[path_id] *= inv_denominator
    return shares


@njit(cache=True, parallel=True)
def departures_from_share_row_kernel(
    od_row: np.ndarray,
    share_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
) -> np.ndarray:
    departures = np.zeros(share_row.shape[0], dtype=np.float64)
    num_ods = od_path_offsets.shape[0] - 1
    for od_index in prange(num_ods):
        start = od_path_offsets[od_index]
        end = od_path_offsets[od_index + 1]
        demand = od_row[od_index]
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            departures[path_id] = demand * share_row[path_id]
    return departures


@njit(cache=True, parallel=True)
def sampled_departures_from_share_row_kernel(
    od_row: np.ndarray,
    share_row: np.ndarray,
    od_path_offsets: np.ndarray,
    od_path_ids: np.ndarray,
    unit_offsets: np.ndarray,
    unit_draws: np.ndarray,
    remainder_draws: np.ndarray,
    sampling_unit: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_paths = share_row.shape[0]
    departures = np.zeros(num_paths, dtype=np.float64)
    realized_shares = np.zeros(num_paths, dtype=np.float64)
    num_ods = od_path_offsets.shape[0] - 1

    for od_index in prange(num_ods):
        start = od_path_offsets[od_index]
        end = od_path_offsets[od_index + 1]
        count = end - start
        if count <= 0:
            continue

        demand = od_row[od_index]
        if demand < 0.0:
            demand = 0.0

        total_probability = 0.0
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            probability = share_row[path_id]
            if probability > 0.0:
                total_probability += probability

        uniform_probability = 1.0 / count
        if demand <= eps:
            if total_probability <= eps:
                for cursor in range(start, end):
                    realized_shares[od_path_ids[cursor]] = uniform_probability
            else:
                inv_total = 1.0 / total_probability
                for cursor in range(start, end):
                    path_id = od_path_ids[cursor]
                    probability = share_row[path_id]
                    realized_shares[path_id] = probability * inv_total if probability > 0.0 else 0.0
            continue

        unit_start = unit_offsets[od_index]
        unit_end = unit_offsets[od_index + 1]
        for draw_index in range(unit_start, unit_end):
            draw = unit_draws[draw_index]
            cumulative = 0.0
            chosen_path = od_path_ids[end - 1]
            if total_probability <= eps:
                threshold = draw * count
                relative_index = int(threshold)
                if relative_index >= count:
                    relative_index = count - 1
                chosen_path = od_path_ids[start + relative_index]
            else:
                threshold = draw * total_probability
                for cursor in range(start, end):
                    path_id = od_path_ids[cursor]
                    probability = share_row[path_id]
                    if probability > 0.0:
                        cumulative += probability
                    if threshold <= cumulative:
                        chosen_path = path_id
                        break
            departures[chosen_path] += sampling_unit

        whole_volume = sampling_unit * float(unit_end - unit_start)
        remainder = demand - whole_volume
        if remainder > eps:
            draw = remainder_draws[od_index]
            cumulative = 0.0
            chosen_path = od_path_ids[end - 1]
            if total_probability <= eps:
                threshold = draw * count
                relative_index = int(threshold)
                if relative_index >= count:
                    relative_index = count - 1
                chosen_path = od_path_ids[start + relative_index]
            else:
                threshold = draw * total_probability
                for cursor in range(start, end):
                    path_id = od_path_ids[cursor]
                    probability = share_row[path_id]
                    if probability > 0.0:
                        cumulative += probability
                    if threshold <= cumulative:
                        chosen_path = path_id
                        break
            departures[chosen_path] += remainder

        inv_demand = 1.0 / demand
        for cursor in range(start, end):
            path_id = od_path_ids[cursor]
            realized_shares[path_id] = departures[path_id] * inv_demand

    return departures, realized_shares


@njit(cache=True, parallel=True)
def sending_kernel(
    cumulative_inflows: np.ndarray,
    cumulative_outflows: np.ndarray,
    free_flow_steps: np.ndarray,
    capacity: np.ndarray,
    t: int,
) -> np.ndarray:
    num_links = free_flow_steps.shape[0]
    sending = np.zeros(num_links, dtype=np.float64)
    current_outflows = cumulative_outflows[t]
    for link_id in prange(num_links):
        ready_index = t + 1 - free_flow_steps[link_id]
        ready_inflow = cumulative_inflows[ready_index, link_id] if ready_index > 0 else 0.0
        available = ready_inflow - current_outflows[link_id]
        if available < 0.0:
            available = 0.0
        sending[link_id] = min(capacity[link_id], available)
    return sending


@njit(cache=True, parallel=True)
def receiving_kernel(
    cumulative_inflows: np.ndarray,
    cumulative_outflows: np.ndarray,
    backward_wave_steps: np.ndarray,
    capacity: np.ndarray,
    jam_storage: np.ndarray,
    t: int,
) -> np.ndarray:
    num_links = backward_wave_steps.shape[0]
    receiving = np.zeros(num_links, dtype=np.float64)
    current_inflows = cumulative_inflows[t]
    for link_id in prange(num_links):
        lag_index = t + 1 - backward_wave_steps[link_id]
        if lag_index < 0:
            lag_index = 0
        remaining_storage = jam_storage[link_id] - (
            current_inflows[link_id] - cumulative_outflows[lag_index, link_id]
        )
        if remaining_storage < 0.0:
            remaining_storage = 0.0
        receiving[link_id] = min(capacity[link_id], remaining_storage)
    return receiving


@njit(cache=True, parallel=True)
def downstream_acceptance_kernel(
    receiving: np.ndarray,
    demand_by_downstream: np.ndarray,
    eps: float,
) -> np.ndarray:
    num_links = receiving.shape[0]
    accepted_ratio = np.ones(num_links, dtype=np.float64)
    for link_id in prange(num_links):
        demand = demand_by_downstream[link_id]
        if demand > eps:
            accepted_ratio[link_id] = min(receiving[link_id], demand) / demand
    return accepted_ratio


@njit(cache=True)
def count_source_loads_queue_kernel(
    receiving: np.ndarray,
    first_link_offsets: np.ndarray,
    first_link_path_ids: np.ndarray,
    source_buffer: np.ndarray,
    source_departure_buffer: np.ndarray,
    eps: float,
) -> int:
    added_slots = 0
    demand_horizon = source_departure_buffer.shape[0]
    num_links = first_link_offsets.shape[0] - 1
    for first_link in range(num_links):
        start = first_link_offsets[first_link]
        end = first_link_offsets[first_link + 1]
        if start == end:
            continue

        pending = 0.0
        for cursor in range(start, end):
            pending += source_buffer[first_link_path_ids[cursor]]
        if pending <= eps or receiving[first_link] <= eps:
            continue

        accepted_total = min(receiving[first_link], pending)
        scale = accepted_total / pending
        for cursor in range(start, end):
            path_id = first_link_path_ids[cursor]
            path_pending = source_buffer[path_id]
            if path_pending <= eps:
                continue
            remaining = path_pending * scale
            if remaining <= eps:
                continue

            for departure_time in range(demand_horizon):
                available = source_departure_buffer[departure_time, path_id]
                if available <= eps:
                    continue
                moved = min(available, remaining)
                if moved <= eps:
                    continue
                added_slots += 1
                remaining -= moved
                if remaining <= eps:
                    break
    return added_slots


@njit(cache=True)
def load_sources_queue_kernel(
    t: int,
    receiving: np.ndarray,
    first_link_offsets: np.ndarray,
    first_link_path_ids: np.ndarray,
    path_od_index: np.ndarray,
    source_buffer: np.ndarray,
    source_departure_buffer: np.ndarray,
    queue_head: np.ndarray,
    queue_tail: np.ndarray,
    cohort_next: np.ndarray,
    cohort_path_id: np.ndarray,
    cohort_path_pos: np.ndarray,
    cohort_departure_time: np.ndarray,
    cohort_entry_time: np.ndarray,
    cohort_amount: np.ndarray,
    link_inflows_t: np.ndarray,
    temporal_link_inflows: np.ndarray,
    temporal_aggregation_factor: int,
    temporal_horizon: int,
    temporal_current_offset: int,
    temporal_departure_offset: int,
    next_free: int,
    record_temporal: bool,
    eps: float,
) -> tuple[int, int]:
    added_nodes = 0
    demand_horizon = source_departure_buffer.shape[0]
    num_links = first_link_offsets.shape[0] - 1
    aggregation_factor = max(temporal_aggregation_factor, 1)
    current_index = (t // aggregation_factor) - temporal_current_offset

    for first_link in range(num_links):
        start = first_link_offsets[first_link]
        end = first_link_offsets[first_link + 1]
        if start == end:
            continue

        pending = 0.0
        for cursor in range(start, end):
            pending += source_buffer[first_link_path_ids[cursor]]
        if pending <= eps or receiving[first_link] <= eps:
            continue

        accepted_total = min(receiving[first_link], pending)
        scale = accepted_total / pending
        for cursor in range(start, end):
            path_id = first_link_path_ids[cursor]
            path_pending = source_buffer[path_id]
            if path_pending <= eps:
                continue
            remaining = path_pending * scale
            if remaining <= eps:
                continue

            for departure_time in range(demand_horizon):
                available = source_departure_buffer[departure_time, path_id]
                if available <= eps:
                    continue
                moved = min(available, remaining)
                if moved <= eps:
                    continue

                source_departure_buffer[departure_time, path_id] -= moved
                source_buffer[path_id] -= moved
                next_free, added = _append_queue_record(
                    queue_head=queue_head,
                    queue_tail=queue_tail,
                    cohort_next=cohort_next,
                    cohort_path_id=cohort_path_id,
                    cohort_path_pos=cohort_path_pos,
                    cohort_departure_time=cohort_departure_time,
                    cohort_entry_time=cohort_entry_time,
                    cohort_amount=cohort_amount,
                    link_id=first_link,
                    path_id=path_id,
                    path_pos=0,
                    departure_time=departure_time,
                    entry_time=t,
                    amount=moved,
                    next_free=next_free,
                    eps=eps,
                )
                added_nodes += added
                link_inflows_t[first_link] += moved

                if record_temporal:
                    departure_index = (departure_time // aggregation_factor) - temporal_departure_offset
                    if (
                        current_index >= 0
                        and current_index < temporal_horizon
                        and departure_index >= 0
                        and departure_index < temporal_horizon
                    ):
                        od_index = path_od_index[path_id]
                        temporal_link_inflows[
                            current_index,
                            departure_index,
                            od_index,
                            first_link,
                        ] += moved

                remaining -= moved
                if remaining <= eps:
                    break

            if source_buffer[path_id] < eps:
                source_buffer[path_id] = 0.0

        receiving[first_link] -= accepted_total

    return next_free, added_nodes


@njit(cache=True)
def accumulate_downstream_demand_queue_kernel(
    t: int,
    sending: np.ndarray,
    queue_head: np.ndarray,
    cohort_next: np.ndarray,
    cohort_path_id: np.ndarray,
    cohort_path_pos: np.ndarray,
    cohort_entry_time: np.ndarray,
    cohort_amount: np.ndarray,
    free_flow_steps: np.ndarray,
    path_next_links: np.ndarray,
    eps: float,
) -> np.ndarray:
    num_links = queue_head.shape[0]
    demand_by_downstream = np.zeros(num_links, dtype=np.float64)
    for link_id in range(num_links):
        remaining_sending = sending[link_id]
        if remaining_sending <= eps:
            continue

        free_flow = free_flow_steps[link_id]
        idx = queue_head[link_id]
        while idx != -1:
            if remaining_sending <= eps:
                break
            if t - cohort_entry_time[idx] < free_flow:
                break

            candidate_amount = min(cohort_amount[idx], remaining_sending)
            next_link = path_next_links[cohort_path_id[idx], cohort_path_pos[idx]]
            if next_link >= 0:
                demand_by_downstream[next_link] += candidate_amount
            remaining_sending -= candidate_amount
            idx = cohort_next[idx]

    return demand_by_downstream


@njit(cache=True)
def apply_moves_queue_kernel(
    t: int,
    sending: np.ndarray,
    accepted_ratio: np.ndarray,
    queue_head: np.ndarray,
    cohort_next: np.ndarray,
    cohort_path_id: np.ndarray,
    cohort_path_pos: np.ndarray,
    cohort_departure_time: np.ndarray,
    cohort_entry_time: np.ndarray,
    cohort_amount: np.ndarray,
    free_flow_steps: np.ndarray,
    path_next_links: np.ndarray,
    path_od_index: np.ndarray,
    pending_link: np.ndarray,
    pending_path_id: np.ndarray,
    pending_path_pos: np.ndarray,
    pending_departure_time: np.ndarray,
    pending_entry_time: np.ndarray,
    pending_amount: np.ndarray,
    link_inflows_t: np.ndarray,
    link_outflows_t: np.ndarray,
    arrived_volume: np.ndarray,
    temporal_link_inflows: np.ndarray,
    temporal_aggregation_factor: int,
    temporal_horizon: int,
    temporal_current_offset: int,
    temporal_departure_offset: int,
    record_temporal: bool,
    demand_horizon: int,
    eps: float,
) -> int:
    pending_count = 0
    num_links = queue_head.shape[0]
    aggregation_factor = max(temporal_aggregation_factor, 1)
    current_index = (t // aggregation_factor) - temporal_current_offset
    for link_id in range(num_links):
        remaining_sending = sending[link_id]
        if remaining_sending <= eps:
            continue

        free_flow = free_flow_steps[link_id]
        idx = queue_head[link_id]
        while idx != -1:
            if remaining_sending <= eps:
                break
            if t - cohort_entry_time[idx] < free_flow:
                break

            candidate_amount = min(cohort_amount[idx], remaining_sending)
            remaining_sending -= candidate_amount
            next_link = path_next_links[cohort_path_id[idx], cohort_path_pos[idx]]
            moved = candidate_amount
            if next_link >= 0:
                moved *= accepted_ratio[next_link]
            if moved > eps:
                cohort_amount[idx] -= moved
                link_outflows_t[link_id] += moved
                if next_link < 0:
                    departure_time = cohort_departure_time[idx]
                    if departure_time < demand_horizon:
                        arrived_volume[departure_time, cohort_path_id[idx]] += moved
                else:
                    departure_time = cohort_departure_time[idx]
                    pending_link[pending_count] = next_link
                    pending_path_id[pending_count] = cohort_path_id[idx]
                    pending_path_pos[pending_count] = cohort_path_pos[idx] + 1
                    pending_departure_time[pending_count] = departure_time
                    pending_entry_time[pending_count] = t
                    pending_amount[pending_count] = moved
                    pending_count += 1
                    if record_temporal:
                        departure_index = (departure_time // aggregation_factor) - temporal_departure_offset
                        if (
                            departure_time >= 0
                            and departure_time < demand_horizon
                            and current_index >= 0
                            and current_index < temporal_horizon
                            and departure_index >= 0
                            and departure_index < temporal_horizon
                        ):
                            od_index = path_od_index[cohort_path_id[idx]]
                            temporal_link_inflows[
                                current_index,
                                departure_index,
                                od_index,
                                next_link,
                            ] += moved
            idx = cohort_next[idx]
    return pending_count


@njit(cache=True)
def prune_empty_heads_queue_kernel(
    queue_head: np.ndarray,
    queue_tail: np.ndarray,
    cohort_next: np.ndarray,
    cohort_amount: np.ndarray,
    eps: float,
) -> int:
    removed = 0
    num_links = queue_head.shape[0]
    for link_id in range(num_links):
        head = queue_head[link_id]
        while head != -1 and cohort_amount[head] <= eps:
            head = cohort_next[head]
            removed += 1
        queue_head[link_id] = head
        if head == -1:
            queue_tail[link_id] = -1
    return removed


@njit(cache=True)
def merge_pending_queue_kernel(
    pending_count: int,
    pending_link: np.ndarray,
    pending_path_id: np.ndarray,
    pending_path_pos: np.ndarray,
    pending_departure_time: np.ndarray,
    pending_entry_time: np.ndarray,
    pending_amount: np.ndarray,
    queue_head: np.ndarray,
    queue_tail: np.ndarray,
    cohort_next: np.ndarray,
    cohort_path_id: np.ndarray,
    cohort_path_pos: np.ndarray,
    cohort_departure_time: np.ndarray,
    cohort_entry_time: np.ndarray,
    cohort_amount: np.ndarray,
    link_inflows_t: np.ndarray,
    next_free: int,
    eps: float,
) -> tuple[int, int]:
    added_nodes = 0
    for index in range(pending_count):
        moved = pending_amount[index]
        if moved <= eps:
            continue
        link_id = pending_link[index]
        next_free, added = _append_queue_record(
            queue_head=queue_head,
            queue_tail=queue_tail,
            cohort_next=cohort_next,
            cohort_path_id=cohort_path_id,
            cohort_path_pos=cohort_path_pos,
            cohort_departure_time=cohort_departure_time,
            cohort_entry_time=cohort_entry_time,
            cohort_amount=cohort_amount,
            link_id=link_id,
            path_id=pending_path_id[index],
            path_pos=pending_path_pos[index],
            departure_time=pending_departure_time[index],
            entry_time=pending_entry_time[index],
            amount=moved,
            next_free=next_free,
            eps=eps,
        )
        added_nodes += added
        link_inflows_t[link_id] += moved
    return next_free, added_nodes


@njit(cache=True)
def is_network_empty_queue_kernel(
    queue_head: np.ndarray,
    source_buffer: np.ndarray,
    eps: float,
) -> bool:
    for idx in range(source_buffer.shape[0]):
        if source_buffer[idx] > eps:
            return False
    for link_id in range(queue_head.shape[0]):
        if queue_head[link_id] != -1:
            return False
    return True


@njit(cache=True, parallel=True)
def estimate_link_travel_times_kernel(
    cumulative_inflows: np.ndarray,
    cumulative_outflows: np.ndarray,
    free_flow_steps: np.ndarray,
    capacity: np.ndarray,
    actual_steps: int,
    akcelik_alpha: float,
    akcelik_j: float,
    akcelik_period_steps: float,
    eps: float,
) -> np.ndarray:
    num_links = free_flow_steps.shape[0]
    link_travel_times = np.zeros((actual_steps, num_links), dtype=np.float64)
    for link_id in prange(num_links):
        free_flow = free_flow_steps[link_id]
        for t in range(actual_steps):
            probe_rank = cumulative_inflows[t + 1, link_id]
            earliest_exit = t + free_flow
            if earliest_exit > actual_steps - 1:
                earliest_exit = actual_steps - 1
            exit_step = earliest_exit
            while (
                exit_step < actual_steps - 1
                and cumulative_outflows[exit_step + 1, link_id] + eps < probe_rank
            ):
                exit_step += 1
            travel_time = float(free_flow)
            queue_delay = float(exit_step - t)
            if queue_delay > travel_time:
                travel_time = queue_delay
            entry_flow = cumulative_inflows[t + 1, link_id] - cumulative_inflows[t, link_id]
            if entry_flow < 0.0:
                entry_flow = 0.0
            link_travel_times[t, link_id] = travel_time + _akcelik_effective_delay(
                entry_flow,
                capacity[link_id],
                akcelik_alpha,
                akcelik_j,
                akcelik_period_steps,
                eps,
            )
    return link_travel_times


@njit(cache=True, parallel=True)
def estimate_path_costs_kernel(
    link_travel_times: np.ndarray,
    path_link_ids: np.ndarray,
    path_link_lengths: np.ndarray,
    demand_horizon: int,
) -> np.ndarray:
    num_paths = path_link_lengths.shape[0]
    path_costs = np.zeros((demand_horizon, num_paths), dtype=np.float64)
    final_row = link_travel_times.shape[0] - 1
    for path_id in prange(num_paths):
        path_len = path_link_lengths[path_id]
        for departure_time in range(demand_horizon):
            current_time = departure_time
            total_cost = 0.0
            for pos in range(path_len):
                row = current_time
                if row > final_row:
                    row = final_row
                link_id = path_link_ids[path_id, pos]
                travel_time = link_travel_times[row, link_id]
                total_cost += travel_time
                current_time += int(math.ceil(travel_time))
            path_costs[departure_time, path_id] = total_cost
    return path_costs
