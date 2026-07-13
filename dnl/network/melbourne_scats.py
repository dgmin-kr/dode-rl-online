from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .structures import Link, Network


PROJECT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = Path(__file__).resolve().parent / "melbourne_scats_metadata.json"

DEFAULT_TIME_STEP_MINUTES = 15.0
BACKWARD_WAVE_SPEED_KPH = 24.0
JAM_DENSITY_PER_LANE_PER_KM = 110.0
DEFAULT_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION = 1
HIGH_CAPACITY_PRIMARY_DIRECTED_LINK_IDS = frozenset({15, 60})
HIGH_CAPACITY_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION = 2
# VITM/Akcelik Table 1 "Arterial urban sealed" setting.
VITM_AKCELIK_LINK_TYPE = "Arterial urban sealed"
VITM_CAPACITY_PER_LANE_PER_HOUR = 900.0
VITM_POSTED_SPEED_FACTOR = 0.75
VITM_AKCELIK_J = 0.8
VITM_AKCELIK_ALPHA = 0.25
DEFAULT_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR = (
    DEFAULT_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION * VITM_CAPACITY_PER_LANE_PER_HOUR
)
HIGH_CAPACITY_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR = (
    HIGH_CAPACITY_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION * VITM_CAPACITY_PER_LANE_PER_HOUR
)

CAPACITY_PER_LANE_PER_HOUR = {
    "primary": VITM_CAPACITY_PER_LANE_PER_HOUR,
    "primary_link": VITM_CAPACITY_PER_LANE_PER_HOUR,
}

DEFAULT_SPEED_KPH = {
    "primary": 50.0,
    "primary_link": 35.0,
}


@dataclass(frozen=True)
class _DirectedLinkSpec:
    start: int
    end: int
    length_km: float
    highway: str
    ref: str
    name: str
    lanes: int
    posted_speed_kph: float
    speed_kph: float
    capacity_per_hour: float
    source_link_id: str
    source_direction: str
    source_direction_capacity_per_hour: float
    is_virtual_reverse: bool = False
    capacity_adjustment: str = ""
    capacity_basis_link_type: str = VITM_AKCELIK_LINK_TYPE
    capacity_per_lane_per_hour: float = VITM_CAPACITY_PER_LANE_PER_HOUR


def _require_metadata(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Melbourne SCATS topology metadata: {path}. "
            "The public release should include this checked runtime topology file. "
            "Regenerate it from the private raw-source pipeline before running this network."
        )


@lru_cache(maxsize=1)
def _load_topology_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_metadata(METADATA_PATH)
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if str(payload.get("network_name", "")) != "melbourne_scats":
        raise ValueError(f"{METADATA_PATH} is not a Melbourne SCATS metadata file.")
    tables = payload.get("tables", {})
    nodes = pd.DataFrame(tables.get("nodes", []))
    links = pd.DataFrame(tables.get("links", []))
    od_zones = pd.DataFrame(tables.get("od_zones", []))
    required_columns = {
        "nodes": (nodes, {"node_id", "lon", "lat"}),
        "links": (
            links,
            {
                "link_id",
                "from_node",
                "to_node",
                "highway",
                "length_km",
                "speed_kph",
                "capacity_per_hour",
                "forward_capacity_per_hour",
                "reverse_capacity_per_hour",
                "capacity_cap_per_hour",
            },
        ),
        "od_zones": (od_zones, {"od_zone_id", "node_id"}),
    }
    for table_name, (frame, columns) in required_columns.items():
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{METADATA_PATH} table {table_name!r} is missing required columns: {sorted(missing)}"
            )
    return nodes, links, od_zones


def get_melbourne_scats_topology_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes, links, od_zones = _load_topology_tables()
    return nodes.copy(), links.copy(), od_zones.copy()


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _lane_capacity_per_hour(highway: str) -> float:
    return float(CAPACITY_PER_LANE_PER_HOUR.get(str(highway), 800.0))


def _uses_uniform_primary_capacity(highway: str) -> bool:
    return str(highway) == "primary"


def _effective_primary_lanes_for_directed_link(directed_link_id: int) -> int:
    if int(directed_link_id) in HIGH_CAPACITY_PRIMARY_DIRECTED_LINK_IDS:
        return int(HIGH_CAPACITY_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION)
    return int(DEFAULT_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION)


def _primary_capacity_per_hour_for_directed_link(directed_link_id: int) -> float:
    return float(_effective_primary_lanes_for_directed_link(directed_link_id) * VITM_CAPACITY_PER_LANE_PER_HOUR)


def _source_direction(reverse: bool) -> str:
    return "reverse" if reverse else "forward"


def _vitm_effective_speed_kph(highway: str, posted_speed_kph: float) -> float:
    speed = max(float(posted_speed_kph), 5.0)
    if str(highway) == "primary":
        return max(speed * float(VITM_POSTED_SPEED_FACTOR), 5.0)
    return speed


def _infer_lanes_from_capacity(capacity_per_hour: float, highway: str) -> int:
    lane_capacity = max(_lane_capacity_per_hour(highway), 1.0)
    return max(1, int(math.ceil(max(float(capacity_per_hour), 1.0) / lane_capacity)))


def _directional_capacity_per_hour(row: Any, *, reverse: bool, directed_link_id: int) -> tuple[float, bool]:
    representative = max(float(getattr(row, "capacity_per_hour", 0.0)), 1.0)
    raw_forward = max(float(getattr(row, "forward_capacity_per_hour", 0.0)), 0.0)
    raw_reverse = max(float(getattr(row, "reverse_capacity_per_hour", 0.0)), 0.0)
    raw_directional = raw_reverse if reverse else raw_forward

    if _uses_uniform_primary_capacity(str(getattr(row, "highway", ""))):
        return float(_primary_capacity_per_hour_for_directed_link(directed_link_id)), bool(
            reverse and raw_directional <= 0.0
        )

    # The simplified graph is used as an OD estimation topology, not as a
    # lane-accurate one-way graph. If OSM has capacity only in one direction,
    # mirror the representative capacity so all OD pairs remain feasible.
    if raw_directional > 0.0:
        capacity = raw_directional
        is_virtual_reverse = False
    else:
        capacity = representative
        is_virtual_reverse = bool(reverse)

    capacity_cap = max(float(getattr(row, "capacity_cap_per_hour", capacity)), 1.0)
    return min(max(capacity, 1.0), max(capacity_cap, 1.0)), is_virtual_reverse


@lru_cache(maxsize=1)
def _build_directed_link_specs() -> tuple[
    tuple[int, ...],
    dict[int, tuple[float, float]],
    tuple[int, ...],
    tuple[_DirectedLinkSpec, ...],
    dict[str, Any],
]:
    nodes_df, links_df, od_zones_df = _load_topology_tables()

    nodes = tuple(sorted(int(row.node_id) for row in nodes_df.itertuples(index=False)))
    node_positions = {
        int(row.node_id): (float(row.lon), float(row.lat))
        for row in nodes_df.itertuples(index=False)
    }
    centroid_nodes = tuple(
        int(row.node_id)
        for row in od_zones_df.sort_values("od_zone_id").itertuples(index=False)
    )

    directed_specs: list[_DirectedLinkSpec] = []
    mirrored_direction_count = 0
    for row in links_df.itertuples(index=False):
        start = int(row.from_node)
        end = int(row.to_node)
        highway = str(row.highway)
        source_link_id = str(row.link_id)
        length_km = max(float(row.length_km), 0.03)
        posted_speed_kph = max(float(getattr(row, "speed_kph", DEFAULT_SPEED_KPH.get(highway, 50.0))), 5.0)
        speed_kph = _vitm_effective_speed_kph(highway, posted_speed_kph)

        for reverse in (False, True):
            directed_link_id = len(directed_specs)
            source_direction = _source_direction(reverse)
            capacity_per_hour, is_virtual_reverse = _directional_capacity_per_hour(
                row,
                reverse=reverse,
                directed_link_id=directed_link_id,
            )
            if is_virtual_reverse:
                mirrored_direction_count += 1
            lanes = (
                _effective_primary_lanes_for_directed_link(directed_link_id)
                if _uses_uniform_primary_capacity(highway)
                else _infer_lanes_from_capacity(capacity_per_hour, highway)
            )
            capacity_adjustment = (
                (
                    "vitm_akcelik_arterial_urban_sealed_high_capacity_exception"
                    if directed_link_id in HIGH_CAPACITY_PRIMARY_DIRECTED_LINK_IDS
                    else "vitm_akcelik_arterial_urban_sealed_effective_one_lane"
                )
                if _uses_uniform_primary_capacity(highway)
                else ""
            )
            capacity_basis_link_type = (
                str(VITM_AKCELIK_LINK_TYPE) if _uses_uniform_primary_capacity(highway) else str(highway)
            )
            capacity_per_lane_per_hour = (
                float(VITM_CAPACITY_PER_LANE_PER_HOUR)
                if _uses_uniform_primary_capacity(highway)
                else _lane_capacity_per_hour(highway)
            )
            directed_specs.append(
                _DirectedLinkSpec(
                    start=end if reverse else start,
                    end=start if reverse else end,
                    length_km=length_km,
                    highway=highway,
                    ref=_clean_text(getattr(row, "ref", "")),
                    name=_clean_text(getattr(row, "name", "")),
                    lanes=lanes,
                    posted_speed_kph=posted_speed_kph,
                    speed_kph=speed_kph,
                    capacity_per_hour=capacity_per_hour,
                    source_link_id=source_link_id,
                    source_direction=source_direction,
                    source_direction_capacity_per_hour=(
                        float(getattr(row, "reverse_capacity_per_hour", 0.0))
                        if reverse
                        else float(getattr(row, "forward_capacity_per_hour", 0.0))
                    ),
                    is_virtual_reverse=is_virtual_reverse,
                    capacity_adjustment=capacity_adjustment,
                    capacity_basis_link_type=capacity_basis_link_type,
                    capacity_per_lane_per_hour=capacity_per_lane_per_hour,
                )
            )

    if not directed_specs:
        raise ValueError(f"No Melbourne SCATS links were parsed from {METADATA_PATH}.")

    metadata = {
        "source_nodes": int(len(nodes_df)),
        "source_undirected_links": int(len(links_df)),
        "directed_links": int(len(directed_specs)),
        "od_centroids": int(len(centroid_nodes)),
        "od_pairs": int(len(centroid_nodes) * max(len(centroid_nodes) - 1, 0)),
        "time_step_minutes": float(DEFAULT_TIME_STEP_MINUTES),
        "mirrored_direction_links": int(mirrored_direction_count),
        "capacity_basis": "VITM Akcelik arterial urban sealed capacity",
        "vitm_akcelik_link_type": str(VITM_AKCELIK_LINK_TYPE),
        "vitm_capacity_per_lane_per_hour": float(VITM_CAPACITY_PER_LANE_PER_HOUR),
        "vitm_posted_speed_factor": float(VITM_POSTED_SPEED_FACTOR),
        "vitm_akcelik_j": float(VITM_AKCELIK_J),
        "vitm_akcelik_alpha": float(VITM_AKCELIK_ALPHA),
        "default_primary_arterial_effective_lanes_per_direction": int(
            DEFAULT_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION
        ),
        "default_primary_arterial_capacity_per_hour": float(DEFAULT_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR),
        "default_primary_arterial_capacity_per_15min": float(
            DEFAULT_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR * DEFAULT_TIME_STEP_MINUTES / 60.0
        ),
        "high_capacity_primary_directed_link_ids": tuple(sorted(HIGH_CAPACITY_PRIMARY_DIRECTED_LINK_IDS)),
        "high_capacity_primary_arterial_effective_lanes_per_direction": int(
            HIGH_CAPACITY_PRIMARY_ARTERIAL_EFFECTIVE_LANES_PER_DIRECTION
        ),
        "high_capacity_primary_arterial_capacity_per_hour": float(
            HIGH_CAPACITY_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR
        ),
        "high_capacity_primary_arterial_capacity_per_15min": float(
            HIGH_CAPACITY_PRIMARY_ARTERIAL_CAPACITY_PER_HOUR * DEFAULT_TIME_STEP_MINUTES / 60.0
        ),
    }
    return nodes, node_positions, centroid_nodes, tuple(directed_specs), metadata


def _make_link(link_id: int, spec: _DirectedLinkSpec, time_step_minutes: float) -> Link:
    free_flow_minutes = 60.0 * max(spec.length_km, 0.03) / max(float(spec.speed_kph), 1.0)
    backward_wave_minutes = 60.0 * max(spec.length_km, 0.03) / BACKWARD_WAVE_SPEED_KPH
    free_flow_steps = max(1, int(math.ceil(free_flow_minutes / float(time_step_minutes))))
    backward_wave_steps = max(1, int(math.ceil(backward_wave_minutes / float(time_step_minutes))))
    step_capacity = max(1.0, float(spec.capacity_per_hour) * float(time_step_minutes) / 60.0)
    jam_storage = max(
        step_capacity * max(2.5, backward_wave_steps + 1.0),
        JAM_DENSITY_PER_LANE_PER_KM * max(spec.lanes, 1) * max(spec.length_km, 0.03),
        20.0,
    )
    return Link(
        link_id=link_id,
        start=int(spec.start),
        end=int(spec.end),
        free_flow_steps=free_flow_steps,
        backward_wave_steps=backward_wave_steps,
        capacity=step_capacity,
        jam_storage=jam_storage,
    )


def build_melbourne_scats_network(
    time_step_minutes: float = DEFAULT_TIME_STEP_MINUTES,
    od_pairs: tuple[tuple[int, int], ...] | None = None,
) -> tuple[Network, list[tuple[int, int]]]:
    nodes, node_positions, centroid_nodes, directed_specs, _ = _build_directed_link_specs()
    links: list[Link] = []
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)

    for spec in directed_specs:
        link_id = len(links)
        link = _make_link(link_id, spec, time_step_minutes)
        links.append(link)
        outgoing[link.start].append(link_id)
        incoming[link.end].append(link_id)

    network = Network(
        nodes=nodes,
        links=tuple(links),
        outgoing_by_node={node: tuple(link_ids) for node, link_ids in outgoing.items()},
        incoming_by_node={node: tuple(link_ids) for node, link_ids in incoming.items()},
        node_positions=dict(node_positions),
        centroid_nodes=centroid_nodes,
    )

    chosen_od_pairs = list(get_default_melbourne_scats_od_pairs() if od_pairs is None else od_pairs)
    return network, chosen_od_pairs


@lru_cache(maxsize=1)
def get_melbourne_scats_zone_centroids() -> tuple[int, ...]:
    _, _, centroid_nodes, _, _ = _build_directed_link_specs()
    return centroid_nodes


@lru_cache(maxsize=1)
def get_default_melbourne_scats_od_pairs() -> tuple[tuple[int, int], ...]:
    centroids = get_melbourne_scats_zone_centroids()
    return tuple((origin, destination) for origin in centroids for destination in centroids if origin != destination)


def get_default_melbourne_scats_trip_flows() -> tuple[float, ...]:
    return tuple(0.0 for _ in get_default_melbourne_scats_od_pairs())


def get_melbourne_scats_network_metadata() -> dict[str, Any]:
    nodes, _, centroid_nodes, _, metadata = _build_directed_link_specs()
    metadata = dict(metadata)
    metadata["topology_nodes"] = len(nodes)
    metadata["od_centroids"] = len(centroid_nodes)
    metadata["od_pairs"] = len(centroid_nodes) * (len(centroid_nodes) - 1)
    return metadata


def get_melbourne_scats_link_metadata() -> tuple[dict[str, Any], ...]:
    _, _, _, directed_specs, _ = _build_directed_link_specs()
    return tuple(
        {
            "link_id": link_id,
            "link_label": f"{spec.start}->{spec.end}",
            "start": int(spec.start),
            "end": int(spec.end),
            "source_link_id": str(spec.source_link_id),
            "source_direction": str(spec.source_direction),
            "source_direction_capacity_per_hour": float(spec.source_direction_capacity_per_hour),
            "highway": str(spec.highway),
            "ref": str(spec.ref),
            "name": str(spec.name),
            "lanes": int(spec.lanes),
            "posted_speed_kph": float(spec.posted_speed_kph),
            "posted_speed_factor": float(VITM_POSTED_SPEED_FACTOR if spec.highway == "primary" else 1.0),
            "speed_kph": float(spec.speed_kph),
            "vitm_akcelik_j": float(VITM_AKCELIK_J if spec.highway == "primary" else 0.0),
            "capacity_basis_link_type": str(spec.capacity_basis_link_type),
            "capacity_per_lane_per_hour": float(spec.capacity_per_lane_per_hour),
            "capacity_per_hour": float(spec.capacity_per_hour),
            "capacity_per_15min": float(spec.capacity_per_hour * DEFAULT_TIME_STEP_MINUTES / 60.0),
            "is_virtual_reverse": bool(spec.is_virtual_reverse),
            "capacity_adjustment": str(spec.capacity_adjustment),
        }
        for link_id, spec in enumerate(directed_specs)
    )
