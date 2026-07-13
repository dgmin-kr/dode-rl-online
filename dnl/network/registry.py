from __future__ import annotations

from dataclasses import dataclass

from .melbourne_scats import (
    build_melbourne_scats_network,
    get_default_melbourne_scats_od_pairs,
    get_default_melbourne_scats_trip_flows,
)
from .structures import Network


NETWORK_NAME = "melbourne_scats"
NETWORK_DISPLAY_NAME = "Melbourne SCATS"
NETWORK_ALIASES = {
    "melbourne": NETWORK_NAME,
    "melbourne_scats": NETWORK_NAME,
    "melbourne-scats": NETWORK_NAME,
    "melbournescats": NETWORK_NAME,
    "scats": NETWORK_NAME,
}


@dataclass(frozen=True)
class NetworkDefinition:
    name: str
    network: Network
    od_pairs: list[tuple[int, int]]
    default_max_paths_per_od: int
    default_action_high: float


def canonical_network_name(network_name: str = NETWORK_NAME) -> str:
    normalized = str(network_name).strip().lower().replace(" ", "_")
    if normalized not in NETWORK_ALIASES:
        raise ValueError(
            f"Unsupported network {network_name!r}. This project is Melbourne-only; "
            f"use {NETWORK_NAME!r}."
        )
    return NETWORK_NAME


def build_network_definition(
    network_name: str = NETWORK_NAME,
    *,
    time_step_minutes: float | None = None,
) -> NetworkDefinition:
    canonical_network_name(network_name)
    network, od_pairs = build_melbourne_scats_network(
        **({} if time_step_minutes is None else {"time_step_minutes": float(time_step_minutes)})
    )
    return NetworkDefinition(
        name=NETWORK_NAME,
        network=network,
        od_pairs=od_pairs,
        default_max_paths_per_od=2,
        default_action_high=30.0,
    )


def get_network_display_name(network_name: str = NETWORK_NAME) -> str:
    canonical_network_name(network_name)
    return NETWORK_DISPLAY_NAME


def get_default_action_high(network_name: str = NETWORK_NAME) -> float:
    canonical_network_name(network_name)
    return build_network_definition(network_name).default_action_high


def get_default_od_pairs(network_name: str = NETWORK_NAME) -> tuple[tuple[int, int], ...]:
    canonical_network_name(network_name)
    return get_default_melbourne_scats_od_pairs()


def get_default_trip_flows(network_name: str = NETWORK_NAME) -> tuple[float, ...]:
    canonical_network_name(network_name)
    return get_default_melbourne_scats_trip_flows()
