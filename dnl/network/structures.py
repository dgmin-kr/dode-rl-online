from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Link:
    link_id: int
    start: int
    end: int
    free_flow_steps: int
    backward_wave_steps: int
    capacity: float
    jam_storage: float

    @property
    def label(self) -> str:
        return f"{self.start}->{self.end}"


@dataclass(frozen=True)
class Network:
    nodes: tuple[int, ...]
    links: tuple[Link, ...]
    outgoing_by_node: dict[int, tuple[int, ...]]
    incoming_by_node: dict[int, tuple[int, ...]]
    node_positions: dict[int, tuple[float, float]]
    centroid_nodes: tuple[int, ...] = field(default_factory=tuple)

    @property
    def num_links(self) -> int:
        return len(self.links)
