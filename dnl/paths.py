from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

from .network.structures import Network

try:
    import networkx as nx
except Exception:  # pragma: no cover - networkx is expected, but keep a safe fallback.
    nx = None


@dataclass(frozen=True)
class Path:
    path_id: int
    od_index: int
    origin: int
    destination: int
    nodes: tuple[int, ...]
    links: tuple[int, ...]

    @property
    def label(self) -> str:
        return " - ".join(str(node) for node in self.nodes)


CandidatePathCacheKey = tuple[
    tuple[int, ...],
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int], ...],
    int,
]
CandidatePathCacheValue = tuple[tuple[Path, ...], tuple[tuple[int, ...], ...]]
_CANDIDATE_PATH_CACHE: dict[CandidatePathCacheKey, CandidatePathCacheValue] = {}


def _candidate_path_cache_key(
    network: Network,
    od_pairs: list[tuple[int, int]],
    max_paths_per_od: int,
) -> CandidatePathCacheKey:
    link_signature = tuple(
        (
            int(link.link_id),
            int(link.start),
            int(link.end),
            int(link.free_flow_steps),
        )
        for link in network.links
    )
    return (
        tuple(int(node) for node in network.nodes),
        link_signature,
        tuple((int(origin), int(destination)) for origin, destination in od_pairs),
        int(max_paths_per_od),
    )


def _copy_cached_candidate_paths(
    cached: CandidatePathCacheValue,
) -> tuple[list[Path], list[list[int]]]:
    paths, paths_by_od = cached
    return list(paths), [list(path_ids) for path_ids in paths_by_od]


def enumerate_simple_paths(
    network: Network,
    origin: int,
    destination: int,
    max_paths: int = 4,
    max_depth: int = 8,
    graph=None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if nx is not None:
        graph = graph if graph is not None else _build_nx_graph(network)

        if origin not in graph or destination not in graph:
            return []
        if not nx.has_path(graph, origin, destination):
            return []

        candidates: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []
        for node_path in islice(nx.shortest_simple_paths(graph, origin, destination, weight="weight"), max_paths):
            link_sequence: list[int] = []
            free_flow_cost = 0
            for start_node, end_node in zip(node_path[:-1], node_path[1:]):
                edge = graph[start_node][end_node]
                link_id = int(edge["link_id"])
                link_sequence.append(link_id)
                free_flow_cost += int(edge["weight"])
            candidates.append(
                (
                    free_flow_cost,
                    len(link_sequence),
                    tuple(int(node) for node in node_path),
                    tuple(link_sequence),
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(nodes, links) for _, _, nodes, links in candidates]

    candidates: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []

    def dfs(
        node: int,
        visited_nodes: set[int],
        node_sequence: list[int],
        link_sequence: list[int],
        free_flow_cost: int,
    ) -> None:
        if len(link_sequence) > max_depth:
            return
        if node == destination and link_sequence:
            candidates.append(
                (
                    free_flow_cost,
                    len(link_sequence),
                    tuple(node_sequence),
                    tuple(link_sequence),
                )
            )
            return

        for link_id in network.outgoing_by_node.get(node, ()):
            link = network.links[link_id]
            if link.end in visited_nodes:
                continue
            dfs(
                node=link.end,
                visited_nodes=visited_nodes | {link.end},
                node_sequence=node_sequence + [link.end],
                link_sequence=link_sequence + [link_id],
                free_flow_cost=free_flow_cost + link.free_flow_steps,
            )

    dfs(origin, {origin}, [origin], [], 0)
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    trimmed = candidates[:max_paths]
    return [(nodes, links) for _, _, nodes, links in trimmed]


def _build_nx_graph(network: Network):
    graph = nx.DiGraph()
    for link in network.links:
        graph.add_edge(link.start, link.end, weight=link.free_flow_steps, link_id=link.link_id)
    return graph


def build_candidate_paths(
    network: Network,
    od_pairs: list[tuple[int, int]],
    max_paths_per_od: int = 4,
) -> tuple[list[Path], list[list[int]]]:
    cache_key = _candidate_path_cache_key(network, od_pairs, max_paths_per_od)
    cached = _CANDIDATE_PATH_CACHE.get(cache_key)
    if cached is not None:
        return _copy_cached_candidate_paths(cached)

    all_paths: list[Path] = []
    paths_by_od: list[list[int]] = []
    graph = _build_nx_graph(network) if nx is not None else None

    for od_index, (origin, destination) in enumerate(od_pairs):
        raw_paths = enumerate_simple_paths(
            network=network,
            origin=origin,
            destination=destination,
            max_paths=max_paths_per_od,
            max_depth=max(6, len(network.nodes) + 1),
            graph=graph,
        )
        if not raw_paths:
            raise ValueError(f"No feasible path found for OD pair {(origin, destination)}.")

        path_ids: list[int] = []
        for nodes, links in raw_paths:
            path_id = len(all_paths)
            all_paths.append(
                Path(
                    path_id=path_id,
                    od_index=od_index,
                    origin=origin,
                    destination=destination,
                    nodes=nodes,
                    links=links,
                )
            )
            path_ids.append(path_id)
        paths_by_od.append(path_ids)

    cached_value = (
        tuple(all_paths),
        tuple(tuple(path_ids) for path_ids in paths_by_od),
    )
    _CANDIDATE_PATH_CACHE[cache_key] = cached_value
    return _copy_cached_candidate_paths(cached_value)

