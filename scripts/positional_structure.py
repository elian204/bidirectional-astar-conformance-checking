#!/usr/bin/env python3
"""Positional split/join features for workflow nets."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from core.petri_net import Place, Transition, WorkflowNet


PREFIX_THRESHOLD = 0.30
SUFFIX_THRESHOLD = 0.70
POSITIONAL_ALPHA = 3.0


POSITIONAL_FEATURE_NAMES: Sequence[str] = (
    "first_and_split_pos",
    "mean_and_split_pos",
    "last_and_join_pos",
    "mean_and_join_pos",
    "first_xor_split_pos",
    "mean_xor_split_pos",
    "last_xor_join_pos",
    "mean_xor_join_pos",
    "prefix_and_split_load_30",
    "suffix_and_join_load_30",
    "prefix_xor_split_load_30",
    "suffix_xor_join_load_30",
    "prefix_tau_density_30",
    "suffix_tau_density_30",
    "front_loaded_and_split_score",
    "back_loaded_and_join_score",
    "front_loaded_xor_split_score",
    "back_loaded_xor_join_score",
    "unavoidable_and_split_count",
    "unavoidable_and_join_count",
    "first_unavoidable_and_split_pos",
    "last_unavoidable_and_join_pos",
)


def _transition_visible(transition: Transition) -> bool:
    label = getattr(transition, "label", None)
    if label is None:
        return False
    text = str(label).strip()
    return bool(text and text.lower() != "tau")


def _node_entry_cost(node: object, visible_only: bool) -> int:
    if isinstance(node, Transition):
        if visible_only:
            return 1 if _transition_visible(node) else 0
        return 1
    return 0


def _shortest_node_costs(
    start_nodes: Iterable[object],
    adjacency: Mapping[object, Sequence[object]],
    *,
    visible_only: bool,
) -> Dict[object, float]:
    """0-1 BFS over a node-cost graph."""

    dist: Dict[object, float] = {node: math.inf for node in adjacency}
    queue: deque[object] = deque()

    for start in start_nodes:
        if start not in adjacency:
            continue
        dist[start] = 0.0
        queue.append(start)

    while queue:
        current = queue.popleft()
        current_dist = dist[current]
        for neighbor in adjacency[current]:
            weight = _node_entry_cost(neighbor, visible_only=visible_only)
            candidate = current_dist + weight
            if candidate >= dist.get(neighbor, math.inf):
                continue
            dist[neighbor] = candidate
            if weight == 0:
                queue.appendleft(neighbor)
            else:
                queue.append(neighbor)

    return dist


def _normalized_position(distance_from_start: float, distance_to_end: float) -> float:
    if not math.isfinite(distance_from_start) or not math.isfinite(distance_to_end):
        return math.nan
    total = distance_from_start + distance_to_end
    if total <= 0:
        return 0.5
    return float(distance_from_start / total)


def _nanmean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def _nanextreme(values: Sequence[float], *, mode: str) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    if mode == "min":
        return float(min(finite))
    if mode == "max":
        return float(max(finite))
    raise ValueError(f"Unsupported mode: {mode}")


def _prefix_load(positions: Sequence[float], loads: Sequence[int]) -> float:
    return float(
        sum(load for position, load in zip(positions, loads) if math.isfinite(position) and position <= PREFIX_THRESHOLD)
    )


def _suffix_load(positions: Sequence[float], loads: Sequence[int]) -> float:
    return float(
        sum(load for position, load in zip(positions, loads) if math.isfinite(position) and position >= SUFFIX_THRESHOLD)
    )


def _weighted_front_score(positions: Sequence[float], loads: Sequence[int]) -> float:
    return float(
        sum(
            load * math.exp(-POSITIONAL_ALPHA * position)
            for position, load in zip(positions, loads)
            if math.isfinite(position)
        )
    )


def _weighted_back_score(positions: Sequence[float], loads: Sequence[int]) -> float:
    return float(
        sum(
            load * math.exp(-POSITIONAL_ALPHA * (1.0 - position))
            for position, load in zip(positions, loads)
            if math.isfinite(position)
        )
    )


def _dominators(
    nodes: Sequence[object],
    predecessors: Mapping[object, Sequence[object]],
    start_node: object,
) -> Dict[object, set]:
    all_nodes = set(nodes)
    dom: Dict[object, set] = {node: set(all_nodes) for node in all_nodes}
    dom[start_node] = {start_node}

    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node == start_node:
                continue
            preds = predecessors.get(node, ())
            if not preds:
                new_dom = {node}
            else:
                pred_sets = [dom[pred] for pred in preds]
                new_dom = set.intersection(*pred_sets) if pred_sets else set()
                new_dom.add(node)
            if new_dom != dom[node]:
                dom[node] = new_dom
                changed = True

    return dom


def compute_positional_features(wf: WorkflowNet) -> Dict[str, float]:
    """Compute positional split/join features for a workflow net."""

    net = wf.net if hasattr(wf, "net") else wf
    places: List[Place] = list(net.places)
    transitions: List[Transition] = list(net.transitions)
    nodes: List[object] = [*places, *transitions]

    forward_adj: Dict[object, List[object]] = {node: [] for node in nodes}
    reverse_adj: Dict[object, List[object]] = {node: [] for node in nodes}
    place_in_degree: Dict[Place, int] = {place: 0 for place in places}
    place_out_degree: Dict[Place, int] = {place: 0 for place in places}

    for transition in transitions:
        preset = tuple(net.preset(transition))
        postset = tuple(net.postset(transition))
        for place in preset:
            forward_adj[place].append(transition)
            reverse_adj[transition].append(place)
            place_out_degree[place] += 1
        for place in postset:
            forward_adj[transition].append(place)
            reverse_adj[place].append(transition)
            place_in_degree[place] += 1

    initial_places = [place for place, count in wf.initial_marking.items() if count > 0]
    final_places = [place for place, count in wf.final_marking.items() if count > 0]
    if not initial_places:
        initial_places = places
    if not final_places:
        final_places = places

    visible_forward = _shortest_node_costs(initial_places, forward_adj, visible_only=True)
    visible_reverse = _shortest_node_costs(final_places, reverse_adj, visible_only=True)

    transition_pos: Dict[Transition, float] = {}
    place_pos: Dict[Place, float] = {}

    for transition in transitions:
        transition_pos[transition] = _normalized_position(
            visible_forward.get(transition, math.inf),
            visible_reverse.get(transition, math.inf),
        )

    for place in places:
        place_pos[place] = _normalized_position(
            visible_forward.get(place, math.inf),
            visible_reverse.get(place, math.inf),
        )

    and_splits = [transition for transition in transitions if len(net.postset(transition)) > 1]
    and_joins = [transition for transition in transitions if len(net.preset(transition)) > 1]
    xor_splits = [place for place in places if place_out_degree[place] > 1]
    xor_joins = [place for place in places if place_in_degree[place] > 1]

    and_split_positions = [transition_pos[transition] for transition in and_splits]
    and_join_positions = [transition_pos[transition] for transition in and_joins]
    xor_split_positions = [place_pos[place] for place in xor_splits]
    xor_join_positions = [place_pos[place] for place in xor_joins]

    and_split_loads = [max(len(net.postset(transition)) - 1, 0) for transition in and_splits]
    and_join_loads = [max(len(net.preset(transition)) - 1, 0) for transition in and_joins]
    xor_split_loads = [max(place_out_degree[place] - 1, 0) for place in xor_splits]
    xor_join_loads = [max(place_in_degree[place] - 1, 0) for place in xor_joins]

    prefix_transitions = [
        transition
        for transition in transitions
        if math.isfinite(transition_pos[transition]) and transition_pos[transition] <= PREFIX_THRESHOLD
    ]
    suffix_transitions = [
        transition
        for transition in transitions
        if math.isfinite(transition_pos[transition]) and transition_pos[transition] >= SUFFIX_THRESHOLD
    ]

    prefix_tau_density = 0.0
    if prefix_transitions:
        prefix_tau_density = float(
            sum(1 for transition in prefix_transitions if not _transition_visible(transition))
            / len(prefix_transitions)
        )

    suffix_tau_density = 0.0
    if suffix_transitions:
        suffix_tau_density = float(
            sum(1 for transition in suffix_transitions if not _transition_visible(transition))
            / len(suffix_transitions)
        )

    pseudo_source = object()
    pseudo_sink = object()
    dom_nodes: List[object] = [pseudo_source, *nodes, pseudo_sink]
    dom_predecessors: Dict[object, List[object]] = {node: [] for node in dom_nodes}
    for node in nodes:
        dom_predecessors[node] = list(reverse_adj[node])
    for place in initial_places:
        dom_predecessors[place].append(pseudo_source)
    dom_predecessors[pseudo_sink] = list(final_places)

    dominators = _dominators(dom_nodes, dom_predecessors, pseudo_source)
    sink_dominators = dominators[pseudo_sink]
    unavoidable_and_splits = [transition for transition in and_splits if transition in sink_dominators]
    unavoidable_and_joins = [transition for transition in and_joins if transition in sink_dominators]

    return {
        "first_and_split_pos": _nanextreme(and_split_positions, mode="min"),
        "mean_and_split_pos": _nanmean(and_split_positions),
        "last_and_join_pos": _nanextreme(and_join_positions, mode="max"),
        "mean_and_join_pos": _nanmean(and_join_positions),
        "first_xor_split_pos": _nanextreme(xor_split_positions, mode="min"),
        "mean_xor_split_pos": _nanmean(xor_split_positions),
        "last_xor_join_pos": _nanextreme(xor_join_positions, mode="max"),
        "mean_xor_join_pos": _nanmean(xor_join_positions),
        "prefix_and_split_load_30": _prefix_load(and_split_positions, and_split_loads),
        "suffix_and_join_load_30": _suffix_load(and_join_positions, and_join_loads),
        "prefix_xor_split_load_30": _prefix_load(xor_split_positions, xor_split_loads),
        "suffix_xor_join_load_30": _suffix_load(xor_join_positions, xor_join_loads),
        "prefix_tau_density_30": prefix_tau_density,
        "suffix_tau_density_30": suffix_tau_density,
        "front_loaded_and_split_score": _weighted_front_score(and_split_positions, and_split_loads),
        "back_loaded_and_join_score": _weighted_back_score(and_join_positions, and_join_loads),
        "front_loaded_xor_split_score": _weighted_front_score(xor_split_positions, xor_split_loads),
        "back_loaded_xor_join_score": _weighted_back_score(xor_join_positions, xor_join_loads),
        "unavoidable_and_split_count": float(len(unavoidable_and_splits)),
        "unavoidable_and_join_count": float(len(unavoidable_and_joins)),
        "first_unavoidable_and_split_pos": _nanextreme(
            [transition_pos[transition] for transition in unavoidable_and_splits],
            mode="min",
        ),
        "last_unavoidable_and_join_pos": _nanextreme(
            [transition_pos[transition] for transition in unavoidable_and_joins],
            mode="max",
        ),
    }
