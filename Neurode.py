from __future__ import annotations
from enum import Enum, auto
from typing import Dict

class MultiLinkNode:
    class Side(Enum):
        UPSTREAM = auto()
        DOWNSTREAM = auto()

class Neurode:
    """Base node with neighbor/weight bookkeeping."""
    # we now keep learning_rate as an *instance* attribute.
    def __init__(self):
        self._neighbors: Dict[MultiLinkNode.Side, set] = {
            MultiLinkNode.Side.UPSTREAM: set(),
            MultiLinkNode.Side.DOWNSTREAM: set(),
        }
        self._weights: Dict[Neurode, float] = {}
        self._value: float = 0.0
        self._data_ready_from: Dict[MultiLinkNode.Side, set] = {
            MultiLinkNode.Side.UPSTREAM: set(),
            MultiLinkNode.Side.DOWNSTREAM: set(),
        }
        # NEW: bias & instance LR
        self._bias: float = 0.0
        self.learning_rate: float = 0.1

    # ----- Bias helpers (used by FF/BP) -----
    def get_bias(self) -> float:
        return self._bias

    def set_bias(self, b: float) -> None:
        self._bias = float(b)

    def adjust_bias(self, delta: float) -> None:
        self._bias += float(delta)

    # ----- Edge helpers -----
    def get_weight(self, node: "Neurode") -> float:
        return self._weights.get(node, 0.0)

    def set_weight(self, node: "Neurode", value: float) -> None:
        self._weights[node] = float(value)

    # ----- Value property -----
    @property
    def value(self) -> float:
        return self._value

    # ----- Neighbor registration / data flow checks -----
    def add_neighbor(self, node: "Neurode", side: MultiLinkNode.Side):
        self._neighbors[side].add(node)

    def _check_in(self, node: "Neurode", side: MultiLinkNode.Side) -> bool:
        """Returns True only when we've heard from all neighbors on that side."""
        self._data_ready_from[side].add(node)
        ready = self._data_ready_from[side] == self._neighbors[side]
        if ready:
            self._data_ready_from[side].clear()
        return ready

    # Stubs for subclasses to implement
    def data_ready_upstream(self, node: "Neurode"): ...
    def data_ready_downstream(self, node: "Neurode"): ...
