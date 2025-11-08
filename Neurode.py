"""Implement the MultiLinkNode and Neurode classes."""
from __future__ import annotations
from enum import Enum, auto
import random
from abc import ABC, abstractmethod


class MultiLinkNode(ABC):
    """
    Base class which implements enum for upstream and downstream nodes,
    and tracks neighbor lists and 'all reported' bitmasks.
    """

    class Side(Enum):
        UPSTREAM = auto()
        DOWNSTREAM = auto()

    def __init__(self):
        # Bitmasks of which neighbors have reported on each side
        self._reporting_nodes = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0,
        }
        # Target bitmasks (all neighbors) for each side
        self._reference_value = {
            MultiLinkNode.Side.UPSTREAM: 0,
            MultiLinkNode.Side.DOWNSTREAM: 0,
        }
        # Neighbor lists
        self._neighbors = {
            MultiLinkNode.Side.UPSTREAM: [],
            MultiLinkNode.Side.DOWNSTREAM: [],
        }

    def __str__(self):
        upstream_id = [str(id(n)) for n in self._neighbors[MultiLinkNode.Side.UPSTREAM]]
        downstream_id = [str(id(n)) for n in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]]
        return (
            f"Node ID: {id(self)}\n"
            f"Upstream Neighbors: {', '.join(upstream_id)}\n"
            f"Downstream Neighbors: {', '.join(downstream_id)}\n"
        )

    @abstractmethod
    def _process_new_neighbor(self, node: "MultiLinkNode", side: "MultiLinkNode.Side"):
        """Hook for subclasses to react when a neighbor is (re)attached on a side."""
        raise NotImplementedError

    def reset_neighbors(self, nodes: list, side: "MultiLinkNode.Side"):
        """
        Replace the neighbor list on a side, call hook for each neighbor,
        and update the 'all reported' reference bitmask.
        """
        self._neighbors[side] = list(nodes)  # copy
        for node in nodes:
            self._process_new_neighbor(node, side)
        # bitmask with len(nodes) 1-bits (e.g., for 3 nodes => 0b111 == 7)
        self._reference_value[side] = (1 << len(nodes)) - 1

    def add_neighbor(self, node: "MultiLinkNode", side: "MultiLinkNode.Side"):
        """
        Append a single neighbor on the given side and update reference bitmask.
        Calls _process_new_neighbor so subclasses can initialize weights, etc.
        """
        self._neighbors[side].append(node)
        self._process_new_neighbor(node, side)
        # Update the "all reported" reference mask to match new neighbor count
        self._reference_value[side] = (1 << len(self._neighbors[side])) - 1


class Neurode(MultiLinkNode):
    """Neurode adds values and upstream weights to MultiLinkNode."""

    _learning_rate = 0.05  # class-level default

    @property
    def learning_rate(self):
        return Neurode._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        Neurode._learning_rate = float(value)

    def __init__(self):
        super().__init__()
        self._value = 0.0
        self._weights = {}  # maps UPSTREAM neighbor -> weight

    def _process_new_neighbor(self, node: "Neurode", side: "MultiLinkNode.Side"):
        # Assign a random weight when an UPSTREAM neighbor is attached
        if side is MultiLinkNode.Side.UPSTREAM:
            # Keep existing weight if relinking the same neighbor
            if node not in self._weights:
                self._weights[node] = random.random()

    def _check_in(self, node: "Neurode", side: "MultiLinkNode.Side") -> bool:
        """
        Mark that 'node' has reported on 'side'. When all neighbors on that side
        have reported, reset mask and return True.
        """
        idx = self._neighbors[side].index(node)
        self._reporting_nodes[side] |= (1 << idx)
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        return False

    def get_weight(self, node: "Neurode") -> float:
        return self._weights[node]

    @property
    def value(self) -> float:
        return self._value

    # Optional: a setter if you need to directly assign outputs during forward pass
    @value.setter
    def value(self, v: float):
        self._value = float(v)
