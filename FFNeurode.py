"""Implement FFNeurode Class."""
from __future__ import annotations
from Neurode import Neurode, MultiLinkNode
from math import exp

class FFNeurode(Neurode):
    """Inherit from parent, Neurode, implement feed-forward process."""

    @staticmethod
    def _sigmoid(value: float):
        """Return sigmoid function result."""
        return 1 / (1 + exp(-value))

    def _calculate_value(self):
        """Calculate sum of weighted upstream node values (+ bias)."""
        weighted_sum = 0.0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node_value = node.value
            weight = self.get_weight(node)
            weighted_sum += node_value * weight
        weighted_sum += self.get_bias()  # NEW: bias addend
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """Indicate to downstream neighbors that node has data ready."""
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)

    def data_ready_upstream(self, node: Neurode):
        """Handle data from upstream node."""
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set input node value, indicate to downstream nodes."""
        self._value = float(input_value)
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)
