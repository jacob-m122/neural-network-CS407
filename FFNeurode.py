"""Implement FFNeurode Class."""
from __future__ import annotations
from Neurode import Neurode, MultiLinkNode
from math import exp

class FFNeurode(Neurode):
    """Inherit from Neurode; implement feed-forward process with optional bias."""

    def __init__(self):
        # Cooperative init ensures MultiLinkNode/Neurode internals are set up
        super().__init__()

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Return sigmoid function result."""
        return 1.0 / (1.0 + exp(-value))

    def _get_bias_safe(self) -> float:
        """Use get_bias() if available; otherwise no-bias fallback."""
        gb = getattr(self, "get_bias", None)
        if callable(gb):
            return float(gb())
        return 0.0

    def _calculate_value(self):
        """Calculate sum of weighted upstream node values (+ bias)."""
        weighted_sum = 0.0
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            weighted_sum += node.value * self.get_weight(node)
        weighted_sum += self._get_bias_safe()  # bias addend (safe)
        self._value = self._sigmoid(weighted_sum)

    def _fire_downstream(self):
        """Notify downstream neighbors that this node has data ready."""
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)

    def data_ready_upstream(self, node: Neurode):
        """Handle data from an upstream node."""
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set input node value and notify downstream nodes."""
        self._value = float(input_value)
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)
