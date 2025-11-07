# BPNeurode.py
from Neurode import Neurode, MultiLinkNode

class BPNeurode(Neurode):
    def __init__(self):
        super().__init__()
        self._delta = 0.0

    @staticmethod
    def _sigmoid_derivative(value: float):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value: float = None):
        if expected_value is not None:
            delta_error = expected_value - self.value
            self._delta = delta_error * self._sigmoid_derivative(self.value)
        else:
            down = self._neighbors[MultiLinkNode.Side.DOWNSTREAM]
            weighted = sum(d.get_weight(self) * d.delta for d in down)
            self._delta = weighted * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node: Neurode):
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._update_weights()
            self._fire_upstream()

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node: Neurode, adjustment: float):
        self._weights[node] += adjustment

    def _update_weights(self):
        for dn in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adj = self.learning_rate * dn.delta * self._value
            dn.adjust_weights(self, adj)

    def _fire_upstream(self):
        for up in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            up.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta
