"""Implement the LayerList class for input, hidden, and output layer management."""


from DoublyLinkedList import DoublyLinkedList
from Neurode import Neurode

class LayerList(DoublyLinkedList):
    """implement LayerList class."""

    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        """
        Initialize superclass, private neurode_type, input layer, output layer.

        Add input layer to head and output layer after current.
        Link input and output layers.
        """
        super().__init__()
        self._neurode_type = neurode_type
        self.input_layer = [
            neurode_type() for _ in range(inputs)
            ]
        self.output_layer = [
            neurode_type() for _ in range(outputs)
            ]
        self.add_to_head(self.input_layer)
        self.add_after_current(self.output_layer)
        self.linking_helper(self.input_layer, self.output_layer)

    def linking_helper(self, upstream_layer, downstream_layer):
        """Link neighboring layers."""
        for neurode in upstream_layer:
            neurode.reset_neighbors(downstream_layer, Neurode.Side.DOWNSTREAM)
        for neurode in downstream_layer:
            neurode.reset_neighbors(upstream_layer, Neurode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        """
        Ensure current layer is not output layer (tail).

        Insert hidden layer between input and output layers, link layers.
        """
        if self._curr == self._tail:
            raise IndexError
        new_hidden_layer = [
            self._neurode_type() for _ in range(num_nodes)
        ]
        self.add_after_current(new_hidden_layer)
        upstream_layer = self._curr.data
        self.move_forward()
        self.linking_helper(upstream_layer, new_hidden_layer)
        downstream_layer = self._curr.next.data
        self.linking_helper(new_hidden_layer, downstream_layer)

    def remove_layer(self):
        """
        Remove specific layer.

        Ensure it is not removing output layer (tail).
        """
        if self._curr.next == self._tail:
            raise IndexError
        upstream_layer = self._curr.data
        self.remove_after_current()
        if self._curr.next:
            downstream_layer = self._curr.next.data
            self.linking_helper(upstream_layer, downstream_layer)

    @property
    def input_nodes(self):
        """Access Input layer neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Access output layer neurodes."""
        return self._tail.data


"""
Layerlist created.
Input neurodes: 2
Output neurodes: 1
"""
