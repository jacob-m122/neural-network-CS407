"""Implement the LayerList class for input, hidden, and output layer management."""

from DoublyLinkedList import DoublyLinkedList
from Neurode import Neurode  # expects helpers like add_neighbor(), set_weight(), reset_weights()
import math
import random


class _BiasNeurode(Neurode):
    """Constant-1 bias node with no upstream neighbors."""
    def __init__(self):
        super().__init__()
        # lock the activation to 1.0
        self._value = 1.0


class LayerList(DoublyLinkedList):
    """Implement LayerList class with optional bias, seeding, and output helpers."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        neurode_type: type(Neurode),
        *,
        use_bias: bool = True,
        seed: int | None = None,
        init_mode: str = "fan_in",   # "fan_in" or "uniform"
        softmax_output: bool = False
    ):
        """
        Initialize superclass, create input & output layers, link them.

        Args:
            inputs: number of input neurodes
            outputs: number of output neurodes
            neurode_type: class to instantiate for each neurode
            use_bias: if True, add a bias node feeding each non-input layer
            seed: if provided, makes initial weights deterministic
            init_mode: "fan_in" (scaled uniform ±1/sqrt(fan_in), recommended) or "uniform" (0..1 default)
            softmax_output: if True, output helper will softmax outputs by default
        """
        super().__init__()
        self._neurode_type = neurode_type
        self._use_bias = use_bias
        self._seed = seed
        self._init_mode = init_mode
        self._softmax_output_default = softmax_output

        if self._seed is not None:
            random.seed(self._seed)

        # bias nodes keyed by the python id() of a layer list object
        self._bias_nodes: dict[int, _BiasNeurode] = {}

        # Build layers
        self.input_layer = [neurode_type() for _ in range(inputs)]
        self.output_layer = [neurode_type() for _ in range(outputs)]

        # Chain layers in the DLL
        self.add_to_head(self.input_layer)
        self.add_after_current(self.output_layer)

        # Wire input -> output
        self.linking_helper(self.input_layer, self.output_layer)

    # --------------------------
    # Core linking + initialization
    # --------------------------
    def linking_helper(self, upstream_layer, downstream_layer):
        """Link neighboring layers; optionally attach a bias node to downstream."""
        # Existing behavior: reset neighbors both ways
        for neurode in upstream_layer:
            neurode.reset_neighbors(downstream_layer, Neurode.Side.DOWNSTREAM)
        for neurode in downstream_layer:
            neurode.reset_neighbors(upstream_layer, Neurode.Side.UPSTREAM)

        # Optional bias for the downstream layer
        if self._use_bias:
            self._attach_bias_to_layer(downstream_layer)

        # Optional deterministic/scaled init for incoming weights to downstream layer
        if self._seed is not None or self._init_mode in ("fan_in", "uniform"):
            self._seed_incoming_weights(downstream_layer)

    def _attach_bias_to_layer(self, layer):
        """Ensure a single bias node feeds every node in `layer` as an extra UPSTREAM neighbor."""
        layer_key = id(layer)
        bias = self._bias_nodes.get(layer_key)
        if bias is None:
            bias = _BiasNeurode()
            self._bias_nodes[layer_key] = bias

        # Attach (without disturbing existing upstreams) using add_neighbor helper
        for node in layer:
            # If add_neighbor helper exists, use it (minimal change).
            # Fallback would be to rebuild via reset_neighbors, but we avoid that.
            node.add_neighbor(bias, Neurode.Side.UPSTREAM)

    def _seed_incoming_weights(self, layer):
        """
        Re-seed incoming weights to each node in `layer`.
        Modes:
            - fan_in: Xavier/LeCun-like uniform in [-1/sqrt(fan_in), +1/sqrt(fan_in)]
            - uniform: leave the default 0..1 random from Neurode (no change)
        """
        if self._init_mode == "uniform":
            # Respect default random init already created by reset_neighbors/add_neighbor.
            # If a seed was provided, we can re-randomize deterministically in [0,1).
            if self._seed is not None:
                for node in layer:
                    node.reset_weights(seed=self._seed)
            return

        if self._init_mode == "fan_in":
            for node in layer:
                # The node has upstream neighbors already (including bias if enabled)
                fan_in = node.fan_in()
                if fan_in <= 0:
                    continue
                limit = 1.0 / math.sqrt(fan_in)
                # Overwrite each upstream weight with scaled uniform
                for up in node.neighbors(Neurode.Side.UPSTREAM):
                    w = random.uniform(-limit, limit)
                    node.set_weight(up, w)

    # --------------------------
    # Public API: topology edits
    # --------------------------
    def add_layer(self, num_nodes: int):
        """
        Insert hidden layer between current layer (not tail) and the next, then link.
        Keeps your existing behavior; bias and seeding are applied automatically.
        """
        if self._curr == self._tail:
            raise IndexError
        new_hidden_layer = [self._neurode_type() for _ in range(num_nodes)]
        self.add_after_current(new_hidden_layer)

        # Link previous current (upstream) -> new hidden
        upstream_layer = self._curr.data
        self.move_forward()  # move onto the new hidden layer node
        self.linking_helper(upstream_layer, new_hidden_layer)

        # Link new hidden -> existing downstream
        downstream_layer = self._curr.next.data
        self.linking_helper(new_hidden_layer, downstream_layer)

    def remove_layer(self):
        """
        Remove the layer after current, ensuring it is not the output layer.
        Reconnect upstream to the next downstream; bias & seeding reapplied.
        """
        if self._curr.next == self._tail:
            raise IndexError
        upstream_layer = self._curr.data
        removed = self._curr.next.data
        # Clean any bias node tied to the removed layer
        self._bias_nodes.pop(id(removed), None)

        self.remove_after_current()
        if self._curr.next:
            downstream_layer = self._curr.next.data
            self.linking_helper(upstream_layer, downstream_layer)

    # --------------------------
    # Learning rate convenience
    # --------------------------
    def set_learning_rate(self, lr: float):
        """
        Convenience to set LR across all nodes. Works whether your Neurode uses
        an instance-level or class-level learning_rate property.
        """
        for layer in self:
            for n in layer.data:
                n.learning_rate = lr

    # --------------------------
    # Output helpers / softmax
    # --------------------------
    @staticmethod
    def _softmax(vals, temperature: float = 1.0):
        m = max(vals)
        exps = [math.exp((v - m) / max(1e-9, temperature)) for v in vals]
        s = sum(exps) or 1.0
        return [e / s for e in exps]

    def output_values(self, *, softmax: bool | None = None, temperature: float = 1.0):
        """
        Return output layer activations, with optional softmax.
        If softmax is None, uses self._softmax_output_default.
        """
        values = [n.value for n in self._tail.data]
        use_sm = self._softmax_output_default if softmax is None else softmax
        if use_sm:
            return self._softmax(values, temperature)
        return values

    # --------------------------
    # Properties (unchanged)
    # --------------------------
    @property
    def input_nodes(self):
        """Access input layer neurodes."""
        return self._head.data

    @property
    def output_nodes(self):
        """Access output layer neurodes."""
        return self._tail.data