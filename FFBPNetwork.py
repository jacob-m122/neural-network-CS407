# FFBPNetwork.py
from __future__ import annotations
from typing import Type, Callable, Dict, List, Optional

import logging
import math
import random
import numpy as np

from NNData import NNData, Order, Set
from LayerList import LayerList
from FFBPNeurode import FFBPNeurode
from RMSE import RMSE
from CrossEntropy import CrossEntropy  # optional, for swapping loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmptySetException(Exception):
    """Raised when a requested TRAIN/TEST set is empty."""
    pass



class FFBPNetwork:
    """
    Feed-Forward Back-Propagation Network.
    - Topology is managed by LayerList.
    - Nodes are FFBPNeurode (inherits FF + BP behavior).
    - Loss model is pluggable (RMSE or CrossEntropy), must expose:
        * add(pred: List[float], true: List[float]) -> None
        * error (property: float)
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        error_model: Type[RMSE],
        *,
        learning_rate: float = 0.1,
        seed: Optional[int] = 42,
        output_activation: str = "sigmoid",  # or "softmax"
    ):
        """
        Args:
            num_inputs:  number of input nodes
            num_outputs: number of output nodes
            error_model: class (e.g., RMSE or CrossEntropy) with .add() and .error
            learning_rate: scalar LR propagated into nodes
            seed: RNG seed (for reproducible seeding elsewhere)
            output_activation: "sigmoid" (default) or "softmax" (multiclass)
        """
        self._list = LayerList(num_inputs, num_outputs, neurode_type=FFBPNeurode)
        self._error_model = error_model
        self._num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_activation = output_activation
        self._seed = seed
        if seed is not None:
            random.seed(seed)

        # push LR into currently existing nodes
        for n in (self._list.input_nodes + self._list.output_nodes):
            n.learning_rate = learning_rate
        self._learning_rate = learning_rate

    # ---------------------------------------------------------------------
    # Topology management
    # ---------------------------------------------------------------------
    def add_hidden_layer(self, num_nodes: int, position: int = 0):
        """
        Insert a hidden layer after moving `position` steps from head.
        Ensures new nodes inherit the network learning rate.
        """
        self._list.reset_to_head()
        for _ in range(position):
            if self._list.curr.next is not None:
                self._list.move_forward()
            else:
                print("Unable to move forward.")
        new_nodes = self._list.add_layer(num_nodes)
        if new_nodes:
            for n in new_nodes:
                n.learning_rate = self._learning_rate

    # ---------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------
    def _apply_output_activation(self):
        """Post-process output layer with optional softmax (for multiclass)."""
        if self._output_activation != "softmax":
            return
        outs = self._list.output_nodes
        logits = [n.value for n in outs]
        m = max(logits)
        exps = [math.exp(z - m) for z in logits]
        s = sum(exps) or 1.0
        for n, e in zip(outs, exps):
            # normalize in place (OK because FFBPNeurode stores value in _value)
            n._value = e / s  # noqa: SLF001

    @staticmethod
    def _argmax_idx(vec: List[float]) -> int:
        return max(range(len(vec)), key=lambda k: vec[k])

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def train(
        self,
        data_set: NNData,
        epochs: int = 1000,
        verbosity: int = 1,
        order: Order = Order.SHUFFLE,
        compute_accuracy: bool = True,
        on_epoch_end: Optional[Callable[[int, Dict[str, List[float]]], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train for `epochs` on the TRAIN set.
        Returns:
            history: dict with keys 'rmse' and optionally 'accuracy'
        """
        if data_set.number_of_samples(Set.TRAIN) == 0:
            raise EmptySetException("TRAIN set is empty; prime/split your NNData first.")

        history: Dict[str, List[float]] = {"rmse": []}
        if compute_accuracy:
            history["accuracy"] = []

        for epoch in range(epochs):
            metric = self._error_model()
            correct = 0
            total = 0

            data_set.prime_data(Set.TRAIN, order)

            while not data_set.pool_is_empty(Set.TRAIN):
                features, labels = data_set.get_one_item(Set.TRAIN)

                # 1) feed inputs
                for neurode, feature in zip(self._list.input_nodes, features):
                    neurode.set_input(float(feature))

                # 2) optional softmax on outputs
                self._apply_output_activation()

                predicted_values = [n.value for n in self._list.output_nodes]

                # normalize label shape to list[float]
                if hasattr(labels, "tolist"):
                    expected_values = labels.tolist()
                else:
                    expected_values = (
                        list(labels) if isinstance(labels, (list, tuple)) else [labels]
                    )

                # 3) accumulate loss
                metric.add(predicted_values, expected_values)

                # 4) backprop
                for neurode, expected in zip(self._list.output_nodes, expected_values):
                    neurode.set_expected(float(expected))

                # 5) accuracy
                if compute_accuracy:
                    if len(predicted_values) == 1 and len(expected_values) == 1:
                        # binary
                        p_class = 1 if predicted_values[0] >= 0.5 else 0
                        t_class = int(round(float(expected_values[0])))
                    else:
                        # multiclass one-hot
                        p_class = self._argmax_idx(predicted_values)
                        t_class = self._argmax_idx(expected_values)
                    correct += int(p_class == t_class)
                    total += 1

            rmse_val = metric.error
            history["rmse"].append(rmse_val)
            if compute_accuracy:
                acc = correct / max(total, 1)
                history["accuracy"].append(acc)

            if verbosity > 0:
                if compute_accuracy:
                    print(f"Epoch {epoch+1}/{epochs}  RMSE={rmse_val:.6f}  ACC={acc:.3f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}  RMSE={rmse_val:.6f}")

            if on_epoch_end:
                on_epoch_end(epoch, history)

        return history

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def test(self, data_set: NNData, order: Order = Order.STATIC, show_examples: int = 3):
        """
        Evaluate on TEST set.
        Prints summary metrics and a few example predictions.
        """
        if data_set.number_of_samples(Set.TEST) == 0:
            raise EmptySetException("TEST set is empty; prime/split your NNData first.")

        metric = self._error_model()
        data_set.prime_data(Set.TEST, order)

        preds: List[List[float]] = []
        trues: List[List[float]] = []

        while not data_set.pool_is_empty(Set.TEST):
            features, labels = data_set.get_one_item(Set.TEST)

            for neurode, feature in zip(self._list.input_nodes, features):
                neurode.set_input(float(feature))

            self._apply_output_activation()

            predicted_values = [n.value for n in self._list.output_nodes]
            if hasattr(labels, "tolist"):
                expected_values = labels.tolist()
            else:
                expected_values = (
                    list(labels) if isinstance(labels, (list, tuple)) else [labels]
                )

            metric.add(predicted_values, expected_values)
            preds.append(predicted_values)
            trues.append(expected_values)

        # accuracy (binary or multiclass)
        pred_classes: List[int] = []
        true_classes: List[int] = []
        for p, t in zip(preds, trues):
            if len(p) == 1 and len(t) == 1:
                pred_classes.append(1 if p[0] >= 0.5 else 0)
                true_classes.append(int(round(float(t[0]))))
            else:
                pred_classes.append(self._argmax_idx(p))
                true_classes.append(self._argmax_idx(t))

        correct = sum(int(pc == tc) for pc, tc in zip(pred_classes, true_classes))
        acc = correct / max(len(true_classes), 1) if true_classes else 0.0

        print(f"(test) Samples: {len(true_classes)}")
        print(f"(test) RMSE: {metric.error:.6f}  Accuracy: {acc:.3f}")

        # show a few examples from the end
        start = max(0, len(preds) - show_examples)
        for i in range(start, len(preds)):
            print(
                f"(test) idx {i}: true={true_classes[i]} pred={pred_classes[i]} "
                f"probs={['%.3f' % x for x in preds[i]]}"
            )

        # confusion matrix (rows=true, cols=pred)
        if true_classes:
            K = max(max(pred_classes), max(true_classes)) + 1
            cm = [[0] * K for _ in range(K)]
            for p, t in zip(pred_classes, true_classes):
                cm[t][p] += 1
            print("(test) Confusion Matrix (rows=true, cols=pred):")
            for row in cm:
                print(row)

    # ---------------------------------------------------------------------
    # Inference helpers
    # ---------------------------------------------------------------------
    def _forward_only(self, features):
        """Run a forward pass only (no backprop), return list of output values."""
        for neurode, feature in zip(self._list.input_nodes, features):
            neurode.set_input(float(feature))
        # apply optional softmax for multi-class
        self._apply_output_activation()
        return [n.value for n in self._list.output_nodes]

    def predict_proba(self, features):
        """
        Return output activations (probabilities for softmax; sigmoid score for binary).
        features: iterable of numbers with length == num_inputs
        """
        return self._forward_only(features)

    def predict(self, features):
        """
        Return class prediction.
        - Binary: threshold 0.5 on single sigmoid output.
        - Multi-class: argmax on outputs (with softmax if enabled).
        """
        probs = self._forward_only(features)
        if len(probs) == 1:
            return 1 if probs[0] >= 0.5 else 0
        return max(range(len(probs)), key=lambda k: probs[k])

    def predict_batch(self, X):
        """
        Vectorized convenience: list of predictions for a batch of feature vectors.
        """
        return [self.predict(x) for x in X]
    
    def softmax(z):
        shift = z - np.max(z)
        exp = np.exp(shift)
        return exp / np.sum(exp)
    