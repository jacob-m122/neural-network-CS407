from NNData import NNData, Order, Set
from LayerList import LayerList
from FFBPNeurode import FFBPNeurode
from RMSE import RMSE
from CrossEntropy import CrossEntropy
from typing import Type, Callable, Dict, List, Optional, Tuple
import logging, math, random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EmptySetException(Exception):
    """Implement custom exception class."""
    pass

class FFBPNetwork():
    """Implement feed-forward back-propagation network class"""

    def __init__(self, num_inputs: int, num_outputs: int, error_model: Type[RMSE], *, learning_rate: float=0.1, seed: Optional[int] = 42, output_activation: str = "sigmoid"):

        """
        Initialize LayerList instance, error model, inputs, and outputs.
        seed: optional RNG seed for reproducibility of weight/bias init (if upstream uses random).
        """

        
        self._list = LayerList(num_inputs, num_outputs, neurode_type=FFBPNeurode)
        self._error_model = error_model
        self._num_inputs = num_inputs
        self.num_outputs = num_outputs
        self._output_activation = output_activation
        self._seed = seed
        if seed is not None:
            random.seed(seed)
        for n in (self._list.input_nodes + self._list.output_nodes):
                n.learning_rate = learning_rate
        self._learning_rate = learning_rate


    def add_hidden_layer(self, num_nodes: int, position=0):
        """Add hidden layer if position is greater than zero, move forward through layers."""
        self._list.reset_to_head()
        for _ in range(position):
            if self._list.curr.next is not None:
                self._list.move_forward()
            else:
                print("Unable to move forward.")
        new_nodes = self._list.add_layer(num_nodes)
        #ensure LR propagates to new nodes
        for n in new_nodes:
            n.learning_rate = self._learning_rate

    # ----- small helpers -----
    def _apply_output_activation(self):
        """Post-process output layer values once all forward ops have settled."""
        outs = self._list.output_nodes
        if self._output_activation == "softmax":
            logits = [n.value for n in outs]
            m = max(logits)
            exps = [math.exp(z - m) for z in logits]
            s = sum(exps) or 1.0
            for n, e in zip(outs, exps):
                n._value = e / s  # normalize in place

    @staticmethod
    def _argmax_idx(vec: List[float]) -> int:
        return max(range(len(vec)), key=lambda k: vec[k])


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
        Returns history dict with 'rmse' and (if enabled) 'accuracy'.
        Prints once per epoch (if verbosity>0).
        """
        if data_set.number_of_samples(Set.TRAIN) == 0:
            raise EmptySetException

        history = {"rmse": []}
        if compute_accuracy:
            history["accuracy"] = []

        for epoch in range(epochs):
            rmse_object = self._error_model()
            correct = 0
            total = 0

            data_set.prime_data(Set.TRAIN, order)

            while not data_set.pool_is_empty(Set.TRAIN):
                features, labels = data_set.get_one_item(Set.TRAIN)

                # feed inputs
                for neurode, feature in zip(self._list.input_nodes, features):
                    neurode.set_input(input_value=feature)

                # optional softmax at output
                self._apply_output_activation()

                predicted_values = [n.value for n in self._list.output_nodes]
                expected_values = labels

                # accumulate loss
                rmse_object += (predicted_values, expected_values)

                # backprop targets
                for neurode, expected in zip(self._list.output_nodes, expected_values):
                    neurode.set_expected(expected)

                # accuracy bookkeeping (argmax on one-hot)
                if compute_accuracy:
                    p = self._argmax_idx(predicted_values)
                    t = self._argmax_idx(expected_values)
                    correct += int(p == t)
                    total += 1

            # end-of-epoch logging
            history["rmse"].append(rmse_object.error)
            if compute_accuracy:
                acc = correct / max(total, 1)
                history["accuracy"].append(acc)

            if verbosity > 0:
                if compute_accuracy:
                    print(f"Epoch {epoch+1}/{epochs}  RMSE={rmse_object.error:.6f}  ACC={acc:.3f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}  RMSE={rmse_object.error:.6f}")

            if on_epoch_end:
                on_epoch_end(epoch, history)

        return history


    def test(self, data_set: NNData, order=Order.STATIC, show_examples: int = 3):
        """
        Utilize testing set to track testing progress.
        Records RMSE and (if one-hot) accuracy. Prints a brief summary and a few examples.
        Returns:
            metrics: dict with 'rmse' and optional 'accuracy'.
        """
        """
        Summarize metrics and show a small sample.
        """
        if data_set.number_of_samples(Set.TEST) == 0:
            raise EmptySetException

        rmse_object = self._error_model()
        data_set.prime_data(Set.TEST, order)

        preds: List[List[float]] = []
        trues: List[List[float]] = []

        while not data_set.pool_is_empty(Set.TEST):
            features, labels = data_set.get_one_item(Set.TEST)
            for neurode, feature in zip(self._list.input_nodes, features):
                neurode.set_input(feature)

            self._apply_output_activation()

            predicted_values = [n.value for n in self._list.output_nodes]
            expected_values = labels
            rmse_object += (predicted_values, expected_values)

            preds.append(predicted_values)
            trues.append(expected_values)

        # accuracy + confusion
        pred_classes = [self._argmax_idx(p) for p in preds]
        true_classes = [self._argmax_idx(t) for t in trues]
        correct = sum(int(p == t) for p, t in zip(pred_classes, true_classes))
        acc = correct / max(len(true_classes), 1)

        print(f"(test) Samples: {len(true_classes)}")
        print(f"(test) RMSE: {rmse_object.error:.6f}  Accuracy: {acc:.3f}")

        # small sample of last 3 items
        for i in range(max(0, len(preds) - 3), len(preds)):
            print(f"(test) Input idx {i}: true={true_classes[i]} pred={pred_classes[i]} "
                  f"probs={['%.3f'%x for x in preds[i]]}")

        # confusion matrix
        K = (max(max(pred_classes), max(true_classes)) + 1) if true_classes else 0
        cm = [[0]*K for _ in range(K)]
        for p, t in zip(pred_classes, true_classes):
            cm[t][p] += 1
        print("(test) Confusion Matrix (rows=true, cols=pred):")
        for row in cm:
            print(row)