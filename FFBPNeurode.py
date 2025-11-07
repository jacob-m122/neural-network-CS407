# FFBPNeurode.py
from FFNeurode import FFNeurode
from BPNeurode import BPNeurode

class FFBPNeurode(FFNeurode, BPNeurode):
    """Feedforward + Backprop Neurode."""
    def __init__(self):
        super().__init__()   # walks FFNeurode -> BPNeurode -> Neurode -> MultiLinkNode

    # not strictly required, but OK to include:
    def reset_neighbors(self, nodes, side):
        return super().reset_neighbors(nodes, side)
