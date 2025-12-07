Feed-Forward Backpropagation Neural Network (FFBPNetwork)

A CS 407 Final Project — Jacob Mitani

Overview

This project is a from-scratch implementation of a Feed-Forward Backpropagation (FFBP) neural network in Python. It supports:

Fully connected layers

Arbitrary hidden-layer topology

Sigmoid activation

RMSE and Cross-Entropy evaluation metrics

Train/test splitting with a custom NNData class

Classification tasks on both XOR and Iris datasets

The project was originally developed as part of Foothill College’s CS 3B course and then reworked and modernized for the University of Oregon’s CS 407 course.

Key Features
✔ Custom Neural Network Architecture

Input, hidden, and output neurodes connected using a custom LayerList and MultiLinkNode system

Forward propagation using sigmoid activation

Backpropagation with delta rule and weight updates

✔ Multiple Loss Metrics

RMSE (Root Mean Squared Error)

CrossEntropy (implemented this term)

✔ Dataset Support

XOR (2-class nonlinear benchmark)

Iris (3-class classification problem)

✔ Flexible Topologies

Experimented with architectures:

4–3–3

4–5–3

4–6–3

Repository Structure
.
├── FFBPNetwork.py          # Main network class
├── FFNeurode.py            # Forward-pass neurode
├── BPNeurode.py            # Backpropagation neurode
├── FFBPNeurode.py          # Combined FF/BP neurode
├── LayerList.py            # Doubly-linked list managing layers
├── DoublyLinkedList.py     # Supporting data structure
├── NNData.py               # Data handling, splitting, and pools
├── RMSE.py                 # RMSE error metric
├── CrossEntropy.py         # Cross-entropy metric
├── run_xor.py              # XOR training/testing script
├── FFBPNetworkTest.py      # Iris dataset experiment
└── README.md               # You're here!

Installation

Requires Python 3.9+.

Clone the repository:

git clone https://github.com/jacob-m122/neural-network-CS407
cd neural-network-CS407


(Optional) Create a virtual environment:

python3 -m venv myenv
source myenv/bin/activate
pip install numpy

Running the XOR Demo
python run_xor.py


Expected output:

100% accuracy

RMSE ≈ 0.03

Correct predictions for all XOR inputs

Example:

=== TEST (XOR) ===
RMSE: 0.0327  Accuracy: 1.000
Confusion Matrix:
[1, 0]
[0, 1]

Running the Iris Demo
python FFBPNetworkTest.py


This trains a network (default: 4–3–3) on the Iris dataset using:

70% training

30% test split

5000 epochs

CrossEntropy evaluation metric

Sigmoid output activation

Example result (for 4–6–3 topology):

(test) RMSE: 0.0386  Accuracy: 0.978

Confusion Matrix:
[16, 0, 0]
[0, 14, 0]
[0, 1, 14]

Topologies Tested
Topology	Activation	Metric	Accuracy	Notes
4–3–3	Sigmoid	CrossEntropy	97.8%	Smallest model, stable
4–5–3	Sigmoid	CrossEntropy	97.8%	Different misclassification pattern
4–6–3	Sigmoid	CrossEntropy	97.8%	Most confident predictions

All models correctly classified 44/45 test samples.

Softmax Exploration (Out of Scope)

A softmax output activation was partially implemented.
However, fully integrating softmax requires:

A different form of gradient computation

Coordinated deltas across output neurodes

This turned out to be outside the scope of this term, but the exploration improved understanding of activation functions and backprop interactions.

Future Improvements

True softmax + categorical cross-entropy backprop

Modular activation support (ReLU, tanh, etc.)

Weight initialization schemes (Xavier / He)

Learning rate scheduling

Visualization tools for loss and decision boundaries

Author

Jacob Mitani
CS Major @ University of Oregon
Email: jacobmitani@uoregon.edu

GitHub: https://github.com/jacob-m122

License

This project is released for educational use. Feel free to fork, extend, or reference it.

If you'd like, I can generate:

A version with screenshots

A minimal version

A more advanced, technical version

A student-friendly or professor-friendly rewrite

Just tell me!