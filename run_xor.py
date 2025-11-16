# run_xor.py
from NNData import NNData, Order, Set
from FFBPNetwork import FFBPNetwork
from RMSE import RMSE
from CrossEntropy import CrossEntropy

# XOR data
XOR_features = [[0,0], [0,1], [1,0], [1,1]]
XOR_labels = [[0], [1], [1], [0]]


data = NNData(features=XOR_features, labels = XOR_labels, train_factor=1.0)
data.prime_data(Order.SHUFFLE)

# Network: 2 -> 2 (hidden) -> 1
net = FFBPNetwork(num_inputs=2, num_outputs=1, error_model=RMSE, learning_rate=0.3, seed=42)
net.add_hidden_layer(4)

print("=== TRAIN (XOR) ===")
net.train(data_set=data, epochs=8000, verbosity=1, order=Order.SHUFFLE)

print("\n=== TEST (XOR) ===")
data.split_set(.5)
net.test(data_set=data, order=Order.STATIC)

print("\n=== PREDICT ===")
for x in XOR_features:
    yhat = net.predict(x)
    print(f"Input {x} -> Predicted {yhat}")
