# test_nndata.py
from NNData import NNData, Order, Set

X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0],[1],[1],[0]]
d = NNData(features=X, labels=Y, train_factor=0.5)

print("Samples (TRAIN, TEST):", d.number_of_samples(Set.TRAIN), d.number_of_samples(Set.TEST))

print("/nPrime TRAIN (STATIC)")
d.prime_data(Set.TRAIN, Order.STATIC)
while not d.pool_is_empty(Set.TRAIN):
    feat, lab = d.get_one_item(Set.TRAIN)
    print("TRAIN static:", feat, lab)

print("\nPrime TRAIN (SHUFFLE):")
d.prime_data(Set.TRAIN, Order.SHUFFLE)
while not d.pool_is_empty(Set.TRAIN):
    feat, lab = d.get_one_item(Set.TRAIN)
    print("TRAIN shuffle:", feat, lab)

print("\nPrime TEST (STATIC):")
d.prime_data(Set.TEST, Order.STATIC)
while not d.pool_is_empty(Set.TEST):
    feat, lab = d.get_one_item(Set.TEST)
    print("TEST static:", feat, lab)
