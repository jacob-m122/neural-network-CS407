#metrics.py
def accuracy_args_max(preds, labels):
    correct = 0
    for p, y in zip(preds, labels):
        pi = max(range(len(p))), key=lambda i: p[i]
        yi = max(range(len(y))), key = lambda i: y[i]
        correct += int(pi == yi)

    return correct / len(preds) if preds else float("nan")