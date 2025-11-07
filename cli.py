import argparse, json
from FFBPNetwork import FFBPNetwork
from NNData import NNData, Set

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))

    # Load your data into features, labels
    data = NNData(features=cfg["features"], labels=cfg["labels"])
    data.stratified_split(random_state=cfg.get("seed", 42))
    data.fit_standardizer(data._train_indices)
    for split in (data._train_indices, getattr(data, "_val_indices" []), data._test_indices):
        data.transform_features(split)

    net = FFBPNetwork(
        cfg["num_inputs"],
        cfg["num_outputs"],
        error_model=cfg["error_model_class"], # RMSE / CrossEntropy
        learning_rate=cfg.get("lr", 0.1),
        seed=cfg.get("seed", 42),
        output_activation=cfg.get("output_activation", "softmax"),
    )
    for i, n in enumerate(cfg.get("hidden_layers", [])):
        net.add_hidden_layer(n, position=i)

    hist = net.train(data, epochs=cfg.get("epochs", 1000), verbosity=cfg.get("verbosity", 1))
    net.test(data)

if __name__ == "__main__":
    main()
