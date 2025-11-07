import json

def save_model(net, path, extra: dict | None = None):
    payload = {
        "output_activation": getattr(net, "_output_activation", "sigmoid"),
        "learning_rate": getattr(net, "_learning_rate", 0.1,),
        "layers": []
    }

    #deterministic layer/node ordering needed;
    layers = net._list.layers_aslists()
    for layer in layers:
        layer_dump = []
        for n in layer:
            w = {str(id(u)): n._weights.get(u, 0.0) for u in n._weights}
            layer_dump.append({"bias": getattr(n, "_bias", 0.0), "weights": w})
        payload["layers"].append(layer_dump)
    if hasattr(net, "_mu"): payload["mu"] = list(net._mu)
    if hasattr(net, "_sigma"): payload["sigma"] = list
    if extra: payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def load_model(net, path):
    with open(path) as f:
        payload = json.load(f)
    net._output_activation = payload.get("output_activation", "sigmoid")
    net._learning_rate = payload.get("learning_rate", 0.1)
    if "mu" in payload: net._mu = payload["mu"]
    if "sigma" in payload: net._sigma = payload["sigma"]
    