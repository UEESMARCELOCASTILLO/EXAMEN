import os
import json
import torch
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def save_model(model, path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)

def load_model(model, path: str):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model

def plot_training_curves(history: dict, outfile: str | None = None):
    plt.figure()
    for k, v in history.items():
        plt.plot(v, label=k)
    plt.xlabel("Epoch")
    plt.title("Training Curves")
    plt.legend()
    if outfile:
        ensure_dir(os.path.dirname(outfile))
        plt.savefig(outfile, bbox_inches="tight", dpi=120)
    plt.show()

def save_history(history: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
