from typing import Dict
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from .utils import plot_training_curves, save_history

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item()
        all_y.append(y.cpu())
        all_p.append(logits.softmax(dim=1).argmax(dim=1).cpu())
    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_p).numpy()
    acc = accuracy_score(y_true, y_pred)
    return {"loss": total_loss / len(loader), "acc": acc}

def train(
    model, train_loader, val_loader, epochs=20, lr=1e-3, device="cpu",
    plot_path: str | None = None, hist_path: str | None = None,
) -> Dict[str, list]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()

        tr_metrics = evaluate(model, train_loader, device=device)
        va_metrics = evaluate(model, val_loader, device=device)
        history["train_loss"].append(running / len(train_loader))
        history["val_loss"].append(va_metrics["loss"])
        history["train_acc"].append(tr_metrics["acc"])
        history["val_acc"].append(va_metrics["acc"])

        print(f"Epoch {ep:03d} | train_loss={history['train_loss'][-1]:.4f} "
              f"val_loss={history['val_loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.3f} "
              f"val_acc={history['val_acc'][-1]:.3f}")

    if plot_path:
        plot_training_curves(history, outfile=plot_path)
    if hist_path:
        save_history(history, hist_path)
    return history
