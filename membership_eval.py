import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from models.mnist_cnn import MNIST_CNN
from utils import get_mnist, loader
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def per_sample_loss(model, dl, device):
    model.eval()
    losses = []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        l = F.cross_entropy(logits, y, reduction="none")
        losses.extend(l.cpu().tolist())
    return losses

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = get_mnist()

    # small "member" set and "non-member" set
    in_idx  = np.random.choice(len(train_ds), 512, replace=False)
    out_idx = np.random.choice(len(test_ds), 512, replace=False)

    in_dl  = DataLoader(Subset(train_ds, in_idx), batch_size=64, shuffle=False)
    out_dl = DataLoader(Subset(test_ds, out_idx), batch_size=64, shuffle=False)

    # load one of your saved models here – you may need to add saving in train_*.py
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load("fedavg_final.pt"))  # or cronus_final.pt

    in_losses  = per_sample_loss(model, in_dl, device)
    out_losses = per_sample_loss(model, out_dl, device)

    y_true = [1]*len(in_losses) + [0]*len(out_losses)
    scores = [-x for x in in_losses] + [-x for x in out_losses]  # lower loss => more likely member
    auc = roc_auc_score(y_true, scores)
    print("Membership AUC:", auc)

if __name__ == "__main__":
    main()
