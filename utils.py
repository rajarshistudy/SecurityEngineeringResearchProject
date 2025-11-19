import torch, random, numpy as np
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_mnist(batch=128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return train, test

def split_clients(dataset, n_clients=8, noniid=False):
    idxs = np.arange(len(dataset))
    if not noniid:
        parts = np.array_split(idxs, n_clients)
        return [Subset(dataset, p.tolist()) for p in parts]
    # simple non-IID: shard by label chunks
    labels = np.array(dataset.targets)
    client_sets = []
    for c in range(n_clients):
        keep_label = c % 10
        client_idx = np.where(labels == keep_label)[0]
        take = np.random.choice(client_idx, size=len(dataset)//(n_clients*2), replace=False)
        client_sets.append(Subset(dataset, take.tolist()))
    return client_sets

def loader(ds, batch=128, shuffle=True): return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2)

@torch.no_grad()
def evaluate(model, dl, device):
    import torch.nn.functional as F
    model.eval(); correct=0; total=0; loss=0.0
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return correct/total, loss/total
