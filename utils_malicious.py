#----------------------------------------------------------------------------
# When using files from this git, remember to change the import statement
# in the malicious files from utils to utils_malicious
# The purpose of creating two separate files is to showecas progression in 
# the experiment and the research
#----------------------------------------------------------------------------

import torch, random, numpy as np
from torch.utils.data import Subset
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets, transforms

def set_seed(s=1337):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_mnist(batch=128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
    return train, test

def split_clients(dataset, n_clients=8, noniid=True):
    # --- Extract labels safely ---
    if isinstance(dataset, Subset):
        base = dataset.dataset
        subset_idx = np.array(dataset.indices)
        labels = np.array(base.targets)[subset_idx]
        idx = np.arange(len(subset_idx))  # indices 0..subset_size-1 (relative)
    else:
        labels = np.array(dataset.targets)
        idx = np.arange(len(labels))

    # --- Non-IID split using shards ---
    if noniid:
        n_shards = n_clients * 2
        shard_size = len(idx) // n_shards

        # Sort by labels
        idx_sorted = idx[np.argsort(labels)]

        # Build shards
        shards = [
            idx_sorted[i * shard_size:(i + 1) * shard_size]
            for i in range(n_shards)
        ]
        np.random.shuffle(shards)

        # Assign two shards per client
        splits = []
        for i in range(n_clients):
            part = np.concatenate([shards[2*i], shards[2*i + 1]])
            splits.append(part)

    else:
        # IID split
        np.random.shuffle(idx)
        splits = np.array_split(idx, n_clients)

    # --- Build actual Subsets mapped to REAL MNIST indices ---
    clients = []
    for s in splits:
        if isinstance(dataset, Subset):
            # map subset-relative indices -> global MNIST indices
            mapped = subset_idx[s]
            clients.append(Subset(dataset.dataset, mapped))
        else:
            clients.append(Subset(dataset, s))

    return clients





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
