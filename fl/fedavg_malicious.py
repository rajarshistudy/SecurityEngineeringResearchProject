import torch, copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

def local_train(model, dataset, device, epochs=1, batch=64, lr=1e-3, malicious=False):
    mdl = copy.deepcopy(model).to(device)
    opt = optim.Adam(mdl.parameters(), lr=lr)
    dl = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)
    mdl.train()
    for _ in range(epochs):
        for x,y in dl:
            x,y = x.to(device), y.to(device)

            if malicious:
                y = (y + 1) % 10

            opt.zero_grad()
            loss = F.cross_entropy(mdl(x), y)
            loss.backward()
            opt.step()
    return mdl.cpu().state_dict()

def average_state_dicts(state_dicts):
    avg = copy.deepcopy(state_dicts[0])
    for k in avg.keys():
        for sd in state_dicts[1:]:
            avg[k] += sd[k]
        avg[k] /= len(state_dicts)
    return avg
