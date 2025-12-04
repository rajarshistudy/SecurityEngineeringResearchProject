import torch, copy, numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

@torch.no_grad()
def kd_loss(student_logits, teacher_logits, T=2.0):
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)

    # emulate reduction='batchmean'
    kl = F.kl_div(s, t, reduction='sum')
    return kl / student_logits.size(0) * (T * T)
def client_predict_logits(model, Xp, device):
    model.eval()
    logits_out = []

    with torch.no_grad():
        for i in range(0, len(Xp), 64):
            batch = Xp[i:i+64].to(device)
            logits = model(batch)        # shape [batch, 10]
            logits_out.append(logits.cpu())

    return torch.cat(logits_out, dim=0)

def aggregate_predictions(logits_list, rule="mean", trim_frac=0.1):
    # logits_list: list of [N,C]
    X = torch.stack(logits_list, dim=0)  # [K,N,C]

    if rule == "mean":
        return X.mean(0)

    if rule == "trimmed-mean":
        K = X.size(0)
        k = int(K * trim_frac)
        sorted_vals, _ = torch.sort(X, dim=0)
        kept = sorted_vals[k:K-k] if K > 2*k else sorted_vals
        return kept.mean(0)

    raise ValueError("unknown rule")

def distill_and_finetune(global_model, client_dataset, Xp, Ybar, device,
                         epochs=1, lr=1e-3, malicious=False):

    mdl = deepcopy(global_model).to(device)
    mdl.train()
    opt = optim.Adam(mdl.parameters(), lr=lr)

    dl_p = DataLoader(TensorDataset(Xp, Ybar), batch_size=64, shuffle=True)
    dl_u = DataLoader(client_dataset, batch_size=64, shuffle=True)

    T = 2.0
    alpha = 1.0

    for _ in range(epochs):
        for (x_p, y_p), (x_u, y_u) in zip(dl_p, dl_u):
            x_p, y_p = x_p.to(device), y_p.to(device)
            x_u, y_u = x_u.to(device), y_u.to(device)

            opt.zero_grad()

            # Poison attack
            if malicious:
                y_u = (y_u + 1) % 10

            # KD on PUBLIC pool
            loss_kd = kd_loss(mdl(x_p), y_p, T=T)

            # supervised on client PRIVATE data
            loss_sup = F.cross_entropy(mdl(x_u), y_u)

            loss = loss_kd + alpha * loss_sup
            loss.backward()
            opt.step()

    return mdl.state_dict()
