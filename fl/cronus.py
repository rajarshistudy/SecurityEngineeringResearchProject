import torch, copy, numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

@torch.no_grad()
def client_predict_logits(model, public_x, device, batch=128):
    mdl = copy.deepcopy(model).to(device); mdl.eval()
    outs = []
    for i in range(0, len(public_x), batch):
        x = public_x[i:i+batch].to(device)
        outs.append(mdl(x).cpu())
    return torch.cat(outs, dim=0)  # [|Xp|, C]

def aggregate_predictions(logit_list, rule="mean", trim_frac=0.1):
    # logit_list: list of [N,C]
    X = torch.stack(logit_list, dim=0)  # [K,N,C]
    if rule == "mean":
        return X.mean(0)
    if rule == "trimmed-mean":
        # drop extremes along K dimension
        K = X.size(0); k = int(K*trim_frac)
        sorted_vals, _ = torch.sort(X, dim=0)
        kept = sorted_vals[k:K-k] if K > 2*k else sorted_vals
        return kept.mean(0)
    raise ValueError("unknown rule")

def distill_and_finetune(model, private_ds, public_x, public_logits, device, epochs=1, lr=1e-3, alpha=0.5, T=2.0):
    mdl = copy.deepcopy(model).to(device)
    opt = optim.Adam(mdl.parameters(), lr=lr)
    pub_dl = DataLoader(TensorDataset(public_x, public_logits), batch_size=64, shuffle=True)
    prv_dl = DataLoader(private_ds, batch_size=64, shuffle=True)

    def kd_loss(student_logits, teacher_logits, T=2.0):
        # standard KL distillation
        s = torch.log_softmax(student_logits/T, dim=1)
        t = torch.softmax(teacher_logits/T, dim=1)
        return torch.kl_div(s, t, reduction='batchmean') * (T*T)

    for _ in range(epochs):
        mdl.train()
        it = zip(prv_dl, pub_dl)
        for (x_p,y_p),(x_u,log_u) in it:
            x_p,y_p = x_p.to(device), y_p.to(device)
            x_u,log_u = x_u.to(device), log_u.to(device)
            opt.zero_grad()
            loss_sup = F.cross_entropy(mdl(x_p), y_p)
            loss_kd  = kd_loss(mdl(x_u), log_u, T=T)
            loss = (1-alpha)*loss_sup + alpha*loss_kd
            loss.backward(); opt.step()
    return mdl.cpu().state_dict()
