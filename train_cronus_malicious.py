import torch
import pandas as pd
from torch.utils.data import Subset
from models.mnist_cnn import MNIST_CNN
from utils import set_seed, get_mnist, split_clients, loader, evaluate
from fl.cronus import client_predict_logits, aggregate_predictions, distill_and_finetune
from fl.fedavg import average_state_dicts

def build_public_pool(train_ds, size=2000):
    idx = torch.randperm(len(train_ds))[:size].tolist()
    Xp = torch.stack([train_ds[i][0] for i in idx], dim=0)
    return Xp, set(idx)

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_ds, test_ds = get_mnist()

    # Build public pool Xp
    Xp, public_idx = build_public_pool(train_ds, size=2000)

    # Remove public pool from training data
    remaining_idx = [i for i in range(len(train_ds)) if i not in public_idx]
    train_ds_rem = Subset(train_ds, remaining_idx)

    clients = split_clients(train_ds_rem, n_clients=8, noniid=True)
    test_dl = loader(test_ds, shuffle=False)

    global_model = MNIST_CNN()
    MALICIOUS_IDS = {0}
    ROUNDS = 20

    # History list for saving results
    history = []

    for r in range(ROUNDS):

        # 1. Clients compute predictions on public data
        logits_list = []
        for cdata in clients:
            logits = client_predict_logits(global_model, Xp, device)
            logits_list.append(logits)

        # 2. Server aggregates predictions
        Ybar = aggregate_predictions(logits_list, rule="mean")

        # 3. Clients distill and fine-tune locally
        states = []
        for cid, cdata in enumerate(clients):
            is_malicious = cid in MALICIOUS_IDS

            sd = distill_and_finetune(
                global_model, cdata,
                Xp, Ybar,
                device,
                epochs=1, lr=1e-3,
                malicious=is_malicious
            )
            states.append(sd)

        # 4. Server averages client-updated models
        avg_sd = average_state_dicts(states)
        global_model.load_state_dict(avg_sd)

        # 5. Evaluate the global model
        acc, _ = evaluate(global_model.to(device), test_dl, device)
        print(f"[Cronus][Round {r+1:02d}] test_acc={acc:.4f}")

        # 6. Store accuracy in history
        history.append({"round": r+1, "test_acc": acc})

    # AFTER all rounds → save logs + final model
    pd.DataFrame(history).to_csv("results_cronus.csv", index=False)
    print("Saved Cronus history to results_cronus.csv")

    torch.save(global_model.state_dict(), "cronus_final.pt")
    print("Saved final Cronus model to cronus_final.pt")

if __name__ == "__main__":
    main()
