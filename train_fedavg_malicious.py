import torch
import pandas as pd
from models.mnist_cnn import MNIST_CNN
from utils import set_seed, get_mnist, split_clients, loader, evaluate
from fl.fedavg import local_train, average_state_dicts

def main():
    history = []
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = get_mnist()
    clients = split_clients(train_ds, n_clients=8, noniid=True)
    test_dl = loader(test_ds, shuffle=False)

    global_model = MNIST_CNN()
    ROUNDS, LOCAL_EPOCHS = 20, 1
    # Choose which clients are malicious
    MALICIOUS_IDS = {0} 

    for r in range(ROUNDS):
        states = []

        for cid, cdata in enumerate(clients):
            is_malicious = cid in MALICIOUS_IDS
            sd = local_train(
                global_model, cdata, device,
                epochs=LOCAL_EPOCHS,
                batch=64,
                lr=1e-3,
                malicious=is_malicious
            )
            states.append(sd)
        avg_sd = average_state_dicts(states)
        global_model.load_state_dict(avg_sd)
        acc,_ = evaluate(global_model.to(device), test_dl, device)
        print(f"[FedAvg][Round {r+1:02d}] test_acc={acc:.4f}")
        history.append({"round": r+1, "test_acc": acc})
    pd.DataFrame(history).to_csv("results_fedavg.csv", index=False)
    print("Saved FedAvg history to results_fedavg.csv")
    torch.save(global_model.state_dict(), "fedavg_final.pt")
    print("Saved final FedAvg model to fedavg_final.pt")

if __name__ == "__main__": main()
