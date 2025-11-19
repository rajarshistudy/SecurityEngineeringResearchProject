import torch, torch.optim as optim, torch.nn.functional as F
from models.mnist_cnn import MNIST_CNN
from utils import set_seed, get_mnist, loader, evaluate
from tqdm import trange

def train_epoch(model, dl, opt, device):
    model.train()
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward(); opt.step()

def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = get_mnist()
    train_dl = loader(train_ds); test_dl = loader(test_ds, shuffle=False)
    model = MNIST_CNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for _ in trange(5, desc="centralized epochs"):
        train_epoch(model, train_dl, opt, device)
        acc, _ = evaluate(model, test_dl, device)
        print(f"test_acc={acc:.4f}")

if __name__ == "__main__": main()
