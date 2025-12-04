import pandas as pd
import matplotlib.pyplot as plt
import os

def load_csv_safe(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"ERROR: Could not find {filename}. "
                                f"Make sure you ran training scripts with poisoning enabled.")
    return pd.read_csv(filename)

def main():
    # Load poisoned results
    fed = load_csv_safe("results_fedavg.csv")      # run after poisoning FedAvg
    cro = load_csv_safe("results_cronus.csv")      # run after poisoning Cronus

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(fed["round"], fed["test_acc"], 
             label="FedAvg (poisoned)", linewidth=2, color='red')
    plt.plot(cro["round"], cro["test_acc"], 
             label="Cronus (poisoned)", linewidth=2, color='blue')

    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Robustness Under Malicious Client Attack\nFedAvg vs Cronus", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save figure
    plt.savefig("poisoned_fedavg_vs_cronus.png", dpi=200)
    print("Saved poisoned_fedavg_vs_cronus.png")

if __name__ == "__main__":
    main()
