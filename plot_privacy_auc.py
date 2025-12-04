import matplotlib.pyplot as plt


fedavg_auc = 0.490753173828125      
cronus_auc = 0.4784984588623047      

models = ["FedAvg", "Cronus"]
auc_values = [fedavg_auc, cronus_auc]
colors = ["red", "blue"]

plt.figure(figsize=(7,5))
plt.bar(models, auc_values, color=colors)

plt.ylim(0.0, 1.0)
plt.ylabel("Membership Inference AUC")
plt.title("Privacy Leakage Comparison\nFedAvg vs Cronus")

for i, v in enumerate(auc_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)

plt.tight_layout()
plt.savefig("privacy_auc_fedavg_vs_cronus.png", dpi=200)
print("Saved privacy_auc_fedavg_vs_cronus.png")
plt.show()
