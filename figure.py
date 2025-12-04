import pandas as pd
import matplotlib.pyplot as plt

fed = pd.read_csv("results_fedavg.csv")
cro = pd.read_csv("results_cronus.csv")

plt.figure()
plt.plot(fed["round"], fed["test_acc"], label="FedAvg")
plt.plot(cro["round"], cro["test_acc"], label="Cronus")
plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.legend()
plt.title("FedAvg vs Cronus on MNIST")
plt.savefig("accuracy_comparison.png", dpi=200)
