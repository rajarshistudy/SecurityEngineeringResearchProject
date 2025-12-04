import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_cronus.csv")

plt.plot(df["round"], df["test_acc"], label="Cronus")
plt.xlabel("Round")
plt.ylabel("Test Accuracy")
plt.title("Cronus Accuracy vs Rounds")
plt.legend()
plt.grid()
plt.savefig("cronus_accuracy.png", dpi=200)
print("Saved cronus_accuracy.png")
