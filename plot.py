import csv

import matplotlib.pyplot as plt

iters, train, val = [], [], []
with open("losses.csv") as f:
    for row in csv.DictReader(f):
        iters.append(int(row["iter"]))
        train.append(float(row["train_loss"]))
        val.append(float(row["val_loss"]))

plt.plot(iters, train, label="train")
plt.plot(iters, val, label="val")
plt.xlabel("iteration")
plt.ylabel("loss (-log p)")
plt.legend()
plt.tight_layout()
plt.savefig("loss.png", dpi=150)
print("wrote loss.png")
