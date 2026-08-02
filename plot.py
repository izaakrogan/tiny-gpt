"""Plots every losses_*.csv in the directory onto one chart, so you can
compare the models you've trained so far. Solid line is val loss, dashed
is train.
"""

import csv
import glob

import matplotlib.pyplot as plt

for path in sorted(glob.glob("losses_*.csv")):
    model = path[len("losses_"):-len(".csv")]
    iters, train, val = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    line, = plt.plot(iters, val, label=model)
    plt.plot(iters, train, linestyle="--", color=line.get_color())

plt.xlabel("iteration")
plt.ylabel("loss (-log p)")
plt.legend()
plt.tight_layout()
plt.savefig("loss.png", dpi=150)
print("wrote loss.png")
