import torch
accuracy = torch.load("../Allen_UROP/data/resnet18_f_prune_accuracy.txt")
accuracy_2 = torch.load("../Allen_UROP/data/resnet18_nf_prune_accuracy.txt")
training_iterations = torch.load("../Allen_UROP/data/resnet18_training_iteration.txt")

import matplotlib.pyplot as plt

#Title will have to be adjusted. Change the line that says "P= ..."

fig , (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Test Accuracy with Weight Pruning, P = 99.5%")
vert = [80*391, 160*391]
for item in vert:
  ax1.plot([item, item], [-1, 1], "g-", alpha = .4)
  ax2.plot([item, item], [-1, 1], "g-", alpha = .4)
ax1.plot(training_iterations, accuracy, "r-")
ax1.set_title("Forecasting")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Iteration")
ax1.set_xlim(0, 240*391)
ax1.set_ylim(0, 1)
ax2.plot(training_iterations, accuracy_2, "r-")
ax2.set_title("No Forecasting")
ax2.set_xlabel("Iteration")
ax2.set_xlim(0, 240*391)
ax2.set_ylim(0, 1)
ax1.grid(True)
ax2.grid(True)
plt.savefig("./Graphs/Resnet18_Results99_40Epochs")