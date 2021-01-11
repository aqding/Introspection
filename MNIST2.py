import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim


# With updates

epochs = 10
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//(2*epochs), shuffle = False)

def vectorize(matrix):
  temp = torch.tensor([])
  for item in matrix.parameters():
    temp = torch.cat((temp, torch.reshape(item, (-1,))))
  return temp

def validate(output, labels):
  correct = 0
  for entry in (torch.argmax(output, dim = 1) == labels):
    if entry:
      correct += 1
  return correct


mnist_2_testmodel = nn.Sequential(nn.Conv2d(1, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(20, 50, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Flatten(),
                                  nn.Linear(800, 500),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(500, 10),
                                  nn.LogSoftmax())
#Initialize Weights
state_dict = mnist_2_testmodel.state_dict()
for weight in state_dict:
  if(len(state_dict[weight].size()) == 1):
    state_dict[weight] = torch.nn.init.xavier_uniform_(state_dict[weight].view(1, -1))
    state_dict[weight] = state_dict[weight].view((-1,))
  else:
    state_dict[weight] = torch.nn.init.xavier_uniform_(state_dict[weight])
mnist_2_testmodel.load_state_dict(state_dict)

mnist_2_testmodel_optimizer = optim.ASGD(mnist_2_testmodel.parameters(),
                                         lr = .01,
                                         lambd = .0001,
                                         alpha = .75)
mnist_2_testmodel_criterion = nn.NLLLoss()

time_zero_weights = vectorize(mnist_2_testmodel)
first_update = torch.zeros((4, time_zero_weights.size()[0]))
second_update = torch.zeros((4, time_zero_weights.size()[0]))
first_update[3] = time_zero_weights
second_update[3] = time_zero_weights

timestep_dict = {2500:(first_update, 0),
                 1750:(first_update, 1),
                 1000:(first_update, 2),
                 3000:(second_update, 0),
                 2100:(second_update, 1),
                 1200:(second_update, 2)}

xdata = []
ydata_updates = []

timestep = 0
total_correct = 0
for epoch in range(epochs):
  train_iter = iter(trainloader)
  for batch in train_iter:
    for key in timestep_dict.keys():
      if(key == timestep):
        timestep_dict[timestep][0][timestep_dict[timestep][1]] = vectorize(mnist_2_testmodel)
        if(timestep_dict[timestep][1] == 0):
          start = 0
          state_dict = mnist_2_testmodel.state_dict()
          new_parameters = I_model(torch.transpose(timestep_dict[timestep][0], 0, 1)*1000)/1000
          for item in state_dict:
            dimensions = state_dict[item].size()
            total = torch.prod(torch.tensor(dimensions))
            state_dict[item] = new_parameters[start:start+total].view(dimensions)
            start += total
          mnist_2_testmodel.load_state_dict(state_dict)
    if(timestep % 500 == 0 and timestep != 0):
      data, labels = iter(testloader).next()
      valid = validate(mnist_2_testmodel(data), labels)/(10000/(2*epochs))
      print("Step", timestep, "Training Accuracy:", total_correct/(64*500),
            "Test Accuracy:", valid)
      xdata.append(timestep)
      ydata_updates.append(valid)
      total_correct = 0
    mnist_2_testmodel_optimizer.zero_grad()
    data, labels = batch
    output = mnist_2_testmodel(data)
    loss = mnist_2_testmodel_criterion(output, labels)
    loss.backward()
    mnist_2_testmodel_optimizer.step()
    total_correct += validate(output, labels)
    timestep += 1


# Without Updates

mnist_2_testmodel = nn.Sequential(nn.Conv2d(1, 20, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(20, 50, (5, 5)),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Flatten(),
                                  nn.Linear(800, 500),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(500, 10),
                                  nn.LogSoftmax())
#Initialize Weights
state_dict = mnist_2_testmodel.state_dict()
for weight in state_dict:
  if(len(state_dict[weight].size()) == 1):
    state_dict[weight] = torch.nn.init.xavier_uniform_(state_dict[weight].view(1, -1))
    state_dict[weight] = state_dict[weight].view((-1,))
  else:
    state_dict[weight] = torch.nn.init.xavier_uniform_(state_dict[weight])
mnist_2_testmodel.load_state_dict(state_dict)

mnist_2_testmodel_optimizer = optim.ASGD(mnist_2_testmodel.parameters(),
                                         lr = .01,
                                         lambd = .0001,
                                         alpha = .75)
mnist_2_testmodel_criterion = nn.NLLLoss()


ydata_no_updates = []

timestep = 0
total_correct = 0
for epoch in range(epochs):
  train_iter = iter(trainloader)
  for batch in train_iter:
    if(timestep % 500 == 0 and timestep != 0):
      data, labels = iter(testloader).next()
      valid = validate(mnist_2_testmodel(data), labels)/(10000/(2*epochs))
      print("Step", timestep, "Training Accuracy:", total_correct/(64*500),
            "Test Accuracy:", valid)
      ydata_no_updates.append(valid)
      total_correct = 0
    mnist_2_testmodel_optimizer.zero_grad()
    data, labels = batch
    output = mnist_2_testmodel(data)
    loss = mnist_2_testmodel_criterion(output, labels)
    loss.backward()
    mnist_2_testmodel_optimizer.step()
    total_correct += validate(output, labels)
    timestep += 1


# Graphing Data

import matplotlib.pyplot as plt
plt.plot(xdata, ydata_updates, "r-", xdata, ydata_no_updates, "b-")
timesteps = [2500, 3000]
for item in timesteps:
  plt.plot([item, item], [0, 1], "g-", alpha = .4)
plt.title("MNIST 2")
plt.ylabel("Validation Accuracy")
plt.xlabel("Training Step")
plt.axis([0, 9000, .85, 1])
plt.legend(["With Introspection", "Without Introspection"])
plt.grid(True)