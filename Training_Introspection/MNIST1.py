import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim

# With Updates

epochs = 20
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = 50, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = False)

testmodel = nn.Sequential(nn.Conv2d(1, 32, 5, padding =  2),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Conv2d(32, 64, 5, padding = 2),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Flatten(),
                          nn.Linear(3136, 1024),
                          nn.Dropout(),
                          nn.ReLU(),
                          nn.Linear(1024, 10),
                          nn.LogSoftmax()
                          )
 

testoptimizer = optim.SGD(testmodel.parameters(), .01)
testcriterion = nn.NLLLoss()

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

time_zero_params = vectorize(testmodel)
first_update = torch.zeros((4, time_zero_params.size()[0]))
second_update = torch.zeros((4, time_zero_params.size()[0]))
third_update = torch.zeros((4, time_zero_params.size()[0]))
first_update[3] = time_zero_params
second_update[3] = time_zero_params
third_update[3] = time_zero_params

counter = 0
total_correct = 0
timestep_dict = {1200:(first_update, 2),
             2100:(first_update, 1),
             3000:(first_update, 0),
             1600:(second_update, 2),
             2800:(second_update, 1),
             4000:(second_update, 0),
             2000:(third_update, 2),
             3500:(third_update, 1),
             5000:(third_update, 0)}



print(testmodel.state_dict().keys())

state_dict = testmodel.state_dict()
for item in state_dict:
  state_dict[item] = torch.nn.init.normal_(state_dict[item], 0, .01)
testmodel.load_state_dict(state_dict)
print("Initialized")


xdata = []
ydata_updates = []

for epoch in range(epochs):
  train_iter = iter(trainloader)
  for batch in train_iter:
    for timestep in timestep_dict.keys():
      if(counter == timestep):
        timestep_dict[counter][0][timestep_dict[counter][1]] = vectorize(testmodel)
        if(timestep_dict[counter][1] == 0):
          start = 0
          new_parameters = I_model(torch.transpose(timestep_dict[counter][0], 0, 1)*1000)/1000
          state_dict = testmodel.state_dict()
          for item in state_dict:
            dimensions = state_dict[item].size()
            total = torch.prod(torch.tensor(dimensions))
            state_dict[item] = new_parameters[start:start+total].view(dimensions)
            start += total
          testmodel.load_state_dict(state_dict)
          print("Weights updated")
    if(counter % 1000 == 0 and counter != 0):
      test_data, test_labels = iter(testloader).next()
      acc = validate(testmodel(test_data), test_labels)/(10000//epochs)
      print("Step", counter, "Training Accuracy", total_correct/(1000*50), 
            "Test Accuracy", acc)
      xdata.append(counter)
      ydata_updates.append(acc)
      total_correct = 0
    testoptimizer.zero_grad()
    data, labels = batch
    output = testmodel(data)
    loss = testcriterion(output, labels)
    loss.backward()
    testoptimizer.step()
    total_correct += validate(output, labels)
    counter += 1

# Without Updates

testmodel = nn.Sequential(nn.Conv2d(1, 32, 5, padding =  2),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Conv2d(32, 64, 5, padding = 2),
                          nn.ReLU(),
                          nn.MaxPool2d(2, 2),
                          nn.Flatten(),
                          nn.Linear(3136, 1024),
                          nn.Dropout(),
                          nn.ReLU(),
                          nn.Linear(1024, 10),
                          nn.LogSoftmax()
                          )
 

testoptimizer = optim.SGD(testmodel.parameters(), .01)
testcriterion = nn.NLLLoss()



print(testmodel.state_dict().keys())

state_dict = testmodel.state_dict()
for item in state_dict:
  state_dict[item] = torch.nn.init.normal_(state_dict[item], 0, .01)
testmodel.load_state_dict(state_dict)
print("Initialized")

ydata_no_updates = []
for epoch in range(epochs):
  train_iter = iter(trainloader)
  for batch in train_iter:
    if(counter % 1000 == 0 and counter != 0):
      test_data, test_labels = iter(testloader).next()
      acc = validate(testmodel(test_data), test_labels)/(10000//epochs)
      print("Step", counter, "Training Accuracy", total_correct/(1000*50), 
            "Test Accuracy", acc)
      ydata_no_updates.append(acc)
      total_correct = 0
    testoptimizer.zero_grad()
    data, labels = batch
    output = testmodel(data)
    loss = testcriterion(output, labels)
    loss.backward()
    testoptimizer.step()
    total_correct += validate(output, labels)
    counter += 1

#Graphing Data

import matplotlib.pyplot as plt
plt.plot(xdata, ydata_updates, "r-", xdata, ydata_no_updates, "b-")
timesteps = [3000, 4000, 5000]
for item in timesteps:
  plt.plot([item, item], [0, 1], "g-", alpha = .4)
plt.title("MNIST 1")
plt.ylabel("Validation Accuracy")
plt.xlabel("Training Step")
plt.axis([0, 23000, .85, 1])
plt.legend(["With Introspection", "Without Introspection"])
plt.grid(True)
    