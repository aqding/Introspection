import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import time


epochs = 2


mnist_train = datasets.MNIST(root = "./data", train = True, download = True, transform = torchvision.transforms.ToTensor())
mnist_test = datasets.MNIST(root = "./data", train = False, download = True, transform = torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = 50, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = True)
test_iter = iter(testloader)
model = nn.Sequential(nn.Conv2d(1, 8, (5, 5), padding = 2),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(8, 16, (5, 5), padding = 2),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(16, 32, (5, 5), padding = 2),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2, padding = 1),
                      nn.Flatten(),
                      nn.Linear(512, 1024),
                      nn.Dropout(),
                      nn.ReLU(),
                      nn.Linear(1024, 10),
                      nn.LogSoftmax(dim = 1))

def vectorize(matrix):
  temp = torch.tensor([])
  for item in matrix.parameters():
    temp = torch.cat((temp, torch.reshape(item, (-1,))))
  return temp


weight_library = torch.zeros((1200*epochs + 1, 551818))
optimizer = optim.Adam(model.parameters(), .0001)
criterion = nn.NLLLoss()
weight_library[0] = vectorize(model)
counter = 0
#start = time.time()
for epoch in range(epochs):
  running_loss = 0
  correct = 0
  vrunning_loss = 0
  vcorrect = 0
  for batch in iter(trainloader):
    counter+=1
    # if(counter % 100 == 0 and counter != 0):
    #   end = time.time()
    #   print(counter, end-start)
    #   start = time.time()
    optimizer.zero_grad()
    data, labels = batch
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    weight_library[counter] = vectorize(model)
    compare = torch.argmax(output, dim = 1) == labels
    for entry in compare:
      if entry:
        correct += 1
  optimizer.zero_grad()
  data, labels = test_iter.next()
  output = model(data)
  loss = criterion(output, labels)
  loss.backward()
  vrunning_loss += loss.item()
  compare = torch.argmax(output, dim = 1) == labels
  for entry in compare:
    if entry:
        vcorrect += 1
  print("Epoch", epoch+1, "Training Error:", running_loss/(60000/50), "Training Accuracy:", correct/60000)
  print("Epoch", epoch+1, "Testing Error:", vrunning_loss, "Testing Accuracy:", vcorrect/10000*epochs)
    
print(weight_library.size())


#Generating Data
import random

samples = 3
weights = weight_library #torch.load("weights")
data = torch.zeros((weights.size()[1]*samples, 5))

for datapoint in range(weights.size()[1]):
  for i in range(samples):
    timestep = random.randint(10, weights.size()[0]//2)
    data[datapoint*3 + i] = torch.tensor([weights[timestep][datapoint],
                                  weights[int(timestep*7/10)][datapoint], 
                                  weights[int(timestep*4/10)][datapoint],
                                  weights[0][datapoint],
                                  weights[2*timestep][datapoint]])
print("finished", data.size())
info = data*1000

# Training I

dataset = torch.load("info")
I_data = dataset
I_data = I_data[torch.randperm(I_data.size()[0])]

I_model = nn.Sequential(nn.Linear(4, 40),
                        nn.ReLU(),
                        nn.Linear(40, 1))
print(data.size())
optimizer = optim.Adam(I_model.parameters(), .0005)
criterion = nn.L1Loss()



running_loss = 0
I_data = I_data[torch.randperm(I_data.size()[0])]
for step in range(data.size()[0]//20):
  if(step % 1000 == 0 and step!= 0):
    print("Step:", step, "Running Loss:", running_loss/1000)
    running_loss = 0
  optimizer.zero_grad()
  output = I_model(I_data[step*20:step*20+20, :4])
  loss = criterion(output, I_data[step*20:step*20+20, 4:])
  loss.backward()
  optimizer.step()
  running_loss += 20*loss.item()
# print(I_model(I_data[:1, :4]))
# torch.save(I_model.state_dict(), "I")
print("Finished! Running Loss:", running_loss/1000)

# Predicting Masks with Forecasting

print("Pruning with Forecasting")
epochs = 40
batch = 128
# Steps per epoch: 469
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000, shuffle = True)

# Unused Model

# simple_model = nn.Sequential(nn.Conv2d(1, 6, 5, padding = 2),
#                              nn.ReLU(),
#                              nn.AvgPool2d(2, 2),
#                              nn.Conv2d(6, 16, 5),
#                              nn.ReLU(),
#                              nn.AvgPool2d(2, 2),
#                              nn.Conv2d(16, 120, 5)
#                              nn.ReLU(),
#                              nn.Flatten()
#                              nn.Linear(120, 84),
#                              nn.ReLU(),
#                              nn.Linear(84, 10),
#                              nn.Softmax()
#                              )

# Using Lenet_300_100 from LTH experiment

simple_model = nn.Sequential(nn.Linear(28**2, 300),
                             nn.Tanh(),
                             nn.Linear(300, 100),
                             nn.Tanh(),
                             nn.Linear(100, 10),
                             nn.Softmax(dim=1))

testoptimizer = optim.SGD(simple_model.parameters(), .1)
testcriterion = nn.NLLLoss()

def dict_to_vec(model_skel):
  out = torch.tensor([])
  for item in model_skel:
    out = torch.cat((out, torch.reshape(model_skel[item], (-1,))))
  return out

def vec_to_dict(vec, model_skel):
  temp_dict = {}
  current = 0
  for item in model_skel:
    current_size = torch.prod(torch.tensor(model_skel[item].size()))
    temp_dict[item] = torch.reshape(vec[current:current+current_size], model_skel[item].size())
    current += current_size
  return temp_dict

def validate(output, labels):
  correct = 0
  for entry in (torch.argmax(output, dim = 1) == labels):
    if entry:
      correct += 1
  return correct

def mergeSort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        mergeSort(L)
        mergeSort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def prune(weights, iteration, p, zeros, mask):
  pruned = int((total_size-zeros)*(p**iteration))
  original = dict_to_vec(weights)
  flat_mask = dict_to_vec(mask)
  vec = torch.abs(original.clone()).tolist()
  vec2 = torch.abs(original.clone()).tolist()
  mergeSort(vec)
  for number in vec[zeros:zeros+pruned]:
    index = vec2.index(number)
    flat_mask[index] = 0
  zeros += pruned
  return (vec_to_dict(flat_mask, mask),zeros)

total_size = torch.reshape(dict_to_vec(simple_model.state_dict()), (-1, )).size()[0]

mask = {}
init_state_dict = simple_model.state_dict()
for item in init_state_dict:
  # init_state_dict[item] = torch.nn.init.normal_(init_state_dict[item], 0, .01)
  mask[item] = torch.ones(init_state_dict[item].size())
simple_model.load_state_dict(init_state_dict)

# Precomputed values, corresponds to the number of steps for 40/80 epochs, respectively
important_steps = [18760, 37520]
timesteps = []
for step in important_steps:
  timesteps.append(step)
  timesteps.append(step*7//10)
  timesteps.append(step*4//10)
  timesteps.append(0)

weight_holder = torch.zeros((len(timesteps), total_size))
for i in range(len(timesteps)):
  if(i+1 % 4 == 0):
    weight_holder[i] = dict_to_vec(init_state_dict)

print("Pruning without Forecasting")
num_zeroed = 0
counter = 0
trajectory = {}
training_iterations = []
accuracy = []
inserts = 0
for i in range (3):
  print("Pruning Step", i+1)
  # testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = True)
  # iter_test = iter(testloader)
  for m in range(epochs):
    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch, shuffle = True)
    iter_train = iter(trainloader)
    for j in range(60000//batch + 1):
      testoptimizer.zero_grad()
      data, labels = iter_train.next()
      data = torch.reshape(data, (data.size()[0], 28**2))
      output = simple_model(data)
      loss = testcriterion(output, labels)
      loss.backward()
      for parameter in simple_model.parameters():
        for item in mask:
          if(parameter.size() == mask[item].size()):
            parameter.grad = parameter.grad*mask[item]
      testoptimizer.step()
      counter += 1
      if(counter % 1000 == 0):
        trajectory[inserts] = dict_to_vec(simple_model.state_dict())
        inserts += 1
      for k in range(len(timesteps)):
        if(timesteps[k] == counter):
          x = simple_model.state_dict()
          weight_holder[k] = dict_to_vec(x)
      for k in range(len(important_steps)):
        if(important_steps[k] == counter):
          print("Pruning...", k+1)
          acceleration = I_model(torch.transpose(weight_holder[4*k:4*k+4, :], 0, 1)*1000)/1000
          accelerate_dict = vec_to_dict(acceleration, simple_model.state_dict())
          (mask, num_zeroed) = prune(accelerate_dict, k+1, .95, num_zeroed, mask)
          temp_state_dict = simple_model.state_dict()
          for item in mask:
            temp_state_dict[item] = mask[item] * temp_state_dict[item]
          simple_model.load_state_dict(temp_state_dict)
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000, shuffle = True)
    iter_test = iter(testloader)
    test_data, test_labels = iter_test.next()
    test_data = torch.reshape(test_data, (test_data.size()[0], 28**2))
    test_output = simple_model(test_data)
    correct = validate(test_output, test_labels)
    training_iterations.append(counter)
    accuracy.append(correct/10000)
    print("Epoch", m, "acc", correct/10000)


# Using same step masks
simple_model_2 = nn.Sequential(nn.Linear(28**2, 300),
                             nn.Tanh(),
                             nn.Linear(300, 100),
                             nn.Tanh(),
                             nn.Linear(100, 10),
                             nn.Softmax(dim=1))

testoptimizer = optim.SGD(simple_model_2.parameters(), .1)
testcriterion = nn.NLLLoss()

mask = {}
init_state_dict = simple_model_2.state_dict()
for item in init_state_dict:
  # init_state_dict[item] = torch.nn.init.normal_(init_state_dict[item], 0, .01)
  mask[item] = torch.ones(init_state_dict[item].size())
simple_model_2.load_state_dict(init_state_dict)

# Precomputed values, corresponds to the number of steps for 40/80 epochs, respectively
important_steps = [18760, 37520]

num_zeroed = 0
counter = 0
trajectory_2 = {}
accuracy_2 = []
inserts = 0
for i in range (3):
  print("Pruning Step", i+1)
  # testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = True)
  # iter_test = iter(testloader)
  for m in range(epochs):
    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch, shuffle = True)
    iter_train = iter(trainloader)
    for j in range(60000//batch + 1):
      testoptimizer.zero_grad()
      data, labels = iter_train.next()
      data = torch.reshape(data, (data.size()[0], 28**2))
      output = simple_model_2(data)
      loss = testcriterion(output, labels)
      loss.backward()
      for parameter in simple_model_2.parameters():
        for item in mask:
          if(parameter.size() == mask[item].size()):
            parameter.grad = parameter.grad*mask[item]
      testoptimizer.step()
      counter += 1
      if(counter % 1000 == 0):
        trajectory_2[inserts] = dict_to_vec(simple_model_2.state_dict())
        inserts += 1
      for k in range(len(important_steps)):
        if(important_steps[k] == counter):
          print("Pruning...", k+1)
          this_state_dict = simple_model_2.state_dict()
          (mask, num_zeroed) = prune(this_state_dict, k+1, .98, num_zeroed, mask)
          temp_state_dict = simple_model_2.state_dict()
          for item in mask:
            temp_state_dict[item] = mask[item] * temp_state_dict[item]
          simple_model_2.load_state_dict(temp_state_dict)
    testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000, shuffle = True)
    iter_test = iter(testloader)
    test_data, test_labels = iter_test.next()
    test_data = torch.reshape(test_data, (test_data.size()[0], 28**2))
    test_output = simple_model_2(test_data)
    correct = validate(test_output, test_labels)
    accuracy_2.append(correct/10000)
    print("Epoch", m, "acc", correct/10000)


# Graphing Accuracy
import matplotlib.pyplot as plt

fig , (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Test Accuracy with Weight Pruning, P = 98%")
vert = [18760, 37520]
for item in vert:
  ax1.plot([item-500, item-500], [-1, 1], "g-", alpha = .4)
  ax2.plot([item-500, item-500], [-1, 1], "g-", alpha = .4)
ax1.plot(training_iterations, accuracy, "r-")
ax1.set_title("Forecasting")
ax1.set_ylabel("Accuracy")
ax1.set_xlabel("Iteration")
ax1.set_xlim(0, 57900)
ax1.set_ylim(.7, 1)
ax2.plot(training_iterations, accuracy_2, "r-")
ax2.set_title("No Forecasting")
ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Iteration")
ax2.set_xlim(0, 57900)
ax2.set_ylim(.7, 1)
ax1.grid(True)
ax2.grid(True)