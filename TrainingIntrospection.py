# Training on MNIST
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import time
import random

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
#torch.save(weight_library, "weights")

#Generating Data

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

I_data = dataset
I_data = I_data[torch.randperm(I_data.size()[0])]

I_model = nn.Sequential(nn.Linear(4, 40),
                        nn.ReLU(),
                        nn.Linear(40, 1))
print(data.size())
optimizer = optim.Adam(I_model.parameters(), .0005)
criterion = nn.L1Loss()

#Training Introspection

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

torch.save(I_model, "I_model")

