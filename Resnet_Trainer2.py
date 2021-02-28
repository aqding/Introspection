#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import torchvision.transforms as transforms
from resnet_model import *
#from resnet_method import *
import torch.backends.cudnn as cudnn
import numpy

# In[ ]:


# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


# In[ ]:


# model_names = sorted(name for name in __all__
#     if name.islower() and not name.startswith("__")
#                      and name.startswith("resnet")
#                      and callable(resnet.__all__[name]))

# print(model_names)


# In[ ]:
TOTAL_PRUNED = .95
prune_percent = numpy.roots([-1, 1, 1, -TOTAL_PRUNED])[2]

I_model = torch.load("../Allen_UROP/data/introspection.txt")
cuda = torch.device("cuda:1")


# In[ ]:


epochs = 80
batch = 128 #128
test_batch = 100

device = "cuda:1"

# In[ ]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='../Allen_UROP/datasets', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='../Allen_UROP/datasets', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_batch, shuffle=False, num_workers=2)


# In[ ]:


# Steps per epoch (CIFAR): 391
# simple_model = resnet20() #torchvision.models.resnet18()

simple_model = ResNet18() #torch.nn.DataParallel(resnet.__dict__['resnet20']())

simple_model = simple_model.to(device)
if device == 'cuda:1':
    simple_model = torch.nn.DataParallel(simple_model)
    cudnn.benchmark = True
    
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(simple_model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                     #                   milestones=[100, 150], last_epoch=- 1)


# In[ ]:


def dict_to_vec(model_skel):
  out = torch.tensor([]).to(cuda)
  for item in model_skel:
    model_skel[item] = model_skel[item].to(cuda)
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


# In[ ]:


def prune(weights, iteration, p, total_pruned, mask):
    flat_mask = dict_to_vec(mask)
    prunable_weights = int(torch.sum(flat_mask))
    pruned = int(prunable_weights*(p**(iteration)))
    listed_weights = torch.cat([item[mask[name] == 1] for name, item in weights.items()])
    threshold = torch.sort(torch.abs(listed_weights))[0][pruned]
    conv_layers = {}
    for item in mask:
        conv_layers[item] = weights[item]
        conv_layers[item] = conv_layers[item].to(cuda)

    for item in conv_layers:
        conv_layers[item][torch.abs(conv_layers[item]) > threshold] = 1
        conv_layers[item][conv_layers[item] != 1] = 0
  	
    total_pruned += pruned
  
    print("Pruning finished", pruned, "weights pruned, for a total of", total_pruned, "weights")
    return (conv_layers,total_pruned)


# In[ ]:


prunable_layers = [name + '.weight' for name, module in simple_model.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]
mask = {}
init_state_dict = simple_model.state_dict()
for item in init_state_dict:
  if(item in prunable_layers):
    mask[item] = torch.ones(init_state_dict[item].size()).to(cuda)


total_size = torch.reshape(dict_to_vec(mask), (-1, )).size()[0]
total_model_size = torch.reshape(dict_to_vec(simple_model.state_dict()), (-1, )).size()[0]
print(total_model_size, total_size)

# Precomputed values, corresponds to the number of steps for 40/80 epochs, respectively
important_steps = [80*391] #, 160*391]
timesteps = []
for step in important_steps:
  timesteps.append(step)
  timesteps.append(step*7//10)
  timesteps.append(step*4//10)
  timesteps.append(0)

weight_holder = torch.zeros((len(timesteps), total_size))
for i in range(len(timesteps)):
  if(i+1 % 4 == 0):
    new_dict = {}
    for item in mask:
      new_dict[item] = init_state_dict[item]
    weight_holder[i] = dict_to_vec(new_dict)


# In[ ]:


num_zeroed = 0
counter = 0
training_iterations = []
accuracy = []
mask_as_list = []
inserts = 0


# In[ ]:


for i in range (3):
  max_accuracy = 0
  print("Pruning Step", i+1)
  for m in range(epochs):
    for batch_num, (data, labels) in enumerate(trainloader):
      optimizer.zero_grad()
      data = data.to(cuda)
      labels = labels.to(cuda)
      for name, param in simple_model.named_parameters():
          if(name in mask):
              param.data *= mask[name]
      output = simple_model(data)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      
      #state_dict = simple_model.state_dict()
      #for item in mask:
      #  state_dict[item] = state_dict[item].to(cuda)*mask[item]
      #simple_model.load_state_dict(state_dict)
        
      counter += 1
      for k in range(len(timesteps)):
        if(timesteps[k] == counter):
          x = simple_model.state_dict()
          new_dict = {}
          for item in mask:
            new_dict[item] = x[item]
          weight_holder[k] = dict_to_vec(new_dict)
    #testloader = torch.utils.data.DataLoader(cifar10_test, batch_size = test_batch, shuffle = True)
    correct = 0
    for batch_num, (test_data, test_labels) in enumerate(testloader):
        test_data = test_data.to(cuda)
        test_labels = test_labels.to(cuda)
        test_output = simple_model(test_data)
        correct += validate(test_output, test_labels)
    training_iterations.append(counter)
    accuracy.append(correct/10000)
    print("Epoch", m, "acc", correct/10000)
    
    # step scheduler
    lr_scheduler.step()
    for k in range(len(important_steps)):
        if(important_steps[k] == counter):
            print("Pruning", k+1)
            acceleration = I_model(torch.transpose(weight_holder[4*k:4*k+4, :], 0, 1)*1000)/1000
            accelerate_dict = vec_to_dict(acceleration, mask)
            (mask, num_zeroed) = prune(accelerate_dict, k+1, prune_percent, num_zeroed, mask)
  print("Max Accuracy:", max_accuracy)



# In[ ]:




