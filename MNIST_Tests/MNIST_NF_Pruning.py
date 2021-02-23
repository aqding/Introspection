import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import torchvision.transforms as transforms

print("Pruning with Forecasting")
epochs = 160
batch = 128
# Steps per epoch: 469
mnist_train = datasets.MNIST(root = "../Allen_UROP/datasets", train = True, download = True, transform = torchvision.transforms.ToTensor())
mnist_test = datasets.MNIST(root = "../Allen_UROP/datasets", train = False, download = True, transform = torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 100, shuffle = True)

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

I_model = torch.load("../Allen_UROP/data/introspection.txt")

cuda = torch.device("cuda:1")

simple_model = nn.Sequential(nn.Linear(28**2, 300),
                             nn.Tanh(),
                             nn.Linear(300, 100),
                             nn.Tanh(),
                             nn.Linear(100, 10),
                             nn.Softmax(dim=1))
simple_model.to(cuda)
testoptimizer = optim.SGD(simple_model.parameters(), .1)
testcriterion = nn.NLLLoss()

def dict_to_vec(model_skel):
  out = torch.tensor([]).to(cuda)
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
  # original = torch.abs(dict_to_vec(conv_layers)).tolist()
  # mergeSort(original)
  # threshold = original[zeros+pruned]
  for item in conv_layers:
      conv_layers[item][torch.abs(conv_layers[item]) > threshold] = 1
      conv_layers[item][conv_layers[item] != 1] = 0
  total_pruned += pruned
  print("Pruning finished", pruned, "weights pruned, for a total of", total_pruned, "weights")
  return (conv_layers,total_pruned)

total_size = torch.reshape(dict_to_vec(simple_model.state_dict()), (-1, )).size()[0]

mask = {}
init_state_dict = simple_model.state_dict()
for item in init_state_dict:
  # init_state_dict[item] = torch.nn.init.normal_(init_state_dict[item], 0, .01)
  mask[item] = torch.ones(init_state_dict[item].size()).to(cuda)


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
  max_accuracy = 0
  for m in range(epochs):
    for batch_num, (data, labels) in enumerate(trainloader):
      testoptimizer.zero_grad()
      data = data.to(cuda)
      labels = labels.to(cuda)
      for name, param in simple_model.named_parameters():
        if(name in mask):
          param.data *= mask[name]
      data = torch.reshape(data, (data.size()[0], 28**2))
      output = simple_model(data)
      loss = testcriterion(output, labels)
      loss.backward()
      testoptimizer.step()
      counter += 1
    correct = 0
    for name, param in simple_model.named_parameters():
        if(name in mask):
            param.data *= mask[name]
    for batch_num, (test_data, test_labels) in enumerate(testloader):
      test_data = test_data.to(cuda)
      test_data = torch.reshape(test_data, (test_data.size()[0], 28**2))
      test_labels = test_labels.to(cuda)
      test_output = simple_model(test_data)
      correct += validate(test_output, test_labels)
    training_iterations.append(counter)
    accuracy.append(correct/10000)
    if(correct/10000 > max_accuracy):
      max_accuracy = correct/10000
    print("Epoch", m, "acc", correct/10000)
    for k in range(len(important_steps)):
      if(important_steps[k] == counter):
          print("Pruning...", k+1)
          weight_dict = {item: weight.clone().cpu().detach() for item, weight in simple_model.state_dict().items()} 
          (mask, num_zeroed) = prune(weight_dict, k+1, .97, num_zeroed, mask)
  print("Max Accuracy:",max_accuracy)

torch.save(accuracy, "../Allen_UROP/data/mnist_nf_prune_accuracy.txt")
torch.save(training_iterations, "../Allen_UROP/data/mnist_training_iteration.txt")
torch.save(simple_model.state_dict(), "../Allen_UROP/data/mnist_nf_prune_model.txt")
