import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from Resnet import ResNet18

normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)


# define transforms
valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


I_model = torch.load("../Allen_UROP/data/introspection.txt")

cuda = torch.device("cuda:6")

epochs = 160
batch = 128
# Steps per epoch (CIFAR): 391
cifar10_train = datasets.CIFAR10(root = "../Allen_UROP/datasets", train = True, download = True, transform = train_transform)
cifar10_test = datasets.CIFAR10(root = "../Allen_UROP/datasets", train=False, download = True, transform = valid_transform)
trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch, shuffle= True )
testloader = torch.utils.data.DataLoader(cifar10_test, batch_size = 10000, shuffle = True)

simple_model = ResNet18()
simple_model.to(cuda)
testoptimizer = optim.SGD(simple_model.parameters(),.1, momentum = .9,weight_decay=.0001)
testcriterion = nn.CrossEntropyLoss()

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

def prune(weights, iteration, p, total_pruned, mask):
  # pruned = int((total_size-zeros)*(p**(1/iteration)))
  flat_mask = dict_to_vec(mask)
  prunable_weights = int(torch.sum(flat_mask))
  pruned = int(prunable_weights*(p**(1/iteration)))
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
important_steps = [160*391, 320*391]

num_zeroed = 0
counter = 0
training_iterations = []
accuracy = []
mask_as_list = []
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
      output = simple_model(data)
      loss = testcriterion(output, labels)
      loss.backward()
      testoptimizer.step()
      counter += 1
      for k in range(len(timesteps)):
        if(timesteps[k] == counter):
          x = simple_model.state_dict()
          new_dict = {}
          for item in mask:
            new_dict[item] = x[item].clone().detach()
          weight_holder[k] = dict_to_vec(new_dict)
    correct = 0
    for name, param in simple_model.named_parameters():
        if(name in mask):
            param.data *= mask[name]
    for batch_num, (test_data, test_labels) in enumerate(testloader):
      test_data = test_data.to(cuda)
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
          weight_dict = {item: weight.clone().cpu().detach() for item, weight in simple_model.state_dict().items() if item in prunable_layers} 
          (mask, num_zeroed) = prune(weight_dict, k+1, .6971, num_zeroed, mask)
    temp_dict = {item: weight.clone().cpu().detach() for item, weight in simple_model.state_dict().items() if item in prunable_layers} 
    zeroed = 0
    mask_zeroed = 0
    for item in mask:
      temp_dict[item][temp_dict[item] != 0] = 1
      zeroed += int(torch.sum(temp_dict[item]))
      mask_zeroed += int(torch.sum(mask[item]))
    print(total_size-zeroed, total_size-mask_zeroed)
  print("Max Accuracy:",max_accuracy)

torch.save(accuracy, "../Allen_UROP/data/resnet18_nf_prune_accuracy.txt")
torch.save(training_iterations, "../Allen_UROP/data/resnet18_training_iteration.txt")
torch.save(simple_model.state_dict(), "../Allen_UROP/data/resnet18_nf_prune_model.txt")









