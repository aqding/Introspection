import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms

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

cuda = torch.device("cuda:0")

epochs = 120
batch = 128
# Steps per epoch (CIFAR): 391
cifar10_train = datasets.CIFAR10(root = "../Allen_UROP/datasets", train = True, download = True, transform = train_transform)
cifar10_test = datasets.CIFAR10(root = "../Allen_UROP/datasets", train=False, download = True, transform = valid_transform)
trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch, shuffle= True )
testloader = torch.utils.data.DataLoader(cifar10_test, batch_size = 10000, shuffle = True)

simple_model = torchvision.models.resnet18()
simple_model.to(cuda)
testoptimizer = optim.SGD(simple_model.parameters(), .1, momentum = .9 weight_decay=.0001)
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

def prune(weights, iteration, p, zeros, mask):
  pruned = int((total_size-zeros)*(p**iteration))
  flat_mask = dict_to_vec(mask)
  conv_layers = {}
  for item in mask:
    conv_layers[item] = weights[item]
    conv_layers[item] = conv_layers[item].to(cuda)
  original = torch.abs(dict_to_vec(conv_layers)).tolist()
  mergeSort(original)
  threshold = original[zeros+pruned]
  for item in conv_layers:
      conv_layers[item][conv_layers[item] > threshold] = 1
      conv_layers[item][conv_layers[item] < -threshold] = 1
      conv_layers[item][conv_layers[item] != 1] = 0
  zeros += pruned
  print("pruning finished", pruned)
  return (conv_layers,zeros)

mask = {}
init_state_dict = simple_model.state_dict()
for item in init_state_dict:
  if(item == "conv1.weight" or item == "layer1.0.conv1.weight"
    or item =="layer1.0.conv2.weight" or item =="layer1.1.conv1.weight" or item=="layer1.1.conv2.weight"
    or item =="layer2.0.conv1.weight" or item =="layer2.0.conv2.weight" or item=="layer2.1.conv1.weight" or item=="layer2.1.conv2.weight"
    or item =="layer3.0.conv1.weight" or item =="layer3.0.conv2.weight" or item=="layer3.1.conv1.weight" or item=="layer3.1.conv2.weight"
    or item =="layer4.0.conv1.weight" or item =="layer4.0.conv2.weight" or item=="layer4.1.conv1.weight" or item=="layer4.1.conv2.weight"):
    mask[item] = torch.ones(init_state_dict[item].size()).to(cuda)

total_size = torch.reshape(dict_to_vec(mask), (-1, )).size()[0]
total_model_size = torch.reshape(dict_to_vec(simple_model.state_dict()), (-1, )).size()[0]
print(total_model_size, total_size)
# Precomputed values, corresponds to the number of steps for 40/80 epochs, respectively
important_steps = [120*391, 240*391]

num_zeroed = 0
counter = 0
training_iterations = []
accuracy = []
mask_as_list = []
inserts = 0
for i in range (3):
  print("Pruning Step", i+1)
  for m in range(epochs):
    trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size = batch, shuffle = True)
    iter_train = iter(trainloader)
    for j in range(50000//batch + 1):
      testoptimizer.zero_grad()
      data, labels = iter_train.next()
      data = data.to(cuda)
      labels = labels.to(cuda)
      output = simple_model(data)
      loss = testcriterion(output, labels)
      loss.backward()
      mask_list_counter = 0
      for parameter in simple_model.parameters():
        if(mask_list_counter < len(mask_as_list)):
          if(parameter.size() == mask_as_list[mask_list_counter].size()):
              parameter.grad = parameter.grad*mask_as_list[mask_list_counter]
              mask_list_counter += 1
      testoptimizer.step()
      state_dict = simple_model.state_dict()
      for item in mask:
        state_dict[item] = state_dict[item].to(cuda)*mask[item]
      simple_model.load_state_dict(state_dict)
      counter += 1
      for k in range(len(important_steps)):
        if(important_steps[k] == counter):
            print("Pruning...", k+1)
            current = simple_model.state_dict()
            current_convs = {}
            for item in mask:
                current_convs[item] = current[item]
            (mask, num_zeroed) = prune(current_convs, k+1, .94, num_zeroed, mask)
            temp_state_dict = simple_model.state_dict()
            mask_as_list = []
            for item in mask:
                temp_state_dict[item] = mask[item] * temp_state_dict[item].to(cuda)
                temp_state_dict[item] = temp_state_dict[item].to(cuda)
                mask_as_list.append(mask[item])
            simple_model.load_state_dict(temp_state_dict)
    testloader = torch.utils.data.DataLoader(cifar10_test, batch_size = 10000, shuffle = True)
    iter_test = iter(testloader)
    test_data, test_labels = iter_test.next()
    test_data = test_data.to(cuda)
    test_labels = test_labels.to(cuda)
    test_output = simple_model(test_data)
    correct = validate(test_output, test_labels)
    training_iterations.append(counter)
    accuracy.append(correct/10000)
    print("Epoch", m, "acc", correct/10000)

torch.save(accuracy, "../Allen_UROP/data/resnet18_nf_prune_accuracy.txt")
torch.save(training_iterations, "../Allen_UROP/data/resnet18_training_iteration.txt")
torch.save(simple_model.state_dict(), "../Allen_UROP/data/resnet18_nf_prune_model.txt")









