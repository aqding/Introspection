import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn, optim
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        if stable_resnet:
            # Total number of blocks for Stable ResNet
            # https://arxiv.org/pdf/2002.08797.pdf
            L = 0
            for x in num_blocks:
                L+=x
            self.L = L
        else:
            self.L = 1
        
        self.masks = None

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes*8*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.L))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out) / self.temp
        
        return out
            

def resnet18(temp=1.0, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    return model

I_model = torch.load("../Allen_UROP/data/introspection.txt")

cuda = torch.device("cuda:0")

epochs = 120
batch = 128
# Steps per epoch (CIFAR): 391
cifar10_train = datasets.CIFAR10(root = "../Allen_UROP/datasets", train = True, download = True, transform = torchvision.transforms.ToTensor())
cifar10_test = datasets.CIFAR10(root = "../Allen_UROP/datasets", train=True, download = True, transform = torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch, shuffle= True )
testloader = torch.utils.data.DataLoader(cifar10_test, batch_size = 10000, shuffle = True)

simple_model = resnet18()
simple_model.to(cuda)
testoptimizer = optim.SGD(simple_model.parameters(), .001, weight_decay=.0001)
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
      counter += 1
      for k in range(len(timesteps)):
        if(timesteps[k] == counter):
          x = simple_model.state_dict()
          new_dict = {}
          for item in mask:
            new_dict[item] = x[item]
          weight_holder[k] = dict_to_vec(new_dict)
      for k in range(len(important_steps)):
        if(important_steps[k] == counter):
            print("Pruning...", k+1)
            acceleration = I_model(torch.transpose(weight_holder[4*k:4*k+4, :], 0, 1)*1000)/1000
            accelerate_dict = vec_to_dict(acceleration, mask)
            (mask, num_zeroed) = prune(accelerate_dict, k+1, .9, num_zeroed, mask)
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



        
torch.save(accuracy, "../Allen_UROP/data/resnet18_f_prune_accuracy.txt")
torch.save(training_iterations, "../Allen_UROP/data/resnet18_training_iteration.txt")
torch.save(simple_model.state_dict(), "../Allen_UROP/data/resnet18_f_prune_model.txt")
