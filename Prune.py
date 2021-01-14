# Predicting Masks

epochs = 40
batch = 128
# Steps per epoch: 469
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size = batch, shuffle = True)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = True)

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

# Precomputer values, corresponds to the number of steps for 40/80 epochs, respectively
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

num_zeroed = 0
counter = 0
trajectory = {}
inserts = 0
for i in range (3):
  print("Pruning Step", i+1)
  testloader = torch.utils.data.DataLoader(mnist_test, batch_size = 10000//epochs, shuffle = True)
  iter_test = iter(testloader)
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
          (mask, num_zeroed) = prune(accelerate_dict, k+1, .2, num_zeroed, mask)
          temp_state_dict = simple_model.state_dict()
          for item in mask:
            temp_state_dict[item] = mask[item] * temp_state_dict[item]
          simple_model.load_state_dict(temp_state_dict)
    test_data, test_labels = iter_test.next()
    test_data = torch.reshape(test_data, (test_data.size()[0], 28**2))
    test_output = simple_model(test_data)
    correct = validate(test_output, test_labels)
    print("Epoch", m, "acc", correct/250)