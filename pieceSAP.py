from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Net import *
from tqdm import tqdm
import time


sizes = [5, 7, 11, 15, 19]
sigmas = [0.1, 0.3, 0.5, 0.7, 1.0]
epsilons = [.05, .1, .15, .2, .25, .3]
pretrained_model = 'ecg_net.pth'
use_cuda=True
starting_position = []

crafting_sizes = []
crafting_weights = []
for size in sizes:
    for sigma in sigmas:
        crafting_sizes.append(size)
        weight = np.arange(size) - size//2
        weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        crafting_weights.append(weight)

class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)

# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(nn.Conv1d(1, 32, 9),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True),  # 243

                                   nn.Conv1d(32, 32, 9),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True),  # 243

                                   nn.MaxPool1d(2),

                                   nn.Conv1d(32, 64, 9),  # 117
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True),

                                   nn.Conv1d(64, 64, 9),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),  # 58

                                   nn.Conv1d(64, 128, 9),  # 54
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),

                                   nn.Conv1d(128, 128, 9),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),  # 27

                                   self.flatten,

                                   nn.Dropout(),
                                   nn.Linear(2176, 1024),
                                   nn.ReLU(inplace=True),

                                   nn.Dropout(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024, 4),
                                   )

    def forward(self, x):
        x = self.model(x)
        return x


# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cuda'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def choose(inputs, target, adv, length):
    target = target.data.item()
    # print(target)
    store = None
    best = None
    approximate = None

    for i in range(0,250-length):
        temp = adv.clone()
        temp[0][0][:i] = 0
        temp[0][0][i+length:] = 0
        # print(temp)
        outputs = inputs + temp
        prob = model(outputs)
        # print(prob)
        if store == None or prob[0][target] < store:
            store = prob[0][target]
            best = temp

    for j in range(0,250-length):
        if best[0][0][j] != 0:
            approximate = (j//5) *5
            starting_position.append(j//5)
            break
    res = adv.clone()
    res[0][0][:approximate] = 0
    res[0][0][approximate+length:] = 0



    return inputs + res



# # FGSM attack code
def pgd_conv(inputs, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
    best = None
    store = None
    for point in range(0,200):
        crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
        crafting_target = torch.autograd.Variable(targets.clone())
        for i in range(num_steps):
            output = model(crafting_input)
            # print(output)
            # print(targets)
            loss = criterion(output, crafting_target)
            if crafting_input.grad is not None:
                crafting_input.grad.data.zero_()
            loss.backward()
            added = torch.sign(crafting_input.grad.data)
            added[0][0][0:point] = 0
            added[0][0][point+50:] = 0
            step_output = crafting_input + step_alpha * added
            total_adv = step_output - inputs
            total_adv = torch.clamp(total_adv, -eps, eps)
            crafting_output = inputs + total_adv
            crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
        added = crafting_output - inputs
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        for i in range(num_steps*2):
            temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
            for j in range(len(sizes)-1):
                temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
            temp = temp/float(len(sizes))
            output = model(inputs + temp)
            loss = criterion(output, targets)
            loss.backward()
            added = added + step_alpha * torch.sign(added.grad.data)
            added = torch.clamp(added, -eps, eps)
            added[0][0][0:point] = 0
            added[0][0][point+50:] = 0
            added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
        temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
        for j in range(len(sizes)-1):
            temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
        temp = temp/float(len(sizes))
        temp = temp.detach()

        final_output = model(inputs+temp)
        final_loss = criterion(final_output,targets)
        if store == None or final_loss > store:
            store = final_loss
            best = temp

        # crafting_output = choose(inputs,targets, temp, length=50)
        # crafting_output = inputs + temp.detach()
    crafting_output = inputs + best
    crafting_output_clamp = crafting_output.clone()

    return  crafting_output_clamp


def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for batch in tqdm(test_loader):

        # Send the data and label to the device
        data, target = batch['x'], batch['y']
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        output = F.log_softmax(output, dim=1)
        # print('label:', target)
        # print('initial output:', output)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # # Calculate the loss
        # loss = F.nll_loss(output, target)
        #
        # # Zero all existing gradients
        # model.zero_grad()
        #
        # # Calculate gradients of model in backward pass
        # loss.backward()
        #
        # # Collect datagrad
        # data_grad = data.grad.data
        #
        # # Call FGSM Attack
        # perturbed_data = fgsm_attack(data, epsilon, data_grad)

        perturbed_data = pgd_conv(data, target, model, F.nll_loss, epsilon, step_alpha=0.01, num_steps=20, sizes=crafting_sizes, weights=crafting_weights )

        # Re-classify the perturbed image
        output = model(perturbed_data)
        output = F.log_softmax(output,dim=1)
        # print('final output:',output)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            print(classes[target.data.item()], 'fail to attack, Correct num:',correct)
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            print(classes[target.data.item()], 'attack success!')
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def plot_ecg(data):
    x_val = np.linspace(0,1000,250)
    y_val = data
    plt.plot(x_val, y_val)
    # plt.show()

if __name__ == '__main__':
    time_start = time.time()
    test(model, device, val_data[719:], epsilon=0.1)
    time_end = time.time()
    print('time cost', time_end-time_start)

    # list_data = test(model, device, (train_data + val_data), epsilon=0.1)[2]
    # for i in range(len(starting_position)):
    #     list_data[i] = [starting_position[i]] + list_data[i]
    # list_data = np.array(list_data)
    # print(list_data)
    # np.save('save_data', list_data)