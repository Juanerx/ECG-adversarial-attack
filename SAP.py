from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Net import *
from tqdm import tqdm

sizes = [5, 7, 11, 15, 19]
sigmas = [0.1, 0.3, 0.5, 0.7, 1.0]
epsilons = [.05, .1, .15, .2, .25, .3]
pretrained_model = 'ecg_net.pth'
use_cuda=True

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
        return F.log_softmax(x,dim=1)


# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def pgd_conv(inputs, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """

    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
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
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
    for j in range(len(sizes)-1):
        temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
    temp = temp/float(len(sizes))
    crafting_output = inputs + temp.detach()
    crafting_output_clamp = crafting_output.clone()
    # for i in range(crafting_output_clamp.size(0)):
    #     remainder = MAX_SENTENCE_LENGTH - lengths[i]
    #     if remainder > 0:
    #         crafting_output_clamp[i][0][:int(remainder / 2)] = 0
    #         crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    # sys.stdout.flush()
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

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
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
    # accuracies = []
    # examples = []
    #
    # # Run test for each epsilon
    # for eps in epsilons:
    #     acc, ex = test(model, device, val_data, eps)
    #     accuracies.append(acc)
    #     examples.append(ex)
    #
    #
    # plt.figure(figsize=(5,5))
    # plt.plot(epsilons, accuracies, "*-")
    # plt.yticks(np.arange(0, 1.1, step=0.1))
    # plt.xticks(np.arange(0, .35, step=0.05))
    # plt.title("Accuracy vs Epsilon")
    # plt.xlabel("Epsilon")
    # plt.ylabel("Accuracy")
    # plt.show()
    #
    # # Plot several examples of adversarial samples at each epsilon
    # cnt = 0
    # plt.figure(figsize=(8,10))
    # for i in range(len(epsilons)):
    #     for j in range(len(examples[i])):
    #         cnt += 1
    #         plt.subplot(len(epsilons),len(examples[0]),cnt)
    #         plt.xticks([], [])
    #         plt.yticks([], [])
    #         if j == 0:
    #             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
    #         orig,adv,ex = examples[i][j]
    #         plt.title("{} -> {}".format(orig, adv))
    #         # plt.imshow(ex, cmap="gray")
    #         plot_ecg(ex)
    # plt.tight_layout()
    # plt.show()
    acc, ex = test(model, device, val_data, 0.1)
    print(acc)