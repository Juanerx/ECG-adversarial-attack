from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Net import *
from tqdm import tqdm

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
# from six.moves import urllib
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = 'ecg_net.pth'
use_cuda=True

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

# # MNIST Test dataset and dataloader declaration
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
#             transforms.ToTensor(),
#             ])),
#         batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

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

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

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
    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, val_data, eps)
        accuracies.append(acc)
        examples.append(ex)


    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(10,5))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=5)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            # plt.imshow(ex, cmap="gray")
            plot_ecg(ex)
    plt.tight_layout()
    plt.show()