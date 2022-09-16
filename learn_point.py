import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


batch_size = 32

use_cuda = True
print('Cuda Available:', torch.cuda.is_available())
device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')

raw_data = np.load('save_data.npy',allow_pickle=True)
np.random.shuffle(raw_data)

raw_data = raw_data[:-1]

labels = np.array([])
for data in raw_data:
    labels = np.append(labels, data[0])

dataSet = np.delete(raw_data, 0, 1)
dataSet = dataSet.reshape((7968,1,-1))

# print(dataSet)

# Normalize the data
def normalization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


dataSet = normalization(dataSet)
dataSet = dataSet.astype(float)

def batchify_data(x_data, y_data, batch_size):
    N = int(len(y_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({'x': torch.tensor(x_data[i:i+batch_size], dtype=torch.float32),
                        'y': torch.tensor(y_data[i:i+batch_size], dtype=torch.int64)})
    return batches

batchified_data = batchify_data(dataSet, labels, batch_size)
divider = int(len(batchified_data) * 0.80)
train_data = batchified_data[:divider]
val_data = batchified_data[divider:]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = Flatten()
        self.model = nn.Sequential(nn.Conv1d(1,32,9),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True),  # 243

                                   nn.Conv1d(32,32,9),
                                   nn.BatchNorm1d(32),
                                   nn.ReLU(inplace=True),  # 243

                                   nn.MaxPool1d(2),

                                   nn.Conv1d(32,64,9),     # 117
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True),

                                   # nn.Conv1d(64,64,9),
                                   # nn.BatchNorm1d(64),
                                   # nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),        # 58

                                   nn.Conv1d(64,128,9),     # 54
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),

                                   # nn.Conv1d(128,128,9),
                                   # nn.BatchNorm1d(128),
                                   # nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),        # 27

                                   self.flatten,

                                   nn.Dropout(),
                                   nn.Linear(2944,2944),
                                   nn.ReLU(inplace=True),

                                   nn.Dropout(),
                                   nn.Linear(2944,1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024,40),
                                   )

    def forward(self,x):
        return self.model(x)

def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.cpu().numpy(), y.cpu().numpy()))


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses = []
    batch_accuracies = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x,y
        inputs, labels = batch['x'], batch['y']

        # print(inputs)
        # print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get output prediction
        outputs = model(inputs)

        # Predict and store accuracy
        predictions = torch.argmax(outputs, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, labels))

        # Compute losses
        loss = F.cross_entropy(outputs, labels)
        losses.append(loss.data.item())

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(batch_accuracies)
    return avg_loss, avg_accuracy


def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=50):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train | loss: {:.6f}  accuracy: {:.6f}'.format(loss, acc))
        losses.append(loss)
        accuracies.append(acc)

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Valid | loss: {:.6f}  accuracy: {:.6f}'.format(val_loss, val_acc))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save model
        path = './start_point.pth'
        torch.save(model.state_dict(), path)

    return losses,accuracies,val_losses,val_accuracies

if __name__ == '__main__':
    model = Net().to(device)
    train_model(train_data, val_data, model)