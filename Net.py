import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


batch_size = 1

use_cuda = True
print('Cuda Available:', torch.cuda.is_available())
device = torch.device('cuda' if (torch.cuda.is_available() and use_cuda) else 'cpu')

# Load all the data set
FSeg_data = np.load('Classified/FSeg.npy', allow_pickle=True)
# print('Original FSeg_data shape:', FSeg_data.shape)

NSeg_data = np.load('Classified/NSeg.npy', allow_pickle=True)
# print('Original NSeg_data shape:', NSeg_data.shape)

SSeg_data = np.load('Classified/SSeg.npy', allow_pickle=True)
# print('Original SSeg_data shape:', SSeg_data.shape)

VSeg_data = np.load('Classified/VSeg.npy', allow_pickle=True)
# print('Original VSeg_data shape:', VSeg_data.shape)

# choose proper number of data to deal with the imbalance problem
data_num = 8000
FSeg_data = FSeg_data[:800]
NSeg_data = NSeg_data[:2400]
SSeg_data = SSeg_data[:2400]
VSeg_data = VSeg_data[:2400]

# Concatenate all the data together
dataSet = np.concatenate((FSeg_data, NSeg_data, SSeg_data, VSeg_data))
# print('dataSet shape:', dataSet.shape)

# Shuffle the dataSet
np.random.seed(12321)  # for reproducibility
torch.manual_seed(12321)  # for reproducibility
np.random.shuffle(dataSet)

# Extract labels
labels = np.array([])
for data in dataSet:
    labels = np.append(labels, data[0])

# Delete first element of dataset
dataSet = np.delete(dataSet, 0, 1)
print(dataSet.shape)
dataSet = dataSet.reshape((8000,1,-1))
print(dataSet.shape)

# Normalize the data
def normalization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

dataSet = normalization(dataSet)
dataSet = dataSet.astype(float)

classes = ('FSeg', 'NSeg', 'SSeg', 'VSeg')

# Convert the classes to one-hot label
for i in range(len(labels)):
    if labels[i] == 'F':
        labels[i] = 0
    if labels[i] == 'N':
        labels[i] = 1
    if labels[i] == 'S':
        labels[i] = 2
    if labels[i] == 'V':
        labels[i] = 3

labels = labels.astype(int)

def batchify_data(x_data, y_data, batch_size):
    N = int(len(y_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({'x': torch.tensor(x_data[i:i+batch_size], dtype=torch.float32),
                        'y': torch.tensor(y_data[i:i+batch_size], dtype=torch.int64)})
    return batches

batchified_data = batchify_data(dataSet, labels, batch_size)

# Divide the data into training set and validation set
divider = int(len(batchified_data) * 0.90)
train_data = batchified_data[:divider]
val_data = batchified_data[divider:]
print(train_data[0]['x'].shape)
# print(val_data)

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

                                   nn.Conv1d(64,64,9),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),        # 58

                                   nn.Conv1d(64,128,9),     # 54
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),

                                   nn.Conv1d(128,128,9),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(inplace=True),

                                   nn.MaxPool1d(2),        # 27

                                   self.flatten,

                                   nn.Dropout(),
                                   nn.Linear(2176,1024),
                                   nn.ReLU(inplace=True),

                                   nn.Dropout(),
                                   nn.Linear(1024,1024),
                                   nn.ReLU(inplace=True),

                                   nn.Linear(1024,4),
                                   )

    def forward(self,x):
        x = self.model(x)
        return F.log_softmax(x)

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
        loss = F.nll_loss(outputs, labels)
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


def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=100):
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
        path = './ecg_net2.pth'
        torch.save(model.state_dict(), path)

    return losses,accuracies,val_losses,val_accuracies


if __name__ == '__main__':
    model = Net().to(device)

    # train_model(train_data,val_data,model)

