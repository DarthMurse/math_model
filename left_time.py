import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import *
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Loading data
sample_array = np.loadtxt('data_left.txt')
label_array = np.loadtxt('label_left.txt')
label_array *= 1

# Define constants
# The total number of samples
sample_size = sample_array.shape[0]

# Attribute number
attribute = 3

# Output size
output_size = 1

# The total number of time points
time_length = count

# The number of hidden_layers
hidden_layers = 400

# The number of input_layers
input_layers = attribute * time_length

# Batch_size
batch_size = 64

# Number of epochs
epochs = 100

# Learning rate
lr = 0.0001

# Seperate traning, validating and testing sets, with proportion 7:2:1
seperate_point1 = int(0.7 * sample_size)
seperate_point2 = int(0.9 * sample_size)
training_sample = sample_array[:seperate_point1, :]
validating_sample = sample_array[seperate_point1:seperate_point2, :]
testing_sample = sample_array[seperate_point2:]

training_label = label_array[:seperate_point1]
validating_label = label_array[seperate_point1:seperate_point2]
testing_label = label_array[seperate_point2:]

# Preparing datasets
train_set = TensorDataset(torch.Tensor(training_sample), torch.Tensor(training_label))
valid_set = TensorDataset(torch.Tensor(validating_sample), torch.Tensor(validating_label))
test_set = TensorDataset(torch.Tensor(testing_sample), torch.Tensor(testing_label))

# Preparing dataloaders
train_loader = DataLoader(train_set, batch_size=batch_size)
valid_loader = DataLoader(valid_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# If availabel, use gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda is available')
else:
    device = torch.device('cpu')
    print('cuda is unavailable')

class Model(nn.Module):
    def __init__(self, input_layers, hidden_layers, output_size):
        super(Model, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layers = input_layers
        self.output_size = output_size

        self.layers = nn.Sequential(
                nn.Linear(input_layers, hidden_layers),
                nn.Sigmoid(),
                nn.Linear(hidden_layers, 300),
                nn.Sigmoid(),
                nn.Linear(300, output_size),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.layers(x)

#model = Model(input_layers, hidden_layers, output_size)
#optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
#criterion = nn.MSELoss()
#loss_curve = []

def train_loop(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = criterion(pred, y)
        loss_curve.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += criterion(pred, y).sum().item()
            #correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            correct += (abs(pred[:, 0] - y) <= 0.2).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct): >0.1f}%, avg loss: {test_loss:0>f} \n")
    return correct

if __name__ == '__main__':
    for t in range(epochs):    
        print(f"epoch {t+1} -------------------")
        train_loop(train_loader, model, criterion, optimizer)
        test_loop(train_loader, model, criterion)
        test_loop(valid_loader, model, criterion)
        #test_loop(test_dataloader, model, loss_fn)

    plt.plot(loss_curve)
    plt.title('learning curve')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.show()
    torch.save(model, 'model2.pth')
    print("done!")
    x, y = valid_set.__getitem__(0)
    print(model(x))
    print(y)

