"""
This script provides an example of building a neural
network for classifying glass identification dataset on
http://http://archive.ics.uci.edu/ml/datasets/Steel+Plates+Faults
It loads data using a data loader, and trains a neural
network with batch training.
"""

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

# Hyper Parameters
input_size = 26
hidden_size = 70
num_classes = 6
num_epochs = 200
batch_size = 1
learning_rate = 0.1


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion

"""
Step 1: Load data and pre-process data(normalisation)
Here we use data loader to read data
"""


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.as_matrix())

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][0:-1]
        target = self.data_tensor[index][-1] - 1

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n


# load all data
data = pd.read_csv('preprocessed_data.csv')

# normalise input data
for i in data.columns:
    if i  != 'TypeOfSteel' and i != 'class':
        mean_xm = data[i].mean()
        std = data[i].std()
        data[i] = [(number - mean_xm) / std for number in data[i]] 

# randomly split data into training set (80%) and testing set (20%)
msk = np.random.rand(len(data)) < 0.8
train_data = data[msk]
test_data = data[~msk]

# due to the random split, the index is not continous. So here to reset index
train_data = train_data.reset_index(drop=True) 

# define train dataset and a data loader
train_dataset = DataFrameDataset(df=train_data)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
Step 2: Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: 26 neurons, representing the features of Steel Plate Faults
    hidden layer: 70 neurons, using Sigmoid as activation function
    output layer: 6 neurons, representing the type of Steel Plate Faults
"""


# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = Net(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.7)

# store all losses for visualisation
all_losses = []

# alpha used in deciding which pattern needs to be removed
alpha = 0.9

# flags used in the removal process
flag_start = 0
flag_stop = 0


# train the model by batch
for epoch in range(num_epochs):

    # a sign to stop training
    if flag_stop == 1:
        break

    # initialisation
    loss_per_epoch = [0]
    previous_loss = 0
    subset = []
    subsubset = []
    subset_loss = []

    # train the model by batch
    for step, (batch_x, batch_y) in enumerate(train_loader):
        # convert torch tensor to Variable
        X = Variable(batch_x)
        Y = Variable(batch_y.long())

        # Forward + Backward + Optimize 
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss_per_epoch.append(loss.data[0])
        all_losses.append(loss.data[0])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            _, predicted = torch.max(outputs, 1)
            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
                  % (epoch + 1, num_epochs, step + 1,
                     len(train_data) // batch_size + 1,
                     loss.data[0], 100 * sum(correct)/total))
        
    variance_ts = np.var(loss_per_epoch)

    # start bimodal distribution removal
    if variance_ts <= 0.1:
        flag_start = 1

    # it's time to stop training
    if variance_ts <= 0.01:
        flag_stop = 1
        print('stop on epoch [%d]'
			% (epoch + 1))
        break

    # bimodal distribution removal
    if (epoch % 50 == 0) and (epoch > 1):
        if flag_start == 1:
            mean_ts = np.mean(loss_per_epoch)

            # subset that contains patterns whose loss is greater than the mean of the training set
            for i in range(len(loss_per_epoch)-1):
                if loss_per_epoch[i] > mean_ts:
                    subset.append(i)
                    subset_loss.append(loss_per_epoch[i])

            # choose patterns that are permanently removed from the training set
            if len(subset_loss) != 0:
                mean_ss = np.mean(subset_loss)
                standard_d = np.std(subset_loss)
                for i in range(len(subset_loss)-1):
                    if subset_loss[i] >= (mean_ss + alpha * standard_d):
                        subsubset.append(subset[i])

            # remove patterns
            if len(subsubset) != 0:
                for i in range(len(subsubset)-1):
                    train_data = train_data.drop(subsubset[i])
                    print('the %d row pattern was removed' %subsubset[i])
                # reset the dataloader for the next epoch
                train_data = train_data.reset_index(drop=True) 
                train_dataset = DataFrameDataset(df=train_data)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
              
"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every steel plate fault (rows)
which class the network guesses (columns).

"""
train_input = train_data.iloc[:, :input_size]
train_target = train_data.iloc[:, input_size]

inputs = Variable(torch.Tensor(train_input.as_matrix()).float())
targets = Variable(torch.Tensor(train_target.as_matrix() - 1).long())

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, predicted.long().data, targets.data))


"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# get testing data
test_input = test_data.iloc[:, :input_size]
test_target = test_data.iloc[:, input_size]

inputs = Variable(torch.Tensor(test_input.as_matrix()).float())
targets = Variable(torch.Tensor(test_target.as_matrix() - 1).long())

outputs = net(inputs)
_, predicted = torch.max(outputs, 1)

total = predicted.size(0)
correct = predicted.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every steel plate fault (rows)
which class the network guesses (columns).

"""

print('Confusion matrix for testing:')
print(plot_confusion(test_input.shape[0], num_classes, predicted.long().data, targets.data))


