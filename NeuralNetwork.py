"""
    File containing final neural network implementation.

    Resources:
    - "Speech Emotion Recognition using Convolutional Neural Networks" by Somayeh Shahsavarani (model params)
    - https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/ (train/validation code)

    :author Theodora Bendlin
"""
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from AudioDataset import AudioDataset
from torch.optim import Adam
from torch import max, no_grad, save, unsqueeze, device
from numpy import arange
from torch.cuda import is_available, set_device
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Hyper parameters for the neural network
NUM_EPOCHS = 115
NUM_CLASSES = 2
BATCH_SIZE = 256
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.01

# Data paths
DATA_PATH_HIGHINTENSITY = './data_highintensity_only/'
SAVED_DATA_PATH_HIGHINTENSITY = './data_highintensity_only_saved/'
MODEL_STORE_PATH = './models/'

'''
    Definition for the convolutional neural network that was outlined in the
    paper "Speech Emotion Recognition using Convolutional Neural Networks" by
    Somayeh Shahsavarani.
'''

USE_CUDA = is_available()
device = device("cuda:0" if USE_CUDA else "cpu")
if device.type == 'cuda':
  set_device(device)

class ConvNeuralNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 32 * 16, 1024),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Add the channels dimension to get a 4-channel input,
        # which PyTorch expects (BATCH_SIZE, CHANNELS, DIM1, DIM2)
        if len(x.shape) != 4:
            x = unsqueeze(x, 1)

        # Pass through the first two convolutional layers
        out = self.layer1(x)
        out = self.layer2(out)

        # Resize for the fully connected layers
        out = out.reshape(out.size(0), -1)

        # Pass through the fully connected layer
        out = self.fc1(out)
        out = self.fc2(out)

        return out

'''
    Training function for the neural network
    
    Written while following this tutorial:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    
    :parameter train_loader, the DataLoader containing the training data
    :returns model, the fully trained CNN
'''


def train(train_loader, test_loader):
    model = ConvNeuralNetwork(num_classes=NUM_CLASSES)
    if USE_CUDA:
        model = model.cuda()

    # Using cross entropy as the loss function
    # This handily also packages in a softmax classifier
    loss = nn.CrossEntropyLoss()
    if USE_CUDA:
        loss = loss.cuda()

    # Adam optimizer to minimize the loss function
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

    # Keeping track of the losses and accuracy for each pass
    total_step = len(train_loader)
    accuracies = []
    losses = []
    num_top_acc = 0
    for epoch in range(0, NUM_EPOCHS):
        for i, (spectrograms, labels) in enumerate(train_loader):

            # Zero out the gradients
            optimizer.zero_grad()

            # Feed-forward pass
            forward_outputs = model(spectrograms)

            if USE_CUDA:
                labels = labels.to(device)

            # Calculate the cross entropy loss
            loss_value = loss(forward_outputs, labels)

            # Backpropogation pass
            loss_value.backward()
            optimizer.step()

            # Append the losses so that we can keep track of them later
            losses.append(loss_value.item())

            b_size = labels.size(0)
            _, predicted = max(forward_outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / b_size
            accuracies.append(accuracy)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss_value.item(),
                          (correct / b_size) * 100))

            # To avoid overfitting, will stop training after threshold reached 5 times
            if accuracy > 0.97:
                num_top_acc += 1

            if num_top_acc > 5:
                break

        if (epoch + 1) % 10 == 0:
            print()
            print("Validation Accuracy: ", validate(test_loader, model))
            print()
            model.train()

        if num_top_acc > 5:
            break

    plot_data(losses, arange(0, len(losses)), "Losses", "Iteration", "Losses for Training Data")
    plot_data(accuracies, arange(0, len(accuracies)), "Accuracy", "Iteration", "Accuracy for Training Data")

    return model

# Helper function to calculate the confusion matrix
def confusion(matrix, labels, predicted):
  for i in range(0, len(labels)):
    matrix[labels[i]][predicted[i]] += 1


'''
    Validation function for the neural network

    Written while following this tutorial:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    
    :parameter test_loader, the DataLoader containing the training data
    :returns validation_accuracy, the final accuracy of the CNN
'''
def validate(test_loader, model):
    # Setting the model to evaluation mode
    model.eval()

    with no_grad():
        correct = 0
        total = 0
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

        for i, (spectrograms, labels) in enumerate(test_loader):
            if USE_CUDA:
                labels = labels.to(device)

            model_outputs = model(spectrograms)
            _, predicted = max(model_outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            confusion(confusion_matrix, labels, predicted)

        validation_accuracy = (float(correct) / total) * 100
        print("Confusion Matrix: ")
        print(confusion_matrix)

    return validation_accuracy

'''
    Helper function that will plot the loss and accuracy graphs
    
    :parameter y_data, the data to populate the y-axis
    :parameter x_data, the data to populate the x-axis
    :parameter y_label, the label for the y-axis
    :parameter x_label, the label for the x-axis
    :parameter title, the title of the graph
'''
def plot_data(y_data, x_data, y_label, x_label, title):
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def main():
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((129, 129)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = AudioDataset(raw_data_root=DATA_PATH_HIGHINTENSITY,
                           saved_data_root=SAVED_DATA_PATH_HIGHINTENSITY,
                           noise_reps=5,
                           transforms=data_transform)

    dataset_length = dataset.__len__()
    train_dataset, test_dataset = random_split(dataset, [int(dataset_length * .7), int(dataset_length * .3)])

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    trained_model = train(train_loader, test_loader)

    # Print out the final accuracy of the trained model
    accuracy = validate(test_loader, trained_model)
    print("Final Accuracy: ", accuracy)

    save(trained_model.state_dict(), MODEL_STORE_PATH + "model.pt")

main()