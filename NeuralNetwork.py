"""
    File containing final neural network implementation.

    Resources:
    - "Speech Emotion Recognition using Convolutional Neural Networks" by Somayeh Shahsavarani (model params)
    - https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/ (train/validation code)

    :author Theodora Bendlin
"""

import torch.nn as nn
import matplotlib.pyplot as plt
from AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import max, no_grad, save, unsqueeze
from numpy import arange

# Hyper parameters for the neural network
NUM_EPOCHS = 50
NUM_CLASSES = 5
BATCH_SIZE = 20
LEARNING_RATE = 0.001

# Data paths
TEST_DATA_PATH = './Filtered_Data/Test/'
TRAIN_DATA_PATH = './Filtered_Data/Train/'
MODEL_STORE_PATH = './models/'

'''
    Definition for the convolutional neural network that was outlined in the
    paper "Speech Emotion Recognition using Convolutional Neural Networks" by
    Somayeh Shahsavarani.
'''
class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(32 * 32 * 16, 1024)
        self.fc2 = nn.Linear(1024, 5)

    def forward(self, x):
        # Add the channels dimension to get a 4-channel input,
        # which PyTorch expects (BATCH_SIZE, CHANNELS, DIM1, DIM2)
        x = unsqueeze(x, 1)

        # Pass through the first two convolutional layers
        out = self.layer1(x)
        out = self.layer2(out)

        # Resize for the fully connected layers
        out = out.reshape(out.size(0), -1)

        # Apply the drop out algorithm to reduce overfitting
        out = self.drop_out(out)

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
def train(train_loader):
    model = ConvNeuralNetwork()

    # Using cross entropy as the loss function
    # This handily also packages in a softmax classifier
    loss = nn.CrossEntropyLoss()

    # Adam optimizer to minimize the loss function
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Keeping track of the losses and accuracy for each pass
    total_step = len(train_loader)
    accuracies = []
    losses = []
    for epoch in range(0, NUM_EPOCHS):
        for i, (spectograms, labels) in enumerate(train_loader):

            # Feed-forward pass
            forward_outputs = model(spectograms)

            # Calculate the cross entropy loss
            loss_value = loss(forward_outputs, labels)

            # Append the losses so that we can keep track of them later
            losses.append(loss_value)

            # Backpropogation pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            b_size = labels.size(0)
            _, predicted = max(forward_outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracies.append(correct / b_size)

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss_value.item(),
                              (correct / b_size) * 100))

    plot_data(losses, arange(0, len(losses)), "Losses", "Iteration", "Losses for Training Data")
    plot_data(accuracies, arange(0, len(accuracies)), "Accuracy", "Iteration", "Accuracy for Training Data")

    return model


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
        for i, (spectogram, labels) in enumerate(test_loader):
            model_outputs = model(spectogram)
            _, predicted = max(model_outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        validation_accuracy = (float(correct) / total) * 100

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

if __name__ == '__main__':

    train_dataset = AudioDataset(data_root=TRAIN_DATA_PATH, is_test=False)
    test_dataset = AudioDataset(data_root=TEST_DATA_PATH, is_test=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train the model
    trained_model = train(train_loader)

    # Print out the final accuracy of the trained model
    accuracy = validate(test_loader, trained_model)
    print("Final Accuracy: ", accuracy)

    # Save the final model
    save(trained_model.state_dict(), MODEL_STORE_PATH + 'speech_emotion_model.ckpt')