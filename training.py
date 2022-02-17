#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cnn_system import CNNSystem
from utils import plot_confusion_matrix
import numpy as np
from torch import cuda, no_grad, rand, randint
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy
from sklearn.metrics import confusion_matrix

def main():
    
    # Create dataset classes from the serialized data splits

    # Create data loaders from the classes (dummy) --> fc1 input size 6720
    training_dataloader = []
    validation_dataloader = []
    testing_dataloader =  []

    for i in range(10):
        sample = (rand(1,2,260,128), randint(4,(1,)))
        training_dataloader.append(sample)
    for i in range(10):
        sample = (rand(1,2,260,128), randint(4,(1,)))
        validation_dataloader.append(sample)
    for i in range(20):
        sample = (rand(1,2,260,128), randint(4,(1,)))
        testing_dataloader.append(sample)

    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Instantiate our DNN
    cnn = CNNSystem(num_channels=2, in_features=6720, output_classes=4)
        
    # Pass DNN to the available device.
    cnn = cnn.to(device)

    # Give the parameters of our DNN to an optimizer. Using L2 regularization.
    optimizer = Adam(params=cnn.parameters(), lr=1e-3, weight_decay=1e-5)

    # Instantiate our loss function as a class.
    loss_function = CrossEntropyLoss()

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 5
    patience_counter = 0

    # Start training.
    epochs = 100
    best_model = None

    for epoch in range(epochs):

        acc = 0 # store model accuracy information

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, dropout will function
        cnn.train()

        # For each batch of our dataset.
        for i, (feature, cls) in enumerate(training_dataloader):
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Process on the appropriate device.
            feature = feature.to(device)
            cls = cls.to(device)

            # Get the predictions of our model.
            y_hat = cnn(feature)

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=cls)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode, dropout will not function
        cnn.eval()
        
        # Do not calculate gradients, so everything will be faster.
        with no_grad():

            # For every batch of our validation data.
            for i, (feature, cls) in enumerate(validation_dataloader):

                # Process on the appropriate device.
                feature = feature.to(device)
                cls = cls.to(device)

                # Get the predictions of our model.
                y_hat = cnn(feature)

                # Calculate the loss of our model.
                loss = loss_function(input=y_hat, target=cls)

                # Log the validation loss.
                epoch_loss_validation.append(loss.item())

                # Calculate accuarcy
                max_index = y_hat.max(dim = 1)[1]
                acc += (max_index == cls).sum().item()
                
        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(cnn.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience:

            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss '
                  f'{lowest_validation_loss}', end='\n\n')

            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                cnn.eval()

                with no_grad():

                    acc = 0 # store model accuracy information
                    y_true, y_pred = [], [] # store output for confusion matrix

                    for i, (feature, cls) in enumerate(testing_dataloader):

                        # Process on the appropriate device.
                        feature = feature.to(device)
                        cls = cls.to(device)

                        # Get the predictions of our model.
                        y_hat = cnn(feature)

                        # Calculate the loss of our model.
                        loss = loss_function(input=y_hat, target=cls)

                        testing_loss.append(loss.item())

                        # Calculate accuarcy
                        max_index = y_hat.max(dim = 1)[1]
                        acc += (max_index == cls).sum().item()

                        y_pred.append(max_index.tolist())
                        y_true.append(cls.tolist())

                testing_loss = np.array(testing_loss).mean()

                print(f'Testing loss: {testing_loss:7.4f} | '
                      #f'Accuracy {100*acc/len(testing_dataset.files):5.2f} %')
                      f'Accuracy {100*acc/10:5.2f} %')

                # Plot confusion matrix
                y_true = np.array(y_true).flatten()
                y_pred = np.array(y_pred).flatten()
                cm = confusion_matrix(y_true, y_pred)
                classes = ['0', '1', '2', '3'] # TODO: change these labels
                plot_confusion_matrix(cm, classes)

                break

        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f} | '
              #f'Accuracy {100*acc/len(validation_dataset.files):5.2f} %')
              f'Accuracy {100*acc/10:5.2f} %')

if __name__ == "__main__":
    main()