#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cnn_system import CNNSystem
from utils import plot_confusion_matrix
import numpy as np
from torch import cuda, no_grad
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy

def main():

    # Create dataset classes from the serialized data splits

    # Create data loaders from the classes

    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Instantiate our DNN
    cnn = CNNSystem()
        
    # Pass DNN to the available device.
    # cnn = cnn.to(device)

    # Give the parameters of our DNN to an optimizer. Use L2 regularization.
    optimizer = Adam(params=cnn.parameters(), lr=1e-3)

    # Instantiate our loss function as a class.
    loss_function = CrossEntropyLoss()

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 20
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

            # Get the batches.
            x_input = feature
            y_output = cls

            # Give them to the appropriate device.
            # x_input = x_input.to(device)
            # y_output = y_output.to(device)

            # Get the predictions of our model.
            y_hat = cnn(x_input)

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y_output)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in training mode, dropout will not function
        cnn.eval()
        
        # Do not calculate gradients, so everything will be faster.
        with no_grad():

            # For every batch of our validation data.
            for i, (feature, cls) in enumerate(validation_dataloader):

                # Get the batch
                x_1_input = feature
                y_output = cls

                # Pass the data to the appropriate device.
                # x_input = x_input.to(device)
                # y_output = y_output.to(device)

                # Get the predictions of the model.
                y_hat = cnn(x_input)

                # Calculate the loss.
                loss = loss_function(input=y_hat, target=y_output)

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
            print(f'Best epoch {best_validation_epoch} with loss \
                  {lowest_validation_loss}', end='\n\n')

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
                        x_1_input = feature
                        y_output = cls

                        # x_input = x_input.to(device)
                        # y_output = y_output.to(device)

                        y_hat = cnn(x_1_input)

                        loss = loss_function(input=y_hat, target=y_output)

                        testing_loss.append(loss.item())

                        max_index = y_hat.max(dim = 1)[1]
                        acc += (max_index == y_output).sum().item()

                        y_pred.append(max_index.tolist())
                        y_true.append(y_output.tolist())

                testing_loss = np.array(testing_loss).mean()

                print(f'Testing loss: {testing_loss:7.4f} | '
                      f'Accuracy {100*acc/len(testing_dataset.files):5.2f} %')

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
              f'Accuracy {100*acc/len(validation_dataset.files):5.2f} %')

if __name__ == "__main__":
    main()