#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cnn_system import CNNSystem
from utils import plot_confusion_matrix, NUMBER_OF_INSTRUMENTS, INSTRUMENTS
import numpy as np, os
from torch import cuda, no_grad, argmax
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from getting_and_init_the_data import get_dataset, get_data_loader

def main():

    train_data_dir = "/COMP.SGN.220-project/Processed_TrainingData/Train"
    val_data_dir = "/COMP.SGN.220-project/Processed_TrainingData/Validation"
    test_data_dir = "/COMP.SGN.220-project/Processed_TestingData"

    current_dir = os.path.dirname(__file__)
    train_data_dir = current_dir + "\Processed_TrainingData\Train"
    val_data_dir = current_dir + "\Processed_TrainingData\Validation"
    test_data_dir = current_dir + "\Processed_TestingData"

    # Create dataset classes from the serialized data splits
    training_data = get_dataset(train_data_dir)
    validation_data = get_dataset(val_data_dir)
    testing_data = get_dataset(test_data_dir)
    batch_size = 32

    # Create data loaders from the classes --> fc1 input size 960
    training_dataloader = get_data_loader(dataset=training_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
    validation_dataloader = get_data_loader(dataset=validation_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
    testing_dataloader = get_data_loader(dataset=testing_data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

    # This is a demonstration of the dataloader datatype
    dataiter = iter(testing_dataloader)
    data = dataiter.next()
    features, cls = data
    print(features.shape, cls.shape)

    dataiter = iter(training_dataloader)
    data = dataiter.next()
    features, cls = data
    print(features.shape, cls.shape)


    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Instantiate our DNN
    cnn = CNNSystem(num_channels=2, 
                    in_features=6720,
                    output_classes=NUMBER_OF_INSTRUMENTS)
        
    # Pass DNN to the available device.
    cnn = cnn.to(device)

    # Give the parameters of our DNN to an optimizer. Using L2 regularization.
    optimizer = Adam(params=cnn.parameters(), lr=1e-3, weight_decay=1e-5)

    # Instantiate our loss function as a class.
    loss_function = CrossEntropyLoss()

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 10
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
            target = argmax(cls, dim=1)
            loss = loss_function(input=y_hat, target=target)

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
                target = argmax(cls, dim=1)
                loss = loss_function(input=y_hat, target=target)

                # Log the validation loss.
                epoch_loss_validation.append(loss.item())

                # Calculate accuarcy

                max_index = y_hat.max(dim = 1)[1]
                max_index = one_hot(max_index,
                                    num_classes=NUMBER_OF_INSTRUMENTS)
                acc += (max_index == cls).all(dim=1).sum().item()
                
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
                        target = argmax(cls, dim=1)
                        loss = loss_function(input=y_hat, target=target)

                        testing_loss.append(loss.item())

                        # Calculate accuracy
                        max_index = y_hat.max(dim = 1)[1]
                        max_index = one_hot(max_index,
                                            num_classes=NUMBER_OF_INSTRUMENTS)
                        acc += sum(cls[cls==max_index]).item()

                        y_pred.append(argmax(max_index, dim=1).tolist())
                        y_true.append(argmax(cls, dim=1).tolist())

                testing_loss = np.array(testing_loss).mean()

                print(f'Testing loss: {testing_loss:7.4f} | '
                      f'Accuracy {100*acc/len(testing_data.files):5.2f} %')

                # Plot confusion matrix
                y_true = np.array(y_true).flatten()
                y_pred = np.array(y_pred).flatten()
                cm = confusion_matrix(y_true, y_pred)
                classes = INSTRUMENTS.keys()
                plot_confusion_matrix(cm, classes)

                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f} | '
              f'Accuracy {100*acc/len(validation_data.files):5.2f} %')

if __name__ == "__main__":
    main()