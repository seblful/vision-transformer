# train.py
import os
import pathlib
import argparse

import torch

from utils import make_dataloaders
from utils import train
from engine import ViT


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for epochs
parser.add_argument("--epochs", 
                     default=10, 
                     type=int, 
                     help="the number of epochs to train for")

# Get an arg for batch_size
parser.add_argument("--batch_size", 
                     default=32, 
                     type=int, 
                     help="the number of batch size to train for")

# Get an arg for lr
parser.add_argument("--lr", 
                     default=0.001, 
                     type=float, 
                     help="the learning rate to train for")

# Get an arg for train_size
parser.add_argument("--train_size", 
                     default=0.85, 
                     type=float, 
                     help="the ratio of data for training dataset")

# Get an arg for test_size
parser.add_argument("--test_size", 
                     default=0.15, 
                     type=float, 
                     help="the ratio of data for test dataset")

# Get an arg for train_sizefolder with data
parser.add_argument("--data_folder", 
                     default='data', 
                     type=str, 
                     help="the name of folder with data")


# Get our arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
TRAIN_SIZE = args.train_size
TEST_SIZE = args.test_size
DATA_PATH = pathlib.Path(args.data_folder)


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(f"Using device is {device}.")
    
    # Create train and test dataloaders
    train_dataloader, test_dataloader, classes, class_to_idx = make_dataloaders(data_path=DATA_PATH,
                                                                               train_size=TRAIN_SIZE,
                                                                               test_size=TEST_SIZE,
                                                                               batch_size=BATCH_SIZE)
    
    print(f"Length of train_dataloader is {len(train_dataloader)}.\
        \nLength of test_dataloader is {len(test_dataloader)}.")
    
    
    # Instantiation ViT
    vit = ViT(num_classes=len(classes))
    
    # Creating an optimizer
    optimizer = torch.optim.Adam(params=vit.parameters(), 
                                 lr=LEARNING_RATE,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.3)
    
    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train the model and save the training results to a dictionary
    results = train(model=vit,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)

if __name__ == '__main__':
    main()
