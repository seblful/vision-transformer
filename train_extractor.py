# train_extractor.py
import os
import pathlib
import argparse

import torch
from torch import nn
import torchvision

from utils import make_dataloaders
from utils import train



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
SAVE_PRETRAINED_MODEL_PATH = pathlib.Path('models/pretrained_vit_model.pth')


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print(f"Using device is {device}.")
    
    # Create train and test dataloaders
    train_dataloader, test_dataloader, classes, class_to_idx = make_dataloaders(data_path=DATA_PATH,
                                                                               train_size=TRAIN_SIZE,
                                                                               test_size=TEST_SIZE,
                                                                               batch_size=BATCH_SIZE,
                                                                               limit_size=False)
    
    print(f"Length of train_dataloader is {len(train_dataloader)}.\
        \nLength of test_dataloader is {len(test_dataloader)}.")
    
    
    # Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    
    # Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    
    # Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False
    
    # Change the classifier head (set the seeds to ensure same initialization with linear head)
    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(classes)).to(device)
    
    # Creating an optimizer
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                                 lr=LEARNING_RATE,
                                 betas=(0.9, 0.999),
                                 weight_decay=0.3)
    
    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Train the model and save the training results to a dictionary
    results = train(model=pretrained_vit,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)
    
    # Save model
    torch.save(pretrained_vit.state_dict(), SAVE_PRETRAINED_MODEL_PATH)
    print('Model was saved')

if __name__ == '__main__':
    main()
