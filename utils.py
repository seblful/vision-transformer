
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


import matplotlib.pyplot as plt
import torch

def make_dataloaders(data_path, 
                     train_size=0.85,
                     test_size=0.15,
                     batch_size=32):
    '''
    Creates train and test dataloaders
    '''
    # Defining num workers
    num_workers = os.cpu_count()
    
    # Creating transformer to transform data
    transformer = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        ])
    
    # Creating train and test dataset
    full_dataset = datasets.ImageFolder(root=data_path,
                                        transform=transformer)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    

    # Creating train and test dataloader
    train_dataloader = DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
    
    return train_dataloader, test_dataloader, full_dataset.classes, full_dataset.class_to_idx


def visualize_dataloader_images(dataloader,
                                classes=None,
                                nrows=4,
                                ncols=4):
    '''
    Visualizes random images from DataLoader
    '''
    # Taking one batch and one label
    batch, label = next(iter(dataloader))
    
    # Create a nrows*ncols grid of subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    
    # Subplotting
    for i in range(nrows):
        for j in range(ncols):
            random_index = torch.randint(32, size=(1,))[0]
            image = batch[random_index]
            axs[i, j].imshow(image.permute(1, 2, 0))
            axs[i, j].axis('off')
            
            if classes is not None:
                axs[i, j].set_title(classes[label[random_index]])
                
    plt.tight_layout()
    plt.show()
