
import os
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


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
    

def plot_patchified_image(image, patch_size):
    '''
    Plots patched image
    '''
    # Permuting image
    image_permuted = image.permute(1, 2, 0)
    
    # Setting image size (image always square) and num_patches
    image_size = image_permuted.shape[0]
    num_patches = int(image_size / patch_size)
    
    # Check if image size is divided without remainder
    assert image_size % patch_size == 0
    
    figs, axs = plt.subplots(nrows=int(image_size / patch_size),
                            ncols=int(image_size / patch_size),
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)
    
    for i, patch_height in enumerate(range(0, image_size, patch_size)):
        for j, patch_width in enumerate(range(0, image_size, patch_size)):
            # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
            axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size,
                             patch_width:patch_width+patch_size, :])
            
            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1, 
                                 rotation="horizontal", 
                                 horizontalalignment="right", 
                                 verticalalignment="center") 
            axs[i, j].set_xlabel(j+1) 
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()
    
    # Set a super title
    figs.suptitle("Patchified image", fontsize=16)
    plt.show()
    

def plot_image_out_of_conv(image, k=5):
    '''
    Plots patchified image by a convolutional layer
    '''
    # Instantiate k random indexes
    random_indexes = random.sample(range(0, 758), k=k)
    
    figs, axs = plt.subplots(nrows=1, ncols=k)
    
    for i, idx in enumerate(random_indexes):
        # Taking random image and prepare it for plotting
        random_image = image[:, idx, :, :].squeeze().detach().numpy()
        # Plotting each subplot
        axs[i].imshow(random_image)
        # Deleting labels, ticks
        axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]);
        
    
import matplotlib.pyplot as plt

def plot_image_out_of_flat(flattened_image):
    '''
    Plots flattened feauture map
    '''
    # Permute, taking single map and detach image
    single_flatten_map = flattened_image.permute(0, 2, 1)[:, :, 0].detach().numpy()
    
    # Change figsize
    plt.figure(figsize=(20, 20))
    plt.imshow(single_flatten_map)
    plt.axis('off');
