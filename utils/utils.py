import glob
import os
import random
import shutil

import imageio
import numpy as np
import torch
from torchinfo import summary

def print_summary(
        model: torch.nn.Module,
        input_size: tuple,
        device: str = "cpu",
        col_names: list = [
            "input_size",  # Input size of the model
            "output_size",  # Output size of the model
            "num_params",  # Number of learnable parameters in the model
            "params_percent",  # Percentage of learnable parameters in the model
            "kernel_size",  # Kernel size of the model
            "mult_adds",  # Multiply-adds in the model
            "trainable"  # Whether the model is trainable or not
        ],
        depth: int = 3  # Depth of the model (default is 3)
        ):
    """
    Prints a summary of a given model.

    Args:
        model (torch.nn.Module): The model whose summary is to be printed.
        input_size (tuple): The size of the input to the model.
        device (str): The device where the model is to be run (default is "cpu").
        col_names (list): The columns to be shown in the summary (default is a list of all columns).
        depth (int): The depth of the model (default is 3).

    Returns:
        A string representation of the summary of the model.
    """
    return summary(model, input_size, col_names=col_names, depth=depth, device=device)


def weight_init(model):
    """
    Initializes the weights of a given model.

    Args:
        model (torch.nn.Module): The model whose weights are to be initialized.
    """
    class_name = model.__class__.__name__
    
    # Check if the layer is a convolutional layer
    if class_name.find("Conv") != -1:
        # Initialize weights with a normal distribution (mean=0.0, std=0.02)
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    
    # Check if the layer is a batch normalization layer
    elif class_name.find("BatchNorm") != -1:
        # Initialize weights with a normal distribution (mean=1.0, std=0.02)
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        # Initialize bias to zero
        torch.nn.init.constant_(model.bias.data, 0.0)

def generate_gif(imgs_path: str, save_path: str) -> None:
    """
    Generates a .gif file from a directory of images.

    Args:
        imgs_path (str): The path of the directory containing the images.
        save_path (str): The path where the .gif file will be saved.
    """
    with imageio.get_writer(save_path, mode='I') as writer:
        # Iterate over the sorted list of image files in the directory
        for filename in sorted(glob.glob(os.path.join(imgs_path, '*.jpg'))):
            # Read the image from the file
            image = imageio.imread(filename)
            # Append the image to the .gif writer
            writer.append_data(image)

def clean_directory(directory: str):
    """
    Removes the directory and its contents if it exists, 
    and then recreates an empty directory.

    Args:
        directory (str): The path of the directory to clean.
    """
    if os.path.exists(directory):
        # Remove the directory and all its contents
        shutil.rmtree(directory)
    # Create a new empty directory
    os.makedirs(directory)


def set_seed(seed: int = 42) -> None:
    """
    Sets the seed for the random number generators in PyTorch, NumPy, and Python.

    Args:
        seed (int, optional): The seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

