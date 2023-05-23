import torch
import torch.nn.functional as F
import numpy as np


def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Save the model checkpoint to a file.
    
    Args:
        state (dict): The state dictionary containing model parameters and optimizer state.
        filename (str): The filename to save the checkpoint.
    """
    torch.save(state, filename)
    print("Checkpoint saved!")


def load_checkpoint(checkpoint, model, optimizer):
    """
    Load the model checkpoint from a file.
    
    Args:
        checkpoint (dict): The checkpoint dictionary containing model parameters and optimizer state.
        model: The model to load the parameters into.
        optimizer: The optimizer to load the state into.
    
    Returns:
        step (int): The training step or iteration at which the checkpoint was saved.
    """
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    print("Checkpoint loaded!")
    return step
