
import torch
import torch.nn.functional as F
import numpy as np


def save_checkpoint(state, filename="checkpoint.pth"):
    #save the checkpoint in the file passed by parameter
    
    torch.save(state, filename)
    print("Checkpoint saved!")


def load_checkpoint(checkpoint, model, optimizer):
    #Load the model checkpoint from a file and return the the training step or iteration at which the checkpoint was saved.
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    step = checkpoint['step']
    
    print("Checkpoint loaded!")
    return step
