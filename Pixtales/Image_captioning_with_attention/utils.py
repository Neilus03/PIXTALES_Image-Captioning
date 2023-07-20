import torch  
import torch.nn.functional as F
import numpy as np

# Save model state to file
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)  # Save model state
    print("Checkpoint saved!")

# Load model state from file
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])  # Load model parameters
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
    step = checkpoint['step']  # Retrieve step number
    print("Checkpoint loaded!")
    return step

