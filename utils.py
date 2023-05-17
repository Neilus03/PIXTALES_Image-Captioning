import torch
import torch.nn.functional as F
import numpy as np


def save_checkpoint(state, filename):
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


def print_examples(model, device, dataset, num_examples=2):
    """
    Generate and print examples of image captions using the trained model.
    
    Args:
        model: The trained image captioning model.
        device: The device (e.g., 'cpu' or 'cuda') to perform computations on.
        dataset: The dataset object containing image-caption pairs.
        num_examples (int): The number of examples to generate and print.
    """
    model.eval()
    indices = np.random.randint(low=0, high=len(dataset), size=num_examples)
    
    with torch.no_grad():
        for idx in indices:
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model.sample(image)
            caption = [dataset.vocab.itos[token] for token in output]
            caption = ' '.join(caption)
            print(f"Example {idx} - Generated Caption: {caption}")
    
    model.train()
