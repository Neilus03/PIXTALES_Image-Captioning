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


def print_and_export_examples(model, device, dataset, num_examples=2, export_file="examples.txt"):
    """
    Generate examples of image captions using the trained model and print them in the console.
    Additionally, export the examples to a file.
    
    Args:
        model: The trained image captioning model.
        device: The device (e.g., 'cpu' or 'cuda') to perform computations on.
        dataset: The dataset object containing image-caption pairs.
        num_examples (int): The number of examples to generate and print.
        export_file (str): The file to export the examples to.
    """
    model.eval()
    indices = np.random.randint(low=0, high=len(dataset), size=num_examples)

    with torch.no_grad(), open(export_file, 'w') as f:
        for idx in indices:
            image, caption = dataset[idx]
            image = image.unsqueeze(0).to(device)
            caption = ' '.join(caption)

            # Generate caption using the model's caption_image method
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            # Print the example in the console
            print(f"Example {idx}:\n")
            print(f"Actual Caption: {caption}\n")
            print(f"Generated Caption: {generated_caption}\n")

            # Write the example to the export file
            f.write(f"Example {idx}:\n")
            f.write(f"Actual Caption: {caption}\n")
            f.write(f"Generated Caption: {generated_caption}\n")
            f.write("\n")

    model.train()
