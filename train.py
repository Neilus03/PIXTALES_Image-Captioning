import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples

from get_loader import get_loader
from model import CNNtoRNN

def train():
    # Define the image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),  # Resize the image to a specific size
            transforms.RandomCrop((299, 299)),  # Randomly crop the image
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image tensor
        ]
    )
    
    images_path = input("Enter the images path (or press Enter to use the default path): ")
    annotations_path = input("Enter the annotations path (or press Enter to use the default path): ")

    if not images_path:
        images_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/images/"

    if not annotations_path:
        annotations_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/captions.txt"

    # Get the data loader and dataset
    train_loader, dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=transform,
        num_workers=4
    )

    # Set CUDA benchmark for improved performance
    torch.backends.cudnn.benchmark = True

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set flags for loading and saving models
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256  # Dimensionality of the word embedding
    hidden_size = 256  # Number of units in the hidden state of the RNN
    vocab_size = len(dataset.vocab)  # Size of the vocabulary
    learning_rate = 3e-4  # Learning rate for the optimizer
    num_epochs = 100  # Number of training epochs
    num_layers = 1  # Number of layers in the RNN

    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Ignore padding tokens in the loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        # Load the saved checkpoint
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Set the model to training mode
    model.train()

    for epoch in range(num_epochs):
        # Print examples of generated captions
        print_examples(model, device, dataset)

        if save_model:
            # Save the current model checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward pass through the model
            outputs = model(imgs, captions[:-1])  # We want the model to predict the end token

            # Calculate the loss
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # Log the training loss in TensorBoard
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            # Zero the gradients, perform backward pass, and update the weights
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Perform backward pass to calculate gradients
            optimizer.step()  # Update the weights using the gradients

if __name__ == "__main__":
    train()