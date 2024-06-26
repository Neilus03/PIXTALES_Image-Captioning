import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

from utils import save_checkpoint, load_checkpoint
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
        images_path = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr30k_images/"

    if not annotations_path:
        annotations_path = "/content/drive/MyDrive/IMAGE CAPTIONING/results.txt"
    # Get the data loader and dataset
    train_loader, dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=transform,
        num_workers=4,
        dataset='flickr30k',
    )

    # Set CUDA benchmark for improved performance
    torch.backends.cudnn.benchmark = True

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set flags for loading and saving models
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 512  # Dimensionality of the word embedding
    hidden_size = 512  # Number of units in the hidden state of the RNN
    vocab_size = len(dataset.vocab)  # Size of the vocabulary
    learning_rate = 3e-4  # Learning rate for the optimizer
    num_epochs = 30  # Number of training epochs
    num_layers = 4  # Number of layers in the RNN

    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.vocab.stoi["<PAD>"])  # Ignore padding tokens in the loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        # Load the saved checkpoint
        step = load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

    # Set the model to training mode
    model.train()

    # Initialize a list to store the training loss values
    train_loss_values = []

    from datetime import datetime
    print_every = 50  # Change this to control how often you want to print

    for epoch in range(num_epochs):
        total_loss = 0.0  # Variable to track the total loss for the epoch
        start_time = datetime.now()  # Start timing

        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            # Forward pass through the model
            outputs = model(imgs, captions[:-1])  # We want the model to predict the end token
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # Log the training loss in TensorBoard
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Perform backward pass to calculate gradients
            optimizer.step()  # Update the weights using the gradients

            total_loss += loss.item()

            if (idx + 1) % print_every == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{idx + 1}/{len(train_loader)}], Loss: {loss.item()}, Time: {datetime.now() - start_time}')
                start_time = datetime.now()  # Reset timing

        epoch_loss = total_loss / len(train_loader)
        train_loss_values.append(epoch_loss)
        print(f"End of Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        if save_model:
            # Save the final model checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, "30k_checkpoint" + str(epoch + 1) + ".pth")

    # Plot the training loss curve
    plt.plot(range(1, num_epochs + 1), train_loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    if save_model:
        # Save the final model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint(checkpoint, "30k_final_checkpoint.pth")


if __name__ == "__main__":
    train()
