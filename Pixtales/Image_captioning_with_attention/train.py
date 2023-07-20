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

    images_path = input("Enter the images path (or press Enter to use the default path): ")
    annotations_path = input("Enter the annotations path (or press Enter to use the default path): ")

    if not images_path:
        images_path = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr8k_images"

    if not annotations_path:
        annotations_path = "/content/drive/MyDrive/IMAGE CAPTIONING/captions.txt"

    # Get the data loader and dataset
    loader, dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=None,
        num_workers=4,
    )

    # Set CUDA benchmark for improved performance
    torch.backends.cudnn.benchmark = True

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Set flags for loading and saving models
    load_model = True
    save_model = True

    # Hyperparameters
    embed_size = 128  # Dimensionality of the word embedding
    hidden_size = 512  # Number of units in the hidden state of the RNN
    vocab_size = len(dataset.vocab)  # Size of the vocabulary
    learning_rate = 1e-4  # Learning rate for the optimizer
    num_epochs = 3  # Number of training epochs
    num_layers = 1 # Number of layers in the RNN

    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter("runs/flickr")
    step = 0
    
    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Ignore padding tokens in the loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3) #Weight decay for regularization (avoid huge weights)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Adjust the learning rate every 3 epochs by multiplying it with 0.6
    if load_model:
        # Load the saved checkpoint
        checkpoint_path = input("checkpoint path: ")
        step = load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    # Set the model to training mode
    model.train()

    # Initialize a list to store the training loss values
    train_loss_values = []

    # Initialize a list to store the validation loss values
    validation_loss_values = []

    print_every = 20 # Change this to control how often you want to print the loss
    print('starting training ...')


    for epoch in range(num_epochs):
        total_loss = 0.0  # Variable to track the total loss for the epoch

        for idx, (imgs, captions) in enumerate(loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Forward pass through the model
            outputs, att_weights = model(imgs, captions) #originally  outputs = model(imgs, captions[:-1]) bcs we wanted to predict the EOS token

            outputs = outputs.reshape(-1, outputs.shape[-1])  # Reshape the output to [batch_size * sequence_length, vocabulary_size]

            captions = captions.reshape(-1)  # Reshape the captions to be[batch_size * sequence_length]

            loss = criterion(outputs, captions)

            #Log the training loss in TensorBoard 
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Perform backward pass to calculate gradients
            optimizer.step()  # Update the weights using the gradients
            total_loss += loss.item()

            if (idx + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(loader)}], Loss: {loss.item()}') 


        epoch_loss = total_loss / len(loader)
        train_loss_values.append(epoch_loss)
        print(f"End of Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        scheduler.step() # Update the learning rate

        if save_model:
            # Save the final model checkpoint
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, "checkpoint_attention"+str(epoch+1)+".pth")

    # Plot the training loss curve
    plt.plot(range(1, num_epochs+1), train_loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve after training")
    plt.show()

    if save_model:
        # Save the final model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        }
        save_checkpoint(checkpoint, "final_checkpoint_attention.pth")
        
        
if __name__ == "__main__":
    train()
