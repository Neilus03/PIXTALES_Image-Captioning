# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def train():

    # Get paths to images and annotations
    images_path = input("Enter the images path (or press Enter to use the default path): ")
    annotations_path = input("Enter the annotations path (or press Enter to use the default path): ")

    # Set default paths if not provided by the user
    images_path = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr8k/Images"
    annotations_path = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr8k/captions.txt"

    # Use the get_loader function to get the training data and dataset
    train_loader, dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=None,
        num_workers=4,
    )

    # Set CUDA benchmark for better performance
    torch.backends.cudnn.benchmark = True

    # Use CUDA if available, else use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Flags for loading and saving models
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 512  # Dimensionality of the embeddings
    hidden_size = 512  # Size of the hidden layer in the RNN
    vocab_size = len(dataset.vocab)  # Size of the vocabulary
    learning_rate = 3e-4  # Learning rate for the optimizer
    num_epochs = 20  # Number of epochs for training
    num_layers = 4  # Number of layers in the RNN

    # Initialize TensorBoard writer for logging purposes
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # We ignore padding tokens when calculating Cross Entropy loss
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]) 
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Adjust the learning rate every 3 epochs by multiplying it with 0.1
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)
    
    # Load saved checkpoint if the load_model flag is True
    if load_model:
        step = load_checkpoint(torch.load("./checkpoint11.pth"), model, optimizer)

    # Set the model in training mode
    model.train()

    # Initialize a list to store the loss values for each epoch
    train_loss_values = []

    print_every = 50  # Determine how often loss is printed during training
    print('starting training ...')
    
    # Begin training loop
    for epoch in range(num_epochs):
        total_loss = 0.0  # Track total loss within each epoch
        start_time = time()  # Start timing the epoch

        for idx, (imgs, captions) in enumerate(train_loader):
            # Move images and captions to the GPU if available
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Perform a forward pass through the model
            outputs = model(imgs, captions)

            # Reshape the outputs and captions for loss computation
            outputs = outputs.reshape(-1, outputs.shape[-1])  
            captions = captions[:, 1:].reshape(-1)  

            # Calculate loss
            loss = criterion(outputs, captions)

            # Log the training loss in TensorBoard
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            # Perform a backward pass through the model to compute gradients
            optimizer.zero_grad()  
            loss.backward()  

            # Update the weights using the calculated gradients
            optimizer.step()  
            
            # Update the learning rate (now disabled for performance reasons (neil: I tried it and did not work as well as expected)
            #scheduler.step() 
            
            total_loss += loss.item()

            # Print out loss every 'print_every' steps
            if (idx + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item()}, Time: {time()}')

        # Calculate and print out the average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        train_loss_values.append(epoch_loss)
        print(f"End of Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save model after each epoch if save_model flag is True
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, "checkpoint_attention"+str(epoch+1)+".pth")

    # After training is done, plot the loss over time
    plt.plot(range(1, num_epochs+1), train_loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve after training")
    plt.show()

    # Save the final model
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint(checkpoint, "final_checkpoint_attention.pth")


# Run the training function if this script is run directly
if __name__ == "__main__":
    train()

