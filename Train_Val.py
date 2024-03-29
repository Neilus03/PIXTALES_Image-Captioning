import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import  SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from io import BytesIO

from utils import save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoRNN


def train(num_epochs,epoch,model,criterion,optimizer,train_loader):
    print('train')
    # Set flags for loading and saving models
    load_model = False
    if epoch % 10 == 0:
        save_model = True
    else:
        save_model = False

    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter("runs/flickr")
    step = 0

    if load_model:
        # Load the saved checkpoint
        step = load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

    # Set the model to training mode
    model.train()

    # Initialize a list to store the training loss values
    train_loss_values = []
    
    from datetime import datetime
    print_every = 50  # Change this to control how often you want to print
    
    
    total_loss = 0.0  # Variable to track the total loss for the epoch
    start_time = datetime.now()  # Start timing
    print('epoch:',epoch)
    for idx, (imgs, captions) in enumerate(train_loader):
        print(idx)
        imgs = imgs.to(device)
        captions = captions.to(device)
        # Forward pass through the model
        outputs = model(imgs, captions[:-1])  # We want the model to predict the end token
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        
        #Log the training loss in TensorBoard 
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1
        
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass to calculate gradients
        optimizer.step()  # Update the weights using the gradients
        
        total_loss += loss.item()
        
        if (idx + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item()}, Time: {datetime.now() - start_time}')
            start_time = datetime.now()  # Reset timing
            
    epoch_loss = total_loss / len(train_loader)
    print(f"End of Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    if save_model:
        # Save the final model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint(checkpoint, "final_checkpoint.pth")

    return epoch_loss


def evaluate_model(model, dataset, device,val_loader,criterion):
    print('eval')
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            #print(f"Evaluating example {idx+1}/{len(dataset)}")
            image, captions = dataset[idx]
            original_image = transforms.ToPILImage()(image).convert("RGB")  # Create the original image
            image = image.unsqueeze(0).to(device)
            captions = captions.to(device)
            # Forward pass through the model
            outputs = model(image, captions[:-1])  # We want the model to predict the end token
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            val_loss += loss.item() 
            
        val_loss /= len(val_loader.dataset)
    
        return val_loss


if __name__ == "__main__":

    # Define the image transformation
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),  # Resize the image to a specific size
            transforms.RandomCrop((299, 299)),  # Randomly crop the image
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image tensor
        ]
    )


    images_path_train = input("Enter the images path (or press Enter to use the default path): ")
    annotations_path_train = input("Enter the annotations path (or press Enter to use the default path): ")

    if not images_path_train:
        images_path_train = "./image_captioning_splitted/8k/Train/Images_train"

    if not annotations_path_train:
        annotations_path_train = "./image_captioning_splitted/8k/Train/captions_train.txt"


    images_path_val = "./image_captioning_splitted/8k/Test/Images_test"
    annotations_path_val = "./image_captioning_splitted/8k/Test/captions_test.txt"

    # Set CUDA benchmark for improved performance
    torch.backends.cudnn.benchmark = True

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get the data loader and dataset
    train_loader, dataset = get_loader(
        root_folder=images_path_train,
        annotation_file=annotations_path_train,
        transform=transform,
        num_workers=4,
    )

    val_loader, val_dataset = get_loader(
        root_folder=images_path_val,
        annotation_file=annotations_path_val,
        transform=transform,
        num_workers=4,
    )

    # Hyperparameters
    embed_size = 256  # Dimensionality of the word embedding
    hidden_size = 256  # Number of units in the hidden state of the RNN
    vocab_size = len(dataset.vocab)  # Size of the vocabulary
    learning_rate = 3e-4  # Learning rate for the optimizer
    num_epochs = 30  # Number of training epochs
    num_layers = 2  # Number of layers in the RNN
    
    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Ignore padding tokens in the loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Keep track of the losses for each epoch
    total_loss = {'train': [], 'val': []}
    print('Aleluya')
    for epoch in range(num_epochs):
        print(epoch)
        train_loss = train(num_epochs,epoch,model,criterion,optimizer,train_loader)
        val_loss = evaluate_model(model,val_dataset,device,val_loader,criterion)
        total_loss["train"].append(train_loss)
        total_loss["val"].append(val_loss)
        
        print(total_loss['val'])
        plt.plot(total_loss["train"], label="training loss")
        plt.plot(total_loss["val"], label="validation loss")
        
        # Add legend
        plt.legend()

        # Save the plot to a file
        plt.savefig('plot.png')

        # Show the plot
        plt.show()
