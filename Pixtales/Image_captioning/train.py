import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from io import BytesIO

from utils import save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoRNN


def train(num_epochs,epoch,model,criterion,optimizer,train_loader):
    print('Training...')
    # Set flags for saving models
    
    save_model = True


    # Create a SummaryWriter for TensorBoard visualization
    writer = SummaryWriter("runs/flickr")
    step = 0

    

    # Set the model to training mode
    model.train()

    # Initialize a list to store the training loss values
    train_loss_values = []
    
    from datetime import datetime
    print_every = 50  # Change this to control how often you want to print
    
    
    total_loss = 0.0  # Variable to track the total loss for the epoch
    start_time = datetime.now()  # Start timing
    for idx, (imgs, captions) in enumerate(train_loader):
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
        save_checkpoint(checkpoint, f"./checkpoints/final_checkpoint{epoch+1}.pth")

    return epoch_loss


def evaluate_model(model, dataset, device,val_loader,criterion):
    print('Evaluating...')
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    bleu_dict = {}
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

            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)
            
            actual_caption = ' '.join([dataset.vocab.itos[idx] for idx in captions.cpu().numpy() if idx not in {0, 1, 2}])
            well_pred = [1 for word in generated_caption.split() if (word in actual_caption.split())]
            bleu_score = len(well_pred)/len(actual_caption.split())
            
            if len(bleu_dict) < 20:
                bleu_dict[generated_caption] = bleu_score

            elif bleu_score > list(bleu_dict.values())[0]:
                min_key = min(bleu_dict, key=bleu_dict.get)
                bleu_dict.pop(min_key)
                bleu_dict[generated_caption] = bleu_score

            bleu_dict = dict(sorted(bleu_dict.items(), key=lambda x: x[1]))
        val_loss /= len(val_loader.dataset)
        #bleu_score /= len(val_loader.dataset)
        return val_loss, bleu_dict


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


    images_path_train = input("Enter the images path for training: ")
    annotations_path_train = input("Enter the annotations path for training: ")

    images_path_val = input('Enter the images path for validation: ')
    annotations_path_val = input('Enter the annotation path for validation: ')

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
    num_epochs = 20  # Number of training epochs
    num_layers = 2  # Number of layers in the RNN
    
    # Initialize the model, loss function, and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Ignore padding tokens in the loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    val_dataset.vocab = dataset.vocab
    
    load_model = False # You should change this to 'True' if you want to use a pretrained model with a checkpoint.
    
    if load_model == True:
        checkpoint_path = input('Enter checkpoint path: ')
        step = load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    #Keep track of the losses for each epoch
    total_loss = {'train': [], 'val': []}

    print('Aleluya')
    for epoch in range(num_epochs):
        print('Epoch:',epoch)
        train_loss = train(num_epochs,epoch,model,criterion,optimizer,train_loader)
        val_loss, bleu_dict = evaluate_model(model,val_dataset,device,val_loader,criterion)
        print(bleu_dict)
        total_loss["train"].append(train_loss)
        total_loss["val"].append(val_loss)
 
        
        print(total_loss['val'],total_loss['train'])

 
        plt.plot(total_loss["train"], label="training loss", color="blue")
        plt.plot(total_loss["val"], label="validation loss",color="red")
        plt.title("Train (Blue) and Validation (Red) loss")

        # Save the plot to a file
        plt.savefig('plot.png')

        # Show the plot
        plt.show()
