import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def evaluate(model, loader, dataset, device):
    model.eval()  # Put the model in evaluation mode

    for idx, (images, captions) in enumerate(loader):
        images = images.to(device)  # Move the images to the device
        captions = captions.to(device)  # Move the captions to the device

        outputs = model(images, captions)  # Forward pass
        _, preds = torch.max(outputs, dim=2)  # Get the predicted captions

        pred_caption = ' '.join([dataset.vocab.itos[i] for i in preds[0].cpu().numpy()])
        actual_caption = ' '.join([dataset.vocab.itos[i] for i in captions[0].cpu().numpy() if i not in {0, 1, 2}])

        print(f"Image {idx + 1}")
        print(f"Predicted Caption: {pred_caption}")
        print(f"Actual Caption: {actual_caption}")
        
        # Convert the PyTorch tensor to a PIL image and display it
        img = transforms.ToPILImage()(images[0].cpu())
        plt.imshow(img)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Provide your path to root directory and captions file
    root_dir = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr8k/Images"
    captions_file = "/content/drive/MyDrive/Machine and Deep Learning /Deep Learning Project/archive/flickr8k/captions.txt"

    # TODO: Load your trained model
    checkpoint_path = "/content/drive/MyDrive/Machine and Deep Learning /Deep Learning Project/checkpoint_attention3_pro.pth"
    
    checkpoint = torch.load(checkpoint_path)

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)


    transform = transforms.Compose(
        [
            # Resize to the size used in training
            transforms.Resize((224,224)),
            # Convert images to PyTorch tensors
            transforms.ToTensor(),
        ]
    )
    
    # Create a DataLoader for the dataset
    loader, dataset = get_loader(root_dir, captions_file, transform)

    # Evaluate the model
    evaluate(model, loader, dataset, device)
