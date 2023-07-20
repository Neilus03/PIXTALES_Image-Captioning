from utils import save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import CNNtoRNN


import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from nltk.translate.bleu_score import sentence_bleu

def evaluate(model, loader, dataset, device):
    model.eval()  # Put the model in evaluation mode

    for idx, (images, captions) in enumerate(loader):
        images = images.to(device)  # Move the images to the device
        captions = captions.to(device)  # Move the captions to the device

        with torch.no_grad():
            outputs, att_weights = model(images, captions)  # Forward pass
            _, preds = torch.max(outputs, dim=2)  # Get the predicted captions

            for i in range(preds.shape[0]):
                pred_caption = []
                for j in range(preds.shape[1]):
                    word_idx = preds[i, j].item()
                    word = dataset.vocab.itos[word_idx]
                    if word == "<EOS>":
                        break
                    pred_caption.append(word)

                if pred_caption[0] == "<SOS>": # this should happen always
                    pred_caption.pop(0)
                pred_caption = ' '.join(pred_caption)
                actual_caption = ' '.join([dataset.vocab.itos[idx] for idx in captions[i].cpu().numpy() if idx not in {0, 1, 2}])

                print(f"Predicted Caption: {pred_caption}")
                print(f"Actual Caption: {actual_caption}")
                well_pred = [1 for word in pred_caption.split() if (word in actual_caption.split())]
                print("BLEU-1 SCORE: ",len(well_pred)/len(actual_caption.split()))
                # Convert the PyTorch tensor to a PIL image and display it
                img = transforms.ToPILImage()(images[i].cpu())
                plt.imshow(img)
                plt.axis('off')
                plt.show()

if __name__ == "__main__":
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = "/content/drive/MyDrive/IMAGE CAPTIONING/flickr8k_images"
    captions_file = "/content/drive/MyDrive/IMAGE CAPTIONING/captions.txt"

  
    checkpoint_path = input("checkpoint path:")
    #checkpoint_path = "/content/drive/MyDrive/IMAGE CAPTIONING/checkpoint_attention4.pth"
    
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

if __name__ == "__main__":
    main()
