import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from PIL import Image
import matplotlib.pyplot as plt
from utils import save_checkpoint, load_checkpoint, print_and_export_examples
from get_loader import get_loader
from model import CNNtoRNN

def evaluate_model(model, dataset, device):
    model.eval()  # Set the model to evaluation mode
    bleu_scores = []  # Store the BLEU scores for each example

    with torch.no_grad(), open("evaluation_output.html", "w") as f:
        f.write("<html><body>")
        for idx in range(len(dataset)):
            image, captions = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Generate caption using the model
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            # Compute the BLEU score for each reference caption
            reference_captions = [' '.join(caption) for caption in captions]
            bleu_scores_per_caption = [
                sentence_bleu([reference_caption.split()], generated_caption.split())
                for reference_caption in reference_captions
            ]

            # Take the maximum BLEU score among all the reference captions
            max_bleu_score = max(bleu_scores_per_caption)
            bleu_scores.append(max_bleu_score)

            # Write the image and caption to the HTML file
            f.write("<h2>Example {}</h2>".format(idx))
            f.write("<img src='{}' width='300' height='300'><br>".format(dataset.get_image_path(idx)))
            f.write("<strong>Generated Caption:</strong> {}<br>".format(generated_caption))
            f.write("<strong>Reference Captions:</strong><br>")
            for caption in captions:
                f.write("- {}<br>".format(' '.join(caption)))
            f.write("<strong>BLEU Score:</strong> {:.4f}<br><br>".format(max_bleu_score))

        f.write("</body></html>")

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {average_bleu:.4f}")


def visualize_predictions(model, dataset, device, num_examples=5):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad(), open("evaluation_output.html", "a") as f:
        f.write("<h1>Visualizations</h1>")
        for idx in range(num_examples):
            image, captions = dataset[idx]
            original_image = transforms.ToPILImage()(image).convert("RGB")
            image = image.unsqueeze(0).to(device)

            # Generate caption using the model
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            # Write the image and caption to the HTML file
            f.write("<h2>Example {}</h2>".format(idx))
            f.write("<img src='{}' width='300' height='300'><br>".format(dataset.get_image_path(idx)))
            f.write("<strong>Generated Caption:</strong> {}<br>".format(generated_caption))
            f.write("<strong>Original Captions:</strong><br>")
            for caption in captions:
                f.write("- {}<br>".format(' '.join(caption)))
            f.write("<br>")

            # Display image
            plt.imshow(original_image)
            plt.axis("off")
            plt.show()


def main():
    # Define the image transformations for evaluation
    transform_eval = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # S paths to trained model checkpoint and evaluation dataset
    model_checkpoint_path = "checkpoint.pth"
    images_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/images/"
    annotations_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/captions.txt"

    # Set CUDA benchmark for improved performance
    torch.backends.cudnn.benchmark = True

    # Check if CUDA is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and data loader for evaluation
    eval_loader, eval_dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=transform_eval,
        num_workers=4,
        shuffle=False
    )

    # Load the trained model checkpoint
    checkpoint = torch.load(model_checkpoint_path)

    # Get the model parameters from the checkpoint
    embed_size = checkpoint["embed_size"]
    hidden_size = checkpoint["hidden_size"]
    vocab_size = checkpoint["vocab_size"]
    num_layers = checkpoint["num_layers"]

    # Create the model and load the checkpoint parameters
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # Evaluate the model
    evaluate_model(model, eval_dataset, device)

    # Visualize predictions
    visualize_predictions(model, eval_dataset, device, num_examples=5)


if __name__ == "__main__":
    main()



# Specify the paths to your trained model checkpoint and evaluation dataset
model_checkpoint_path = "checkpoint.pth"
images_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/images/"
annotations_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/captions.txt"

# Set CUDA benchmark for improved performance
torch.backends.cudnn.benchmark = True

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformations for evaluation
transform_eval = transforms.Compose([
    transforms.Resize((356, 356)),
    transforms.CenterCrop((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset and data loader for evaluation
eval_loader, eval_dataset = get_loader(
    root_folder=images_path,
    annotation_file=annotations_path,
    transform=transform_eval,
    num_workers=4,
    shuffle=False
)

# Load the trained model checkpoint
checkpoint = torch.load(model_checkpoint_path)

# Get the model parameters from the checkpoint
embed_size = checkpoint["embed_size"]
hidden_size = checkpoint["hidden_size"]
vocab_size = checkpoint["vocab_size"]
num_layers = checkpoint["num_layers"]

# Create the model and load the checkpoint parameters
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)

# Evaluate the model
evaluate_model(model, eval_dataset, device)