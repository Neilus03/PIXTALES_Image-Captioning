import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt
import os
import base64
from PIL import Image
from io import BytesIO

from utils import save_checkpoint, load_checkpoint, print_and_export_examples
from get_loader import get_loader
from model import CNNtoRNN

def evaluate_model(model, dataset, device):
    model.eval()  # Set the model to evaluation mode
    bleu_scores = []  # Store the BLEU scores for each example
    best_bleu_score = 0  # Keep track of the highest BLEU score

    # Create and open the HTML file for writing
    html_file = open("evaluation_results.html", "w")
    html_file.write("<html><body><h1>Evaluation Results</h1>")
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            #print(f"Evaluating example {idx+1}/{len(dataset)}")
            image, captions = dataset[idx]
            original_image = transforms.ToPILImage()(image).convert("RGB")  # Create the original image
            image = image.unsqueeze(0).to(device)

            # Generate caption using the model
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            # Convert the captions from tensors to lists of strings
            if isinstance(captions, list):
                reference_captions = [
                    list(map(dataset.vocab.itos.__getitem__, caption.tolist()))
                    for caption in captions
                ]
            else:
                reference_captions = [
                    list(map(dataset.vocab.itos.__getitem__, captions.tolist()))
                ]

            # Compute the BLEU score for each reference caption
            bleu_scores_per_caption = [
                sentence_bleu(reference_caption, generated_caption)
                for reference_caption in reference_captions
            ]

            # Take the maximum BLEU score among all the reference captions
            max_bleu_score = max(bleu_scores_per_caption)
            bleu_scores.append(max_bleu_score)

            # If this max BLEU score is higher than the best seen so far, print the details
            if max_bleu_score > best_bleu_score:
                print(f"Generated Caption: {generated_caption}")
                print("Reference Captions:")
                for reference_caption in reference_captions:
                    print(' '.join(reference_caption))
                print(f"BLEU Score for Example {idx+1}: {max_bleu_score:.4f}\n")
                best_bleu_score = max_bleu_score  # Update the best BLEU score

                # Save the current image to a buffer as a PNG
                buffer = BytesIO()
                original_image.save(buffer, format="PNG")
                buffer.seek(0)

                # Convert the PNG image to a base64 string
                img_base64 = base64.b64encode(buffer.getvalue()).decode()

                # Write the image, the generated caption, and the BLEU score to the HTML file
                html_file.write("<h2>Example " + str(idx+1) + "</h2>")
                html_file.write('<p><img src="data:image/png;base64,' + img_base64 + '"/></p>')
                html_file.write("<p>Generated Caption: " + generated_caption + "</p>")
                html_file.write("<p>Reference Captions:</p>")
                for reference_caption in reference_captions:
                    html_file.write("<p>" + ' '.join(reference_caption) + "</p>")
                html_file.write("<p>BLEU Score: " + str(max_bleu_score) + "</p>")

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {average_bleu:.4f}")

    # Close the HTML file
    html_file.write("</body></html>")
    html_file.close()

def denormalize(image):
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    return image * std + mean

def visualize_predictions(model, dataset, device, num_examples=5):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for idx in range(num_examples):
            print(f"Visualizing example {idx+1}/{num_examples}")
            image, captions = dataset[idx]
            original_image = transforms.ToPILImage()(denormalize(image)).convert("RGB")
            image = image.unsqueeze(0).to(device)

            # Check if captions is a list of tensors
            if isinstance(captions, list):
                captions = [[dataset.vocab.itos[idx] for idx in caption] for caption in captions]
            elif isinstance(captions, torch.Tensor):
                # If it's a tensor, convert it to a list first
                captions = [dataset.vocab.itos[captions.item()]]

            # Generate caption using the model
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            # Print original captions
            print("Original Captions:")
            for caption in captions:
                print(' '.join(caption))

            # Print generated caption
            print("Generated Caption:")
            print(generated_caption)

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

    # Specify the paths to the trained model checkpoint and evaluation dataset
    model_checkpoint_path = "/home/xnmaster/PIXTALES-2/checkpoint8.pth"
    images_path = "/home/xnmaster/PIXTALES-2/Images/"
    annotations_path = "/home/xnmaster/PIXTALES-2/captions.txt"

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
    embed_size = 256
    hidden_size = 256
    vocab_size = len(eval_dataset.vocab)
    num_layers = 1

    # Create the model and load the checkpoint parameters
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    # Evaluate the model
    print("Starting model evaluation...")
    evaluate_model(model, eval_dataset, device)

    # Visualize predictions
    print("\nStarting visualization of predictions...")
    visualize_predictions(model, eval_dataset, device, num_examples=5)


if __name__ == "__main__":
    main()
