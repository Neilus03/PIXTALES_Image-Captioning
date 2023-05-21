import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

from utils import save_checkpoint, load_checkpoint, print_and_export_examples
from get_loader import get_loader
from model import CNNtoRNN


def evaluate_model(model, dataset, device):
    model.eval()  # Set the model to evaluation mode
    bleu_scores = []  # Store the BLEU scores for each example

    with torch.no_grad():
        for idx in range(len(dataset)):
            print(f"Evaluating example {idx+1}/{len(dataset)}")
            image, captions = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Generate caption using the model
            generated_caption = model.caption_image(image, dataset.vocab)
            generated_caption = ' '.join(generated_caption)

            print(f"Generated Caption: {generated_caption}")

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

            print("Reference Captions:")
            for reference_caption in reference_captions:
                print(' '.join(reference_caption))

            # Compute the BLEU score for each reference caption
            bleu_scores_per_caption = [
                sentence_bleu(reference_caption, generated_caption)
                for reference_caption in reference_captions
            ]

            # Take the maximum BLEU score among all the reference captions
            max_bleu_score = max(bleu_scores_per_caption)
            bleu_scores.append(max_bleu_score)

            print(f"BLEU Score for Example {idx+1}: {max_bleu_score:.4f}\n")

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU Score: {average_bleu:.4f}")


def visualize_predictions(model, dataset, device, num_examples=5):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for idx in range(num_examples):
            print(f"Visualizing example {idx+1}/{num_examples}")
            image, captions = dataset[idx]
            original_image = transforms.ToPILImage()(image).convert("RGB")
            image = image.unsqueeze(0).to(device)

            # Convert the captions from indices to words
            captions = [[dataset.vocab.itos[idx] for idx in caption] for caption in captions]

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
    model_checkpoint_path = "/home/xnmaster/PIXTALES-2/checkpoint.pth"
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
