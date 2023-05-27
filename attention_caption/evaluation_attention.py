import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.translate.ribes_score import sentence_ribes
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import os
import base64
from PIL import Image
from io import BytesIO

from utils_attention import save_checkpoint, load_checkpoint
from get_loader_attention import get_loader
from model_attention import CNNtoRNN


def evaluate_model(model, dataset, device):
    model.eval()  # Set the model to evaluation mode
    bleu_scores = []  # Store the BLEU scores for each example
    cider_scores = []  # Store the CIDEr scores for each example
    meteor_scores = []  # Store the METEOR scores for each example
    nist_scores = []  # Store the NIST scores for each example
    ribes_scores = []  # Store the RIBES scores for each example
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

            # Compute the CIDEr score
            cider_scores.append(compute_cider_score(reference_captions, generated_caption))

            # Compute the METEOR score
            meteor_scores.append(compute_meteor_score(reference_captions, generated_caption))

            # Compute the NIST score
            nist_scores.append(compute_nist_score(reference_captions, generated_caption))

            # Compute the RIBES score
            ribes_scores.append(compute_ribes_score(reference_captions, generated_caption))

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_cider = sum(cider_scores) / len(cider_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    average_nist = sum(nist_scores) / len(nist_scores)
    average_ribes = sum(ribes_scores) / len(ribes_scores)

    print(f"Average BLEU Score: {average_bleu:.4f}")
    print(f"Average CIDEr Score: {average_cider:.4f}")
    print(f"Average METEOR Score: {average_meteor:.4f}")
    print(f"Average NIST Score: {average_nist:.4f}")
    print(f"Average RIBES Score: {average_ribes:.4f}")

    # Close the HTML file
    html_file.write("</body></html>")
    html_file.close()


def compute_cider_score(reference_captions, generated_caption):
    # Create COCO object and set up the reference and hypothesis lists
    coco = COCO()
    coco.dataset['refs'] = [{'image_id': 0, 'caption': ref_caption} for ref_caption in reference_captions]
    coco.dataset['hyps'] = [{'image_id': 0, 'caption': generated_caption}]
    
    # Create COCOEvalCap object for scoring
    coco_eval = COCOEvalCap(coco)

    # Compute CIDEr score
    coco_eval.evaluate()

    # Return CIDEr score
    cider_score = coco_eval.eval['CIDEr']
    return cider_score


def compute_meteor_score(reference_captions, generated_caption):
    # Convert reference and hypothesis captions to strings
    reference_captions = [' '.join(ref_caption) for ref_caption in reference_captions]
    generated_caption = ' '.join(generated_caption)

    # Compute METEOR score
    meteor_score_value = meteor_score(reference_captions, generated_caption)
    return meteor_score_value


def compute_nist_score(reference_captions, generated_caption):
    # Convert reference and hypothesis captions to lists of lists of tokens
    reference_captions = [ref_caption.split() for ref_caption in reference_captions]
    generated_caption = generated_caption.split()

    # Compute NIST score
    nist_score = sentence_nist(reference_captions, generated_caption)
    return nist_score


def compute_ribes_score(reference_captions, generated_caption):
    # Convert reference and hypothesis captions to lists of lists of tokens
    reference_captions = [ref_caption.split() for ref_caption in reference_captions]
    generated_caption = generated_caption.split()

    # Compute RIBES score
    ribes_score = sentence_ribes(reference_captions, generated_caption)
    return ribes_score


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
            generated_caption, attention_maps = model.caption_image(image, dataset.vocab, return_attention=True)
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

            # Display attention maps
            plt.figure(figsize=(10, 10))
            num_words = len(generated_caption.split())
            for i in range(num_words):
                plt.subplot(num_words // 5 + 1, 5, i + 1)
                plt.imshow(attention_maps[i].squeeze().cpu(), cmap='gray')
                plt.axis("off")
                plt.title(generated_caption.split()[i])
            plt.tight_layout()
            plt.show()


def main():
    # Define the image transformations for evaluation
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Specify the paths to the trained model checkpoint and evaluation dataset
    model_checkpoint_path = "path/to/your/checkpoint.pth"
    images_path = "path/to/your/images/"
    annotations_path = "path/to/your/annotations.txt"

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
    embed_size = 512
    hidden_size = 512
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
