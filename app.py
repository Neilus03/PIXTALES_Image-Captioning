import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from flask import Flask, request, jsonify
from model import CNNtoRNN
from get_loader import get_loader


app = Flask(__name__)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model checkpoint
model_checkpoint_path = "C:/Users/34717/Downloads/PIXTALES/checkpoint14.pth"
checkpoint = torch.load(model_checkpoint_path, map_location = device)


transform_eval = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.CenterCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

model_checkpoint_path = "C:/Users/34717/Downloads/PIXTALES/checkpoint14.pth"
images_path = "C:/Users/34717/Desktop/DL/Project/dlnn-project_ia-group_05/Images"
annotations_path = "C:/Users/34717/Desktop/DL/Project/dlnn-project_ia-group_05/captions.txt"

eval_loader, eval_dataset = get_loader(
        root_folder=images_path,
        annotation_file=annotations_path,
        transform=transform_eval,
        num_workers=4,
        shuffle=False)
 
        
# Get the model parameters from the checkpoint
embed_size = 256
hidden_size = 256
vocab_size = len(eval_dataset.vocab)
num_layers = 1

# Create the model and load the checkpoint parameters
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
model.eval()



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Load and preprocess the image
    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image = transform_evalage).unsqueeze(0).to(device)

    # Generate caption using the model
    generated_caption = model.caption_image(image)
    generated_caption = ' '.join(generated_caption)

    return jsonify({'caption': generated_caption})

if __name__ == '__main__':
    app.run(debug = True)
