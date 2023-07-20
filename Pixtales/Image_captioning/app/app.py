from flask import Flask, request, jsonify, render_template, url_for
import os
import torch
import model_val as model
from torchvision import transforms
from PIL import Image
from utils_val import Vocabulary,load_checkpoint
import pandas as pd
import torch.optim as optim

app = Flask(__name__)
UPLOAD_FOLDER = './app/static/uploads'  # Update the path to include the 'static' folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Load the caption generator model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.CNNtoRNN(256,256,2994,1)
optimizer = optim.Adam(model.parameters(), lr=3e-4 )
step = load_checkpoint(torch.load("./app/checkpoint22.pth"), model, optimizer)
model.to(device)
model.eval()


transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),  # Resize the image to a specific size
            transforms.RandomCrop((299, 299)),  # Randomly crop the image
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image tensor
        ]
    )
df = pd.read_csv('./captions.txt')  # Load the captions file into a dataframe
vocab = Vocabulary(5)
vocab.build_vocabulary(df['caption'].tolist())

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Access the uploaded image data from the request
    image = request.files['image']

    # Save the image to a temporary location
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)
    with torch.no_grad():

        # Preprocess the image
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        # Generate caption using the model
        caption = model.caption_image(img, vocab)
        caption = caption[1:-1]
        caption = " ".join(caption)
        print(type(caption),caption)
        
        # Generate the image URL for displaying in the HTML template
        image_url = url_for('static', filename='uploads/' + image.filename)

    return jsonify({'caption': caption, 'image_url': image_url})


if __name__ == '__main__':
    app.run(port = 6500)
