from flask import Flask, request, jsonify, render_template, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'  # Update the path to include the 'static' folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

    # Generate the image URL for displaying in the HTML template
    image_url = url_for('static', filename='uploads/' + image.filename)

    return jsonify({'caption': 'A sample caption for the uploaded image', 'image_url': image_url})


if __name__ == '__main__':
    app.run()
