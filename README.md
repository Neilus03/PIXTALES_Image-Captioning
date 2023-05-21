# PIXTALES

Welcome to PIXTALES - The story behind the picture, a deep learning approach to image captioning. The magic behind this repository is all about connecting visuals to language - generating a narrative for every image. Our project is driven by the power of neural networks and deep learning, aiming to create meaningful and accurate descriptions for any image.

## What is PIXTALES?

PIXTALES is a project that uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to create a model capable of describing the content of images. We utilize the CNN as an "encoder" to transform an input image into a complex feature representation, and the RNN acts as a "decoder", turning those features into a rich, human-readable text.

## Getting Started

First, you need to set up your local environment. We use conda for managing dependencies. After cloning this repository, create a new conda environment using the provided environment.yml file:

```
conda env create --file environment.yml
conda activate xnap-example
```

Next, to run the example code, simply use:

```
python main.py
```

## Code Structure

The code in this repository primarily consists of a model implementation (CNNtoRNN), a dataset loading function (get_loader), and a main file that orchestrates model evaluation. It provides utility functions for generating and visualizing image captions and calculating BLEU scores, a popular metric for evaluating the quality of generated text in comparison to reference text.

In `main.py`, the model is loaded from a checkpoint file and used to generate captions for a set of images. The original and generated captions are printed, and the average BLEU score across all images is computed.

## Model Evaluation

Our evaluation focuses on two areas: the BLEU score and a visual inspection of the generated captions. The BLEU score gives us a quantitative measure of our model's performance, while the visual inspection of the generated captions lets us qualitatively assess the model's output.

## Final Words

This project is an exciting journey into the intersection of computer vision and natural language processing. We hope that this project can serve as a helpful resource for those interested in image captioning, deep learning, and AI in general.

## Contributors
Write here the name and UAB mail of the group members

Xarxes Neuronals i Aprenentatge Profund
Grau de __Write here the name of your estudies (Artificial Intelligence, Data Engineering or Computational Mathematics & Data analyitics)__, 
UAB, 2023


Authors:
  Neil De La Fuente
  Maiol Sabater
  Daniel Vidal
