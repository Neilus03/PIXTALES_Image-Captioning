# PIXTALES
 
Welcome to PIXTALES - The story behind the picture, a deep learning approach to image captioning. The magic behind this repository is all about connecting visuals to language - generating a narrative for every image. Our project is driven by the power of neural networks and deep learning, aiming to create meaningful and accurate descriptions for any image. 
 
## What is PIXTALES?

PIXTALES is a project that uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to create a model capable of describing the content of images. We utilize the CNN as an "encoder" to transform an input image into a complex feature representation, and the RNN acts as a "decoder", turning those features into a rich, human-readable text. 

## Goals

Our Objective is to implement different models with different configurations and structures that are able to train and provide some results, and highlight the best results. Another goal is to try different ways to test our model, using different metrics and testing it with different datasets such as COCO dataset or Flickr30k dataset, but in principle we are going to use the Flickr8k dataset.

## Getting Started

First, you need to set clone this repository in your local (or virtual) machine

```
git clone git@github.com:Neilus03/PIXTALES_Image-Captioning.git
```

You will need to install the needed libraries and set up a good environment with pytorch, spacy, PIL and many more. You will also need to download the en_core_web_sm language model using:

```
python -m spacy download en_core_web_sm
```

Next, to train the code, simply use:

```
python train.py
```
If you want to use our already trained model just load the checkpoint you can download from here:

```
path_to_download_checkpoint
```

and run evaluation:

```
python evaluation.py
```
## Code Structure

The code in this repository primarily consists of a model implementation (CNNtoRNN), a dataset loading function (get_loader), and a main file that orchestrates model evaluation. It provides utility functions for generating and visualizing image captions and calculating BLEU scores, a popular metric for evaluating the quality of generated text in comparison to reference text.

In `main.py`, the model is loaded from a checkpoint file and used to generate captions for a set of images. The original and generated captions are printed, and the average BLEU score across all images is computed.

## Model Evaluation

Our evaluation focuses on two areas: the BLEU score and a visual inspection of the generated captions. The BLEU score gives us a quantitative measure of our model's performance, while the visual inspection of the generated captions lets us qualitatively assess the model's output. (to run the app remember to change the port)

## Final Words

This project is an exciting journey into the intersection of computer vision and natural language processing. We hope that this project can serve as a helpful resource for those interested in image captioning, deep learning, and AI in general.

## References
[CS 152 NN—25: Attention: Image Captioning by Neil Rhodes](https://youtu.be/JTXPrjvhLl8)


## Contributors
Write here the name and UAB mail of the group members

Subject: Neural Networks and Deep Learning.

Degreee in Artificial Intelligence, 2n course.

UAB , 2023.

Authors:
  Neil De La Fuente
  Maiol Sabater
  Daniel Vidal
 
