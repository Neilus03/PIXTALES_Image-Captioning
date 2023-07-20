# Image Captioning: An attention approach

This folder offers an attention based solution for Image Captioning, using a powerful combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), in this case also adding attention, that makes our model much more flexible and interpretable (Also handling better long sentences). Is worth noting that we wanted to implement our own attention mechanism without recurring to third´s pretrained models. We did this knowing that the quality could not be as good as with a pretrained model like BERT or GPT beacuse we wanted to see where we could get to.

## Table of Contents

- [Repository Structure](#Repository-Structure)
- [Requirements](#Requirements)
- [Installation and Usage](#Execution)
- [Architecture](#Architecture)
- [Extending the model](#Extending-the-model)
- [Guided Run-through](#Guided-Run-through)
- [Comments on Attention](#Comments-on-Attention)
- [Qualitative evaluation](#Qualitative-evaluation)


## Repository Structure

This repository includes the following key files:

- `model.py`: Defines the architecture of the model, featuring:
  - `encoderCNN`: Utilizes a VGG-16 CNN pre-trained on ImageNet to extract image features.
  - `decoderRNN`: Employs an LSTM network for decoding feature vectors into comprehensive captions.
  - `AdditiveAttention`: Executes additive attention between image feature vectors and previous hidden states.
  - `CNNtoRNN`: A unified model class that combines the functionality of the encoder and decoder.

- `get_loader.py`: Constructs DataLoader for the project, containing classes for vocabulary creation, dataset processing, and input padding.

- `utils.py`: Contains utility functions for model checkpoint saving and loading.

- `train.py`: Implements the training pipeline including model initialization, training loop, loss tracking, and model saving.

- `evaluation.py`: Provides both qualitative (image and predicted/actual captions) and quantitative (BLEU score) model evaluation.

## Requirements

### -Mandatory
1. Download the required images and captions from the provided [Flickr Dataset](https://drive.google.com/drive/folders/1x2pDFbpIsA8fCUomwBpo6wBy0Yw9LFF_?usp=sharing).
2. Here you can download the last checkpoint that will save you the training time: [attention checkpoint](https://drive.google.com/file/d/1-9Nod5adZa28ViHSY9QsmCNdV_hC_ofp/view?usp=sharing)

**We highly recommend and encourage you to directly go to the [Guided Run - Through](#Guided-Run-Through)**

### -In case you don´t want to go for the guided run through: 
Prior to running the code, please ensure you have the following:

3. Prepare an environment with necessary libraries including `pytorch`, `spacy`, `os`, `pandas`, `numpy`, `PIL (pillow)`.
4. Download Spacy's 'en_core_web_sm' model using `python -m spacy download en_core_web_sm`.


## Execution

**To avoid the whole manual process we really, highly recommend and encourage you to directly go to the [Guided Run - Through](#Guided-Run-Through)**

 In case you finally chose to avoid the guided run through: After setting up the environment and downloading the [requirements](#Requirements), execute the following command in the terminal (inside the `Ìmage_captioning_with_attention` folder) for training, also you can load the checkpoint for starting the training from the last checkpoint but it **may lead to overfitting**:
```
python train.py
```

The script will prompt you to enter the paths for the training and validation images, as well as the training and validation captions.

Once the training starts, you will see the progress in the terminal, including loss evolution, epoch, and step information.

At the end of each epoch, the model will be saved in a .pth file. After training completes, a plot showing the training and validation loss curves will be displayed.

Use the saved checkpoint to evaluate the model by running the `evaluation.py` script. It will ask you to enter the path to the model checkpoint and display generated captions along with their corresponding BLEU scores and ground truth captions.

To run `evaluation.py` is as easy as this:

```
python evaluation.py
```
## Architecture
We thought that it´d be nice to see a scheme of the architecture:
<img width="561" alt="Captura de pantalla 2023-05-30 065228" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/47a0ab20-d939-4bd1-acb5-139371030a27">

NN-layer = Attention layer

## Extending the Model

We encourage exploration and modifications to improve the model. We attempted different configurations and data sources in pursuit of optimal performance. Our optimal configuration was found to be:

- Pretrained CNN: VGG16 pretrained on imageNEt
- Embedding size: 128
- Hidden size: 512
- Batch size: 32
- Learning rate: 1e-4
- Number of LSTM layers: 1
- Scheduler: With gamma = 0.9 and step_size = 1
- Dropout: Included in RNN layer
- Batch normalization: Not included
- Weight Decay: 1e-3

## Guided Run-through

For a detailed walkthrough of the entire process, refer to our Python Notebook. It provides step-by-step instructions, from initialization to model evaluation, facilitating an effective understanding of the Image Captioning model:

[Full guided notebook](https://colab.research.google.com/drive/1oNQmnL9KO0TSOi4bNFqvnAgi3FdRgI8V?usp=sharing)

## Comments on Attention

Attention works amazingly fine, that fine that it overfits too fast, that´s why we had to add an exagerated amount of regularization, trying to control the beast. Luckily we were able to avoid some of this overfitting and it should generalize well. Nevertheless we tested on training set due to time issues and we were not able to make it for the attention maps, which would have been great for explainability.

## Qualitative evaluation

A further qualitative evaluation for our image captioning model with attention can be found on these slides: [Qualitative evaluation attention](https://docs.google.com/presentation/d/12JIvFaUerl8xQFJnY5fT7xTXw3kX638slbluTsOe_fc/edit?usp=sharing)

But here we leave some:

<img width="537" alt="Captura de pantalla 2023-05-30 010402" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/39ac7196-de89-4172-8618-3a55e7ad2a04">

<img width="567" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/320be3f8-4d90-4ac4-88be-6678c25a6335">

<img width="564" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/d7f0b38c-075e-4493-9a65-303b7c79c6ae">

<img width="541" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/3c7f3c5c-d833-4c15-8774-0afa03928247">


The quantitative analysis is done looking at the BLEU scores (our own built BLEU-1 scoring function)
