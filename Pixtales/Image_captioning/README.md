# Image Captioning

## Content

This folder contains the code and files for the Image Captioning project, using a CNN and RNN. 

## Table of Contents

- [Files](#Files)
- [Requirements](#Requirements)
- [Installation and Usage](#Execution)
- [Using different datasets](#Using-Different-Datasets)
- [Configuration](#Trying-Different-Configurations)
- [APP](#APP)
- [QUANTITATIVE and QUALITATIVE ANALYSIS](#QUANTITATIVE-and-QUALITATIVE-ANALYSIS)


### Files

- **model.py**: This file contains the structure of the model in different classes. The classes include:
  - `encoderCNN`: A ResNet CNN pretrained on ImageNet that extracts features from the images.
  - `decoderRNN`: An LSTM network that generates words recursively using the image features and captions as inputs.
  - `CNNtoRNN`: Combines the functionality of `encoderCNN` and `decoderRNN` to encode images and captions and produce codified sentences.

- **get_loader.py**: This file contains a function to create data loaders for the project. It includes the following classes:
  - `Vocabulary`: Creates the vocabulary by tokenizing words in the captions using the "en_core_web_sm" language model from spacy.
  - `FlickrDataset`: Generates a data loader by loading image filenames, captions, and the vocabulary.
  - `Padding`: Adds padding to captions in a batch to ensure uniform length.

- **utils.py**: This file contains two utility functions:
  - `save_checkpoint`: Saves the model in a .pth file.
  - `load_checkpoint`: Loads a pre-trained model from a .pth file.

- **train.py**: This script trains an image captioning model using a CNN-to-RNN architecture. It loads the image and caption datasets, creates data loaders, initializes the model, loss function, and optimizer. The script performs the training loop for a specified number of epochs, logs the training loss using TensorBoard, and saves the final model checkpoint and training loss curve.

- **evaluation.py**: This script evaluates the trained model. It prompts the user to enter the path to the trained model checkpoint, generates captions for sample images, and displays the generated captions, BLEU scores, and ground truth captions for comparison.

### Requirements

To run this project, you need to download the image and caption datasets. The datasets are available for download using the following link: [flickr8k Dataset](https://drive.google.com/drive/folders/1skoIZFClsh_Ol-wiwG_Foo53BQF8KOMW?usp=sharing). Download the training images and captions, as well as the validation images and captions as separate files.

You will also need the following libraries, packages, and frameworks:
- PyTorch
- Spacy
- OS
- Pandas
- NumPy
- PIL (Pillow)

Download the "en_core_web_sm" language model from spacy by running the following command:
```
python -m spacy download en_core_web_sm
```

### Execution

After setting up the environment and downloading the language model, execute the following command in the terminal:
```
python train.py
```

The script will prompt you to enter the paths for the training and validation images, as well as the training and validation captions.

Once the training starts, you will see the progress in the terminal, including loss evolution, epoch, and step information.

At the end of each epoch, the model will be saved in a .pth file. After training completes, a plot showing the training and validation loss curves will be displayed.

Use the saved checkpoint to evaluate the model by running the `evaluation.py` script. It will prompt you to enter the path to the model checkpoint and display generated captions along with their corresponding BLEU scores and ground truth captions.

### Using Different Datasets

We attempted to train our model with different datasets such as COCO or Flickr30k to improve its performance and robustness. However, we encountered issues with data loading and different data formats. Despite attempts to modify and upload the images in different formats, we faced problems during training when loading certain images.

### Trying Different Configurations

During the project, we experimented with various configurations to optimize the model's performance. We tried different embedding and hidden sizes, learning rates, batch sizes, number of layers, and training with or without a scheduler. After exploring these options, we found the following configuration to yield the best results:

- Embedding size: 256
- Hidden size: 256
- Batch size: 32
- Learning rate: 3e-4
- Number of layers: 2
- Scheduler: False
- Dropout: 0.5
- Batch normalization: True

Feel free to modify these parameters according to your specific needs and dataset characteristics to achieve the desired results.

## APP

Finally, we wanted some visual way to show the results of our model. And we thought that creating an 'app' could be a great way to do it. The app folder consists in 3 python files:

- **app.py**: This file contains the main commands and executations for the app to work, it holds the structure:
  - Reads the model, you can download the last model checkpoint we have trained using the following link: [last model](https://drive.google.com/file/d/1n7GJIqaEWQTSJlSO5kmS-h7QU8HNO6Th/view?usp=sharing) , also note that the parameters of the model such as nº of layers, embeding size, hidden size and the vocabulary's length must be equal to whatever model you are reading
  - Perform the necessary calculations and calls the necessary files and functions
  - Alocates the port the web will be hosted
  - Defines which file is the home file, which html in this case should read.
  - Generates the captions for the given images

- **model_val.py**: This file contains the structure of the model in different classes. The classes include:
  - `encoderCNN`: A ResNet CNN pretrained on ImageNet that extracts features from the images.
  - `decoderRNN`: An LSTM network that generates words recursively using the image features and captions as inputs.
  - `CNNtoRNN`: Combines the functionality of `encoderCNN` and `decoderRNN` to encode images and captions and produce codified sentences.

- **utils_val**: This file has only 2 main things:
  - `load_checkpoint`: This function allows us to load the model.
  - `Vocabulary class`: to give to the loaded model.
  
Besides those 3 files, the app folder also has 3 folders:

- **__pycache__**: This folder is not used directly, it creates shortcuts so the app does not have to run from 0 every time we execute it. This is done automatically.

- **static**: This folder has 2 things:
  - `Background`: The picture for the background of the app
  - `Uploades`: This is a folder where all the uploaded pictures in the app will be alocated.
  
- **templates**: This folder has only 1 file wich is called index, this index file is an html with the structrue of the whole interface for the app.

And finally a .txt file:

- **captions.txt**: The file that contains all the captions from where the vocab will be extracted.

#### APP WORKING PROPERLY

- All the previous requirements are recommended to have, and also for the app, we will need to install another library called `flask`. So besides all the other libraries, packages and frameworks the app will be also using `flask`.

- A checkpoint is needed you can download it from the last model link in the app.py explanation. Just take the file and add it into the app folder.

- When runing the `app.py` file, some warnings will show up in the console, do not worry those warnings are because of loading the model. When the model is loaded you should see "Checkpoint loaded!" on your screen, after that a url will appear in the form of "http://localhost:5000" or "http://127.0.0.1:5000", acces to that url and if everything is correct, the app interface should appear. 

- When dealing with images, just press the select image button and an image from the local machine will be chosen. After that, click the upload button and just wait for the caption and the image to pop up. 

- If another model is trying to be loaded and there is no sucess, check in the `app.py` if the parameters of the model are equal to the model the user is trying to load.


#### APP POSSIBLE ERRORS AND CLARIFICATIONS

For correct usage, we want to execute the `app.py` file from the previous directory of the app folder, in this case from Image_captioning because of how the directories are implemented in the code. Otherwise if facing problems with directories, all the directories that must be changed for proper working are in the `app.py` file. 

Also it can happen that the port where the web has to be located is already used, some warning saying so will show up. To solve this issue just go ahead to the very last line of code of the `app.py` and change the port variable to change the port where the web will be alocated.

If for some reason the app provided here does not work, try downloading the app from the [google drive](https://drive.google.com/drive/folders/19pBiWG4_G3fbP5SOmNv2oCtIgJYS2QBd?usp=sharing) (The checkpoint is already on the folder).

## QUANTITATIVE and QUALITATIVE ANALYSIS

As the quantitative analysis we will be looking into 2 different things: The plot we obtained with both losses and the bleu score generated at the end of each validation, then for the qualitative analysis we´ll have a look to the captions generated by the model.

Starting with the plot, first we need to see how both losses behavead:  

<img width="479" alt="image" src="https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/87651732/2cd286a9-d828-48ac-9879-c5bbea83198f">


As we see, both losses go down slowly but steady. Of course the training loss is lower than the validation loss but still both were going down. This slowlyness can mean some things, either the learning rate is low, or we landed in a semi-plateaux and it is getting hard to get out of there because the learning rate is low or because the model would need more expresivity to perform better , we think it may be this, also we should note that we have tried different hyperparameters and we had models that worked better before, nonetheless we decided to leave this because it had a proper validation and was not being validated on the training set, this also means that the model is being trained with less pictures which also leads to lower performance.

Now we will look into the the bleu score: for the last epoch we can see some interesting captions for the images

{'<SOS> a man in a blue shirt and a black shirt is standing in front of a crowd . <EOS>': 0.9090909090909091,
  
 '<SOS> a man in a black shirt and a black hat is standing in front of a building . <EOS>': 1.0,
  
 '<SOS> a man in a blue shirt is standing on a bench with a <UNK> in the background . <EOS>': 0.928571428571428,
  
 '<SOS> a man and a woman are sitting on a bench in front of a building . <EOS>': 0.82,
  
 '<SOS> a man in a blue shirt is standing on a bench with a man in a black shirt . <EOS>': 1.0}
  
This is a dictionary with the caption as key and the bleu score as value, we can see that most of the setences could be perfectly a valid caption except for some of them that repeat words or mention that a man has 2 different shirts or other things that do not make a lot of sense but because the bleu score is comparing words, they are still a pretty valid ones since most of the sentece could be a caption.
  
And finally the image caption evaluation. Remember how we said the loss would get down really slow, about 0.03 for each epoch at the end. Captions, can be divided between captions that have no relation with the image, other captions recognize some objects, colors or places but the rest of the caption is quite wrong and captions that have nothing to do with the actual image. Note that the model has some preferences for some objects, colors or places. This bias is due to the abundance of this object in the dataset, for example: there are a lot of dogs in the dataset and if an image has a dog the model will most likely detect the dog. Also note that semantically the model is not 100% correct, so some errors may be generated, for example: if a dog is swiming it may say that the dog is running through the water, but no dog can run through the water, they swim. With this said, let's look into some examples:
  
![image](https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/132783746/b823dda8-ab7f-4552-9e16-55d7e07b4c9e)

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/132783746/e31de0bd-0a7f-44be-8795-562653c195ad)

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/132783746/c13e23f3-c196-4a4d-8bfa-13198e8610f0)

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/132783746/cb8e2479-f8d5-4f7f-8e1c-76e079d05403)

![image](https://github.com/DCC-UAB/dlnn-project_ia-group_05/assets/132783746/89785cd1-9a35-4cac-9299-b7aa0809e352)









