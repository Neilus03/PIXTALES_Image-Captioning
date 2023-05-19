# Text -> Numbers
# Need of a vocabulary that maps each word to an id
# Set Up Pytorch dataset to load the data
# Set up padding of every batch (all examples need to have same seqlen)
# Set up DataLoader

# Import necessary libraries
import os               # For operating system related operations like path management
import pandas as pd     # For handling dataframes
import torch            # The main PyTorch library
import spacy            # For natural language processing tasks, in this case tokenization
from torch.nn.utils.rnn import pad_sequence  # For padding sequences to the same length
from torch.utils.data import DataLoader, Dataset  # For handling the dataset
from PIL import Image   # For handling image data
from torchvision import transforms # For resize of the images
from spacy import *

spacy_en = spacy.load("en_core_web_sm") # Load the English language model for SpaCy

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        

    def __len__(self):
        return len(self.itos)
    
	# Static method for tokenizing English text using SpaCy tokenizer
    @staticmethod 
    def tokenizer_eng(text):
        
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]
	
	# Method for building vocabulary
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 #idx starts at 4 because for [0:3] are reserved words "<PAD>", "<SOS>", "<EOS>" and "<UNK>"
		# Iterate through each sentence in the list
        for sentence in sentence_list:
            # Tokenize the sentence and iterate through each word
            for word in self.tokenizer_eng(sentence):
                # If the word is not already in frequencies dictionary, add it with a count of 1
                if word not in frequencies:
                    frequencies[word] = 1
                # If the word is already in the frequencies dictionary, increment its count
                else:
                    frequencies[word] += 1

                # If a word's frequency reaches the frequency threshold, add it to the vocabulary
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    # Method to convert text into numericalized tokens
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        tokens = []
        for token in tokenized_text:
            if token in self.stoi:
                # If a word is in the vocabulary, use its corresponding token
                tokens.append(self.stoi[token])
            else:
                # If a word is not in the vocabulary, use the token for "<UNK>"
                tokens.append(self.stoi["<UNK>"])
        return tokens

# Define the custom Dataset class
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir  # Base directory for images
        self.df = pd.read_csv(captions_file)  # Load the captions file into a dataframe
        self.transform = transform  # Transformations to apply to images

        # Get the image and caption columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize the vocabulary and build it
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)  # Return the total number of items in the dataset

    def __getitem__(self, id):
        caption = self.captions[id]  # Get the caption corresponding to the id (I think there might be an error bc there are 5 cations for the same id)
        img_id = self.imgs[id]  # Get the image id corresponding to the id
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")  # Open and convert the image

        if self.transform is not None:
            img = self.transform(img)  # Apply transformations to the image

        # Prepare caption: Start of Sentence + numericalized caption + End of Sentence
        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numerical_caption)  # Return the image and its corresponding numericalized caption 
    												 # (numericalized captions = [1,stoi(W1),...,stoi(Wn),2])
    
# Padding class is used to pad the captions to the same length for each batch
class Padding:
    def __init__(self, pad_idx):
        # pad_idx is the index of the <PAD> token in the vocabulary
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        # For each item in the batch, add a new dimension to the image tensor at the front
        imgs = [item[0].unsqueeze(0) for item in batch]
        # Concatenate the image tensors along the new dimension
        imgs = torch.cat(imgs, dim=0)
        # Extract the caption tensors from the batch
        targets = [item[1] for item in batch]
        # Pad the caption tensors along the sequence dimension (dimension 0) using the <PAD> token
        targets = pad_sequence(targets, batch_first = False, padding_value = self.pad_idx)
        
        # Return the batch of image tensors and the padded caption tensors
        return imgs, targets
    
# get_loader function is used to create a DataLoader for the dataset
def get_loader(root_folder, 
               annotation_file, 
               transform,  
               batch_size=256, 
               num_workers=4, 
               shuffle=True, 
               pin_memory=True,
               ):
    
    # Create a FlickrDataset object for the given root_folder and annotation_file, using the provided transform
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    # Get the index of the <PAD> token in the vocabulary
    pad_idx = dataset.vocab.stoi["<PAD>"]

    # Create a DataLoader for the dataset
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Padding(pad_idx=pad_idx)
    )

    # Return both the DataLoader and the dataset
    return loader, dataset

# The main function is the entry point of the script
def main():
    # Define the transform to be applied to the images
    transform = transforms.Compose(
        [
            # Resize the images to 224x224
            transforms.Resize((224, 224)),
            # Convert the images to PyTorch tensors
            transforms.ToTensor(),
        ]
    )

    # Create a DataLoader for the dataset located at the specified path
    images_path = input("Enter the images path (or press Enter to use the default path): ")
    annotations_path = input("Enter the annotations path (or press Enter to use the default path): ")

    if not images_path:
        images_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/images/"

    if not annotations_path:
        annotations_path = "/Users/nde-la-f/Documents/Image_caption/flickr8k/captions.txt"

    dataloader = get_loader(images_path, annotation_file=annotations_path, transform=transform)

    # Iterate over the DataLoader
    for idx, (imgs, captions) in enumerate(dataloader):
        # Print the shape of the image and caption tensors for each batch
        print(imgs.shape)
        print(captions.shape)

