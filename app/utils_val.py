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

def load_checkpoint(checkpoint, model, optimizer):
    """
    Load the model checkpoint from a file.
    
    Args:
        checkpoint (dict): The checkpoint dictionary containing model parameters and optimizer state.
        model: The model to load the parameters into.
        optimizer: The optimizer to load the state into.
    
    Returns:
        step (int): The training step or iteration at which the checkpoint was saved.
    """
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    print("Checkpoint loaded!")
    return step