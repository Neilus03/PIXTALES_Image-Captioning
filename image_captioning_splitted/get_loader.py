# Import necessary libraries
import os
import pandas as pd
import torch
import spacy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

spacy_en = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            if not isinstance(sentence, str):  # Check if the sentence is not a string
                print(f"Unexpected non-string value in captions: {sentence}")
            else:
                for word in self.tokenizer_eng(sentence):
                    if word not in frequencies:
                        frequencies[word] = 1
                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=3, dataset='flickr30k'):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = dataset
        
        if self.dataset == 'COCO':
            self.df = pd.read_csv(captions_file)
            self.df = self.df.dropna()
            self.imgs = self.df['image_id']
            self.captions = self.df['caption']
            
        elif dataset == 'flickr8k' or dataset == 'flickr30k':
            self.df = pd.read_csv(captions_file)
            self.df = self.df.dropna()
            self.imgs = self.df['image']
            self.captions = self.df['caption']
        
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        caption = self.captions[id]
        img_id = self.imgs[id]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numerical_caption)


class Padding:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(root_folder, annotation_file, transform, dataset='flickr30k', batch_size=32, num_workers=4, shuffle=True,
               pin_memory=True):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform, dataset=dataset)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Padding(pad_idx=pad_idx)
    )

    return loader, dataset
