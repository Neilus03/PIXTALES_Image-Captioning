import torch
import torch.nn as nn
import torchvision.models as models

import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False, p_dropout=0.5):
        super(EncoderCNN, self).__init__()

        # Load the pretrained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Replace the fully connected layer with a new linear layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

        # Activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        
  
    def forward(self, images):
        # Extract features from the images using the ResNet-50 model
        features = self.resnet(images)

        # Set the requires_grad attribute of parameters based on the train_CNN flag
        for name, parameter in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = self.train_CNN

        # Apply ReLU activation function and dropout to the features
        return self.dropout(self.relu(features))    					
		
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, p_dropout=0.5):
        super(DecoderRNN, self).__init__()

        # Embedding layer to convert word indices to dense vectors
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer to process the embedded captions
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        # Linear layer to convert hidden states to output logits
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, features, captions):
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))

        # Concatenate the features and embedded captions
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        # Pass the embeddings through the LSTM layer
        hiddens, _ = self.lstm(embeddings)

        # Convert the hidden states to output logits
        outputs = self.linear(hiddens)

        return outputs
    



class CNNtoRNN(nn.Module):
    