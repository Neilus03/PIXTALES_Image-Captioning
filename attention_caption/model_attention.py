import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_out, decoder_hidden):
        # Compute dot product between encoder output and decoder hidden state
        attention = torch.bmm(encoder_out, decoder_hidden)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attention, dim=1)

        # Compute context vector
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_out)

        return context, attn_weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=True, p_dropout=0.5):
        super(EncoderCNN, self).__init__()

        self.train_CNN = train_CNN
        
        # Load the pretrained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Replace the fully connected layer with a new linear layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

    def forward(self, images):
        # Extract features from the images using the ResNet-50 model
        features = self.resnet(images)

        # Set the requires_grad attribute of parameters based on the train_CNN flag
        for name, parameter in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = self.train_CNN

        # return the features (in this case I tried without applying ReLU activation function and dropout )
        return features    		


class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, p_dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        # Embedding layer to convert word indices to dense vectors
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer to process the embedded captions
        self.lstm = nn.LSTM(embed_size*2, hidden_size, num_layers)

        # Linear layer to convert hidden states to output logits
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(p_dropout)

        # Attention layer
        self.attention = Attention()

    def forward(self, features, captions, encoder_out):
        # Embed the captions
        embeddings = self.dropout(self.embed(captions))

        # Apply Attention mechanism
        context, attn_weights = self.attention(encoder_out, features.unsqueeze(0))
        
        # Concatenate the context vector and embedded captions
        embeddings = torch.cat((context.squeeze(1), embeddings), dim=2)

        # Pass the embeddings through the LSTM layer
        hiddens, _ = self.lstm(embeddings)

        # Convert the hidden states to output logits
        outputs = self.linear(hiddens)

        return outputs, attn_weights
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        # Instantiate the EncoderCNN module
        self.EncoderCNN = EncoderCNN(embed_size)

        # Instantiate the DecoderWithAttention module
        self.DecoderWithAttention = DecoderWithAttention(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Pass the images through the EncoderCNN module to extract features
        features = self.EncoderCNN(images)

        # Pass the features and captions through the DecoderWithAttention module
        outputs, attn_weights = self.DecoderWithAttention(features, captions)
        
        return outputs, attn_weights
    
    
    def caption_image(self, image, vocabulary, max_length=20):
        
        result_caption = []
        attn_weights_list = []  # Store attention weights for each timestep
        
        with torch.no_grad():
            # Encode the input image using the EncoderCNN module
            x = self.EncoderCNN(image).unsqueeze(0)
            h, c = self.DecoderWithAttention.lstm.init_hidden_state(x)
            
            for _ in range(max_length):
                # Generate the embeddings for the current timestep
                embeddings = self.DecoderWithAttention.embed(torch.tensor(
                    [[vocabulary.stoi["<SOS>"]]]
                ))
                
                # Compute the attention-weighted encoding using the attention mechanism
                attention_weighted_encoding, attn_weights = self.DecoderWithAttention.attention(
                    x, h[-1]
                )
                
                attn_weights_list.append(attn_weights.squeeze().cpu().numpy())
                
                # Pass the embeddings and attention-weighted encoding through the LSTM layer
                h, c = self.DecoderWithAttention.lstm(
                    torch.cat([embeddings.squeeze(1), attention_weighted_encoding],dim=1),(h, c))
                
                # Generate the output logits using the linear layer
                preds = self.DecoderWithAttention.linear(h)
                
                # Get the predicted word by selecting the word with the highest logit value
                _, predicted = preds.max(1)
                result_caption.append(predicted.item())
                
                # Check if the predicted word is the end-of-sequence token ("<EOS>")
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        
        #store the attn weights in an array, we might use them later for visualization of the attention maps
        attn_weights_list = np.array(attn_weights_list)        
        
        # Convert the list of word indices to a list of actual words using the vocabulary
        return [vocabulary.itos[idx] for idx in result_caption[1:]]

