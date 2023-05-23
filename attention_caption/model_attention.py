
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())[:-1]  # remove the last max pooling layer
        self.features = nn.Sequential(*features)
        self.reduce_dim = nn.Conv2d(512, 256, kernel_size=1)  # 1x1 convolution to reduce the depth to 256

    def forward(self, images):
        features = self.features(images)  # Shape: [batch_size, 512, 14, 14]
        features = self.reduce_dim(features)  # Shape: [batch_size, 256, 14, 14]
        features = features.permute(0, 2, 3, 1)  # Shape: [batch_size, 14, 14, 256]
        features = features.view(features.size(0), -1, features.size(-1))  # Shape: [batch_size, 196, 256]
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(encoder_dim + decoder_dim, decoder_dim)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, features, hidden):
        attention = self.attention(torch.cat([hidden, features], dim=1))
        v = self.v(torch.tanh(attention))
        alpha = torch.softmax(v, dim=1)

        attention_weighted_encoding = torch.sum(alpha * features, dim=1)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        # Define the dimensions
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Attention layer
        self.attention = Attention(embed_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers)  # * 2 because we concatenate attention weighted encoding

        # Linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        attention_weighted_encoding, alpha = self.attention(features, embeddings[:, -1, :])
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs, alpha


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        # Instantiate the EncoderCNN module
        self.EncoderCNN = EncoderCNN()

        # Instantiate the DecoderRNN module
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.EncoderCNN(images)
        outputs = self.DecoderRNN(features, captions)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        alpha_list = []

        with torch.no_grad():
            x = self.EncoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                h, alpha = self.DecoderRNN.lstm(x, states)
                output = self.DecoderRNN.linear(h.squeze(0))
                _, predicted = output.max(1)
                result_caption.append(predicted.item())
                alpha_list.append(alpha)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                x = self.DecoderRNN.embed(predicted).unsqueeze(0)

        return [vocabulary.itos[idx] for idx in result_caption], alpha_list








#This is the only chaanged file to add attention, train, evaluation, get_loader, and utils remain the same, just changing imports
'''
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
        outputs, attn_weights = self.DecoderWithAttention(features, captions, features) #the last argument is the encoder_out, but as it is the same size as the features I just wrote features 2 times 
        
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

'''


'''import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_out, decoder_hidden):
        print("encoder_out shape: ", encoder_out.shape)
        print("decoder_hidden shape: ", decoder_hidden.shape)

        # Reshape the decoder hidden state (batch_size, hidden_size) -> (batch_size, hidden_size, 1)
        decoder_hidden = decoder_hidden.unsqueeze(2)
        print("decoder_hidden shape after applying unsqueeze(2): ", decoder_hidden.shape)

        # Transpose encoder_out to match the dimensions for matrix multiplication
        # encoder_out: (batch_size, channels) -> (batch_size, 1, channels)
        encoder_out = encoder_out.unsqueeze(1)
        print("encoder_out shape after unsqueezing(1): ", encoder_out.shape)

        # Compute dot product between encoder output and decoder hidden state
        # attention: (batch_size, 1, channels) x (batch_size, channels, 1) -> (batch_size, 1, 1)
        attention = torch.bmm(encoder_out, decoder_hidden)
        print("attention shape: ", attention.shape)

        # Apply softmax to get attention weights
        # attn_weights: (batch_size, 1, 1)
        attn_weights = torch.softmax(attention, dim=1)
        print("attn_weights shape: ", attn_weights.shape)

        # Compute context vector
        # context: (batch_size, 1, channels) x (batch_size, 1, 1) -> (batch_size, 1, channels)
        context = torch.bmm(encoder_out.transpose(1, 2), attn_weights)
        print("context shape: ", context.shape)

        # Squeeze dimensions for proper output shape
        # context: (batch_size, channels)
        context = context.squeeze(1)
        print("context shape after squeezing(1): ", context.shape)

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

        # Return the features
        return features


class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, p_dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        # Embedding layer to convert word indices to dense vectors
        self.embed = nn.Embedding(vocab_size, embed_size)

        # LSTM layer to process the embedded captions
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers)

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
        context, attn_weights = self.attention(encoder_out, features)

        # Adjust dimensions before concatenation
        embeddings = embeddings.permute(0, 2, 1)  # (batch_size, embed_size, seq_length)
        context = context.unsqueeze(1)  # (batch_size, 1, hidden_size)
        context = context.expand(embeddings.size(2), 1, embeddings.size(1), 1)  # (1, batch_size, seq_length, hidden_size)

        # Concatenate the context vector and embedded captions
        embeddings = torch.cat((embeddings, context), dim=1)

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
        outputs, attn_weights = self.DecoderWithAttention(features, captions, features)

        return outputs, attn_weights


    def caption_image(self, image, vocabulary, max_length=20):
        result_caption = []
        attn_weights_list = []

        with torch.no_grad():
            # Encode the input image using the EncoderCNN module
            x = self.EncoderCNN(image).unsqueeze(0)
            h, c = torch.zeros(self.DecoderWithAttention.lstm.num_layers, 1, self.DecoderWithAttention.lstm.hidden_size), torch.zeros(self.DecoderWithAttention.lstm.num_layers, 1, self.DecoderWithAttention.lstm.hidden_size)

            for _ in range(max_length):
                # Generate the embeddings for the current timestep
                embeddings = self.DecoderWithAttention.embed(torch.tensor([[vocabulary.stoi["<SOS>"]]]))

                # Compute the attention-weighted encoding using the attention mechanism
                attention_weighted_encoding, attn_weights = self.DecoderWithAttention.attention(x, h[-1])

                attn_weights_list.append(attn_weights.squeeze().numpy())

                # Pass the embeddings and attention-weighted encoding through the LSTM layer
                h, c = self.DecoderWithAttention.lstm(torch.cat([embeddings, attention_weighted_encoding.unsqueeze(1)], dim=2), (h, c))

                # Generate the output logits using the linear layer
                preds = self.DecoderWithAttention.linear(h.squeeze(1))

                # Get the predicted word by selecting the word with the highest logit value
                _, predicted = preds.max(1)
                result_caption.append(predicted.item())

                # Check if the predicted word is the end-of-sequence token ("<EOS>")
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        attn_weights_list = np.array(attn_weights_list)

        # Convert the list of word indices to a list of actual words using the vocabulary
        return [vocabulary.itos[idx] for idx in result_caption[1:]]
'''

