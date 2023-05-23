

import torch
import torch.nn as nn
import torchvision.models as models


# Encoder part of the model using a pre-trained VGG16
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        
        # Remove the last max pooling layer from VGG16
        features = list(vgg16.features.children())[:-1]
        self.features = nn.Sequential(*features)

    def forward(self, images):
        
        # Pass images through convolutional layers
        features = self.features(images)  # Shape: [batch_size, 512, 14, 14]
        
        print("Shape of features after EncoderCNN: ", features.shape)  # Debug print

        # Permute the tensor dimensions
        features = features.permute(0, 2, 3, 1)  # Shape: [batch_size, 14, 14, 512]
        
        # Flatten the tensor
        features = features.view(features.size(0), -1, features.size(-1))  # Shape: [batch_size, 196, 512]
        
        return features

# Dot-Product Attention
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        
        print("Shape of encoder_outputs: ", encoder_outputs.shape)  # Debug print
        print("Shape of decoder_hidden before unsqueeze: ", decoder_hidden.shape)  # Debug print


        # Perform a batch matrix multiplication with encoder_outputs and decoder_hidden
        # Shape of attention_scores: [batch_size, num_pixels, 1]
        attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        
        print("Shape of attention_scores: ", attention_scores.shape)  # Debug print

        # Apply softmax to compute attention weights
        # Shape of attention_weights: [batch_size, num_pixels]
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Compute context vector as the weighted sum of the encoder_outputs
        # Shape of context_vector: [batch_size, encoder_dim]
        context_vector = torch.sum(encoder_outputs * attention_weights.unsqueeze(2), dim=1)
        
        return context_vector, attention_weights


# Decoder part of the model
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        # Define the dimensions
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Attention layer
        self.attention = DotProductAttention()

        # LSTM layer - Attention context vector and word embedding are inputs, so size is 2*embed_size
        self.lstm = nn.LSTM(2*embed_size, hidden_size, num_layers, batch_first=True)

        # Linear layer mapping from hidden dimension to vocab size
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Get the word embeddings of the captions
        embeddings = self.embed(captions)  # Shape: [batch_size, caption_length, embed_size]
        
        # Initialize LSTM hidden state
        h, c = torch.zeros(embeddings.shape[0], 1, self.hidden_size).to(embeddings.device), \
               torch.zeros(embeddings.shape[0], 1, self.hidden_size).to(embeddings.device)
        
        # Store the outputs here
        outputs = torch.empty((embeddings.shape[0], captions.shape[1], self.vocab_size)).to(embeddings.device)

        # Iterating through each timestep in the captions' length
        for t in range(captions.shape[1]):
            # Compute the attention weights and apply to encoder features
            context_vector, _ = self.attention(features, h[-1])
            # Concatenate the context vector with the current word embedding
            # Shape: [batch_size, 1, 2*embed_size]
            input_lstm = torch.cat((context_vector.unsqueeze(1), embeddings[:, t].unsqueeze(1)), dim=2)
            # Pass embeddings to LSTM
            h, c = self.lstm(input_lstm, (h, c))  # Shape: [batch_size, 1, hidden_size]
            # Pass LSTM output through linear layer to get scores for each word in the vocabulary
            output = self.linear(h.squeeze(1))  # Shape: [batch_size, vocab_size]
            # Store the output
            outputs[:, t, :] = output

        return outputs


# Full model combining the encoder and the decoder
# Full model combining the Encoder (CNN) and the Decoder (RNN with attention)
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        # Encoder: Convolutional Neural Network (CNN) using a pre-trained VGG16 model
        self.EncoderCNN = EncoderCNN()

        # Decoder: Recurrent Neural Network (RNN) with a dot-product attention mechanism
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Extract high-level visual features from input images using the encoder
        features = self.EncoderCNN(images)  # Shape: [batch_size, 196, 512]

        # Generate a sequence of words (caption) from the image features using the decoder
        outputs = self.DecoderRNN(features, captions)  # Shape: [batch_size, caption_length, vocab_size]
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        # Switch off gradients computation because we are in inference mode
        with torch.no_grad():
            # Extract image features using the encoder
            x = self.EncoderCNN(image).unsqueeze(0)  # Shape: [1, 196, 512]

            # LSTM states initialization
            states = None

            # Generate a caption for the image up to the max_length or until <EOS> token is found
            for _ in range(max_length):
                # LSTM forward step
                h, _ = self.DecoderRNN.lstm(x, states)
                # Map LSTM hidden state output to the vocabulary size to get word scores
                output = self.DecoderRNN.linear(h.squeeze(0))  # Shape: [1, vocab_size]
                # Pick the word with the highest score as the next word of the generated caption
                _, predicted = output.max(1)
                # Store the generated word to the result caption
                result_caption.append(predicted.item())

                # If the generated word is <EOS>, stop generation
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                # Embed the generated word to serve as the input of the next LSTM step
                x = self.DecoderRNN.embed(predicted).unsqueeze(0)  # Shape: [1, 1, embed_size]

        # Convert the list of generated word indices to a list of words
        return [vocabulary.itos[idx] for idx in result_caption]


