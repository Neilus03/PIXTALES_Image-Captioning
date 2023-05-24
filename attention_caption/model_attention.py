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
        features = self.features(images)

        # Debug print
        #print("Shape of features after EncoderCNN: ", features.shape) # [32, 512, 14, 14]

        # Permute the tensor dimensions
        features = features.permute(0, 2, 3, 1)
        #print("Shape of features after permuting: ", features.shape) # [32, 14, 14, 512]

        # Flatten the tensor
        features = features.view(features.size(0), -1, features.size(-1))
        #print('features after applying view to flatten:', features.shape) # [32, 196, 512]
        return features

# Dot-Product Attention
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        
        # Debug prints   
        print("Shape of encoder_outputs: ", encoder_outputs.shape) # [32, 196, 512]
        print("Shape of decoder_hidden before unsqueeze: ", decoder_hidden.shape) # [32, 256]

        # Perform a batch matrix multiplication with encoder_outputs and decoder_hidden
        # Shape of attention_scores: [batch_size, num_pixels, 1]
        
        print('Shape of decoder_hidden after unsqueeze(2).shape:', decoder_hidden.unsqueeze(2).shape) # [32, 256, 1]

        attention_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2) # Expected output shape [196, 512]

        # Debug print
        print("Shape of attention_scores: ", attention_scores.shape)

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
        self.num_layers = num_layers

        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size) # from 2994 to 256

        # Attention layer
        self.attention = DotProductAttention()

        # LSTM layer - Attention context vector and word embedding are inputs, so size is 2*hidden_size
        self.lstm = nn.LSTM(2*hidden_size, hidden_size, num_layers, batch_first=True)

        # Linear layer mapping from hidden dimension to vocab size
        self.linear = nn.Linear(hidden_size, vocab_size) #from 256 to 2994

    def forward(self, features, captions):
        # Get the word embeddings of the captions
        embeddings = self.embed(captions) 
        print('Features shape in decoder fordward: ', features.shape) # [32. 192, 512]
        # Initialize LSTM hidden state
        print("Embeddings shape at every pos:", embeddings.shape) # [32, 25, 256]
 
        h = torch.zeros(self.num_layers, embeddings.shape[0], features.shape[2]).to(embeddings.device) # shape of h [1, 32, 512]
        c = torch.zeros(self.num_layers, embeddings.shape[0], features.shape[2]).to(embeddings.device) # shape of c [1, 32, 512]
        
        # Store the outputs here
        outputs = torch.empty((embeddings.shape[0], captions.shape[1], self.vocab_size)).to(embeddings.device)
        #print("outputs shape:", outputs.shape) 
        # Iterating through each timestep in the captions' length
        #print("captions shape", captions.shape)
        for t in range(captions.shape[1]):
            # Compute the attention weights and apply to encoder features
            context_vector, _ = self.attention(features, h[-1])

            #print("context_vector shape: ",context_vector.shape)
            #print("context_vector.unsqueeze(1) shape: ",context_vector.unsqueeze(1).shape)
            
            #print("embeddings[:, t] shape:", embeddings[:, t].shape)
            #print("embeddings[:, t].unsqueeze(1) shape:", embeddings[:, t].unsqueeze(1).shape)
            # Concatenate the context vector with the current word embedding
            input_lstm = torch.cat((context_vector.unsqueeze(1), embeddings[:, t].unsqueeze(1)), dim=2)
            # Pass embeddings to LSTM
            print("shape of h:",h.shape)  # [32, 32, 512]
            print("shape of c:",c.shape)  # [32, 32, 512]
            print("input_lstm shape:", input_lstm.shape) # [32, 1, 768]

            h, c = self.lstm(input_lstm.squeeze(1), (h.squeeze(1), c.squeeze(1))) # [32, 768], [32, 512], [32, 512]
            
            #print('h shape:', h.shape)
            #print('h with squeeze(1)', h.squeeze(1).shape)

            # Pass LSTM output through linear layer to get scores for each word in the vocabulary
            output = self.linear(h.squeeze(1)) 

            # Store the output
            outputs[:, t, :] = output

        return outputs

# Full model combining the encoder and the decoder
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        # Encoder: Convolutional Neural Network (CNN) using a pre-trained VGG16 model
        self.EncoderCNN = EncoderCNN()

        # Decoder: Recurrent Neural Network (RNN) with a dot-product attention mechanism
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Extract high-level visual features from input images using the encoder
        features = self.EncoderCNN(images) 

        # Generate a sequence of words (caption) from the image features using the decoder
        outputs = self.DecoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            # Extract image features using the encoder
            x = self.EncoderCNN(image).unsqueeze(0)

            # LSTM states initialization
            states = None

            for _ in range(max_length):
                # LSTM forward step
                h, _ = self.DecoderRNN.lstm(x, states)
                # Map LSTM hidden state output to the vocabulary size to get word scores
                output = self.DecoderRNN.linear(h.squeeze(0))
                # Pick the word with the highest score as the next word of the generated caption
                _, predicted = output.max(1)
                # Store the generated word to the result caption
                result_caption.append(predicted.item())

                # If the generated word is <EOS>, stop generation
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                # Embed the generated word to serve as the input of the next LSTM step
                x = self.DecoderRNN.embed(predicted).unsqueeze(0) 

        # Convert the list of generated word indices to a list of words
        return [vocabulary.itos[idx] for idx in result_caption]
