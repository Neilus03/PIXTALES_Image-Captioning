
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Encoder class definition using a pre-trained VGG16 model
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()  # Calling the parent class constructor
        
        vgg16 = models.vgg16(pretrained=True)  # Loading a pre-trained VGG16 model
        
        # Removing the last max pooling layer from VGG16
        features = list(vgg16.features.children())[:-1]  # Get all layers except the last one
        self.features = nn.Sequential(*features)  # Store these layers in a sequential model for our encoder

    def forward(self, images):
        features = self.features(images)  # Pass images through our VGG16 

        features = features.permute(0, 2, 3, 1)  # Permute the tensor dimensions to [batch_size, height, width, channels]

        # Flatten the tensor along height and width dimensions to be used in a fully connected
        features = features.view(features.size(0), -1, features.size(-1))
        
        return features 
   
# Disclaimer: We decided to call pixels to the patches that are outputted by the cnn, Lets say if the outputed feature map is 14x14 we'll say that we now have 196 pixels, 
# we know that those are not ppixels per se but is easier to undestand foor us and hopefully for you too.
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, feature_map_depth):#hidden size is the size of the previous RNN outputted hidden state #and feature map depth is 512 which is = to the channels of the outputted feature map of shape [32,196,512]
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size
        self.feature_map_depth = feature_map_depth
        self.attention_layer = nn.Linear(hidden_size + feature_map_depth, 1)

    def forward(self, encoder_out, h_prev): #shape of encoder_out = ([32,196,512][batch_size, num_pixels, feature_map_depth]) and shape of previous hidden state = ([32, 512][batch_size, hidden_size]) 
        
        #we want to have a prev_h that matches the size of encoder_out to be able to concat them
        h_prev_repeated = h_prev.unsqueeze(1).repeat(1, encoder_out.shape[1], 1) #([32, 512][batch_size, hidden_size]) ~> ([32,1,512][batch_size,1,hidden_size]) ~> ([32,196,512][batch_size, num_pixels, hidden_size]) 

        #we concattenate both tensors to have an nput for the fully connected layer which will then output the attention weights
        att_input = torch.cat((encoder_out, h_prev_repeated), dim=2) # ([32, 196, 1024][batch_size, num_pixels, hidden_size+feature_map_depth])                                                             
        
        #we get the attention scores for the "pixel" we're looking at each step
        att_scores = self.attention_layer(att_input) # [32, 196, 1][batch_size, num_pixels, 1]

        #Get rid of the last dimension
        att_scores = att_scores.squeeze(2) # [32, 196][batch_size, num_pixels]

        #Make all the scores sum up to 1 via softmax, hence we'll get the weights
        att_weights = F.softmax(att_scores, dim=1) #[32, 196][batch_size, num_pixels]

        return att_weights #[32, 196][batch_size, num_pixels]

   
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, feature_map_depth, vocab_size, num_layers, dropout_p=0.3):
        super(DecoderRNN, self).__init__()

        # The feature map depth after applying the CNN 
        self.feature_map_depth = feature_map_depth 

        # Word embedding layer, converts words into vectors of fixed size (embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size) 

        # LSTM layer: takes both word embeddings and the context vector as input, hence input size is embed_size + feature_map_depth
        self.lstm = nn.LSTM(embed_size + feature_map_depth, feature_map_depth, num_layers, batch_first=True, dropout=dropout_p if num_layers > 1 else 0) # Apply dropout between LSTM layers if num_layers > 1

        # This layer is used to map the output of the LSTM into the size of the vocabulary
        # This will give us the scores for each word in the vocabulary being the next word in the caption,
        # that later should be passed through a softmax to turn the scores into probabilities
        self.linear = nn.Linear(feature_map_depth, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)

        # Attention layer is used to compute the attention weights and the context vector from encoder features and the previous hidden state of the LSTM
        # Here, we are assuming that the size of the encoder features is equal to the hidden state size of the LSTM for simplicity (both 512)
        self.attention = AdditiveAttention(feature_map_depth, feature_map_depth) # Using hidden_size as feature_map_depth for simplicity as both have the same value (512)

    def forward(self, features, captions):
        # Compute the word embeddings of the captions
        embeddings = self.dropout(self.embed(captions))  # Apply dropout to embeddings

        # Compute the attention weights
        att_weights = self.attention(features, features.mean(dim=1))  # We use the mean features as the initial hidden state

        # Compute the context vector
        context_vector = torch.sum(features * att_weights.unsqueeze(2), dim=1)  # (batch_size, hidden_size)

        # Concatenate the context vector with the word embeddings
        context_vector_repeated = context_vector.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        
        lstm_input = torch.cat((context_vector_repeated, embeddings), dim=2)
        
        # Pass the concatenated input through the LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # Pass the LSTM output through the linear layer to get the output scores for each word in the vocabulary
        outputs = self.linear(self.dropout(lstm_out))  # Apply dropout to LSTM outputs

        return outputs, att_weights
    
   
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()

        # Instantiate the encoder (CNN), it is going to encode images into feature representations
        self.encoder = EncoderCNN()

        # Instantiate the decoder (RNN), it will generate captions based on encoded image features
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Encode the images to get the feature maps, these are the "image representations" that the decoder will use
        features = self.encoder(images) # [batch_size, num_pixels, encoder_feature_size]
        
        # Pass features and captions through the decoder, it will try to generate captions based on image features
        outputs, att_weights = self.decoder(features, captions)

        return outputs, att_weights #I want to store attention weights so that I can plot attention maps later on
