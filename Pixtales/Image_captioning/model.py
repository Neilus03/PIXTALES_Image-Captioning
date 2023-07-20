import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=True, p_dropout=0.5):
        super(EncoderCNN, self).__init__()

        self.train_CNN = train_CNN
        
        # Load the pretrained ResNet-50 model
        self.resnet = models.resnet50(pretrained=True)

        # Replace the fully connected layer with a new linear layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)

        # Activation function and dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.batch_norm = nn.BatchNorm1d(embed_size)  # Batch Normalization layer
  
    def forward(self, images):
        # Extract features from the images using the ResNet-50 model
        features = self.resnet(images)

        # Set the requires_grad attribute of parameters based on the train_CNN flags
        for name, parameter in self.resnet.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = self.train_CNN
        # Apply ReLU activation function and dropout to the features and Batch Norm
        features = self.batch_norm(features)
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
        
        if embeddings.dim() == 3:
            features = features.unsqueeze(0)
        elif embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
            features = features.unsqueeze(0)
            
        # Concatenate the features and embedded captions
        embeddings = torch.cat((features, embeddings), dim=0)

        # Pass the embeddings through the LSTM layer
        hiddens, _ = self.lstm(embeddings)

        # Convert the hidden states to output logits
        outputs = self.linear(hiddens)

        return outputs
    



class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        
        # Instantiate the EncoderCNN module
        self.EncoderCNN = EncoderCNN(embed_size)

        # Instantiate the DecoderRNN module
        self.DecoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        # Pass the images through the EncoderCNN module to extract features
        features = self.EncoderCNN(images)

        # Pass the features and captions through the DecoderRNN module
        outputs = self.DecoderRNN(features, captions)

        # Return the final output logits
        return outputs
    
    def caption_image(self, image, vocabulary, max_length = 42):
        # Initialize an empty list to store the generated caption
        result_caption = []
        
        # Disable gradient calculation
        with torch.no_grad():
        
            # Pass the image through the EncoderCNN model to extract features
            x = self.EncoderCNN(image)
            
            # Initialize the LSTM states to None
            states = None
             
            # Generate the caption word by word 
            for _ in range(max_length):
                
                # Pass the image features and LSTM states through the LSTM layer
                hiddens, states = self.DecoderRNN.lstm(x, states)
                
                # Pass the LSTM output through the linear layer to get the output logits
                output = self.DecoderRNN.linear(hiddens.unsqueeze(0))
                #print is to debug
                #print(output.shape)
                
                # Get the predicted word by taking the index of the highest logit value

                predicted = output.argmax(2).squeeze()
                #is argmax(2) bc due to previous unsqueeze the shape is [1,1,2994] instead of [1,2994]
                #Also is worth to note that the squeeze is used to lower one dimension for inputting it to the lstm which takes at most 3dimensions
                
                '''
                ANOTHER OPTION WOULD BE TO GET THE PREDICTED WORD SAMPLING INSTEAD OF THE PREVIOUS GREEDY APPROACH
                (neil: a mi me gusta más, decid vosotros,
                 dani: Tb me gusta mas)
                
                # Apply softmax to convert output logits to probabilities
        		probs = output.squeeze(0).softmax(dim=0)

        		# Sample a word index from the output probabilities
        		predicted = Categorical(probs).sample().item()
                '''
                
                # Append the predicted word index to the result caption list
                result_caption.append(predicted.item())
                
                # Convert the predicted word back to a tensor and add a batch dimension
                x = self.DecoderRNN.embed(torch.tensor(predicted.item(), device=x.device)).unsqueeze(0)
                
                # Check if the predicted word is the end-of-sequence token ("<EOS>")
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        
        # Convert the list of word indices to a list of actual words using the vocabulary
        return [vocabulary.itos[idx] for idx in result_caption]


