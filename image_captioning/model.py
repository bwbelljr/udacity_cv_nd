import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer that transforms words into
        # vectors of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM that takes embeddings and transforms
        # them into hidden states of size hidden_size
        # and with num_layers
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Define final, fully connected layer that maps
        # hidden state output to the output, vocab_size
        self.fc = nn.Linear(hidden_size, vocab_size)        
        
    def forward(self, features, captions):
        '''Define feed-forward behavior'''
        
        # pass captions through embedding layer
        # With help from knowledge on reshaping captions
        # https://knowledge.udacity.com/questions/72357
        embeddings = self.word_embeddings(captions[:,:-1])
        
        # Concatenate image feature embeddings and caption embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # Pass concatenated embeddings through LSTM layer
        out, hidden = self.lstm(embeddings)        
        
        outputs = self.fc(out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        '''accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)'''
        # with help from: https://knowledge.udacity.com/questions/97862
        
        # initialize output list of indices
        output_list = []     
        
        # iterate over max length of words
        for _ in range(max_len):
            # Pass inputs through LSTM. First time, inputs
            # will be image embeddings. In subsequent times,
            # inputs will be word embeddings of previously
            # predicted word
            out, states = self.lstm(inputs, states)
            out = out.squeeze(1)

            # pass output through fully-connected layer that maps
            # from hidden state size to vocab size
            out = self.fc(out)

            # pass output values through softmax
            # out = nn.functional.softmax(out, dim=1)
            
            # Get the most likely word
            _, word = torch.max(out, dim=1)
            
            # add word to list
            output_list.append(word.item())
            
            # if word=1, then it is <end>
            if word == 1:
                break
                
            # Make the next output the word we just predicted
            inputs = self.word_embeddings(word).unsqueeze(1)
            
        return output_list