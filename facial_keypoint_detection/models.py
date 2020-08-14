## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        """
        Neural network defined based on NaimeshNet

        Network takes in square (same width/height), grayscale image
        Network ends with a linear layer representing the keypoints with 136,
        (2 for each 68 (x, y) pairs)

        """

        # Inherit attributes, methods from nn.Module class
        super(Net, self).__init__()
         
        # Convolutional layers

        # 1 input image channel (grayscale), 32 output channels
        # 4x4 square convolutional kernel
        # output size = (W-F)/S + 1 = (224-4)/1 + 1 = 221
        # output Tensor for one image will have dimensions (32, 221, 221)
        # after one pool layer, (32, 110, 110), rounded down
        self.conv1 = nn.Conv2d(1, 32, 4)

        # second conv layer: 32 inputs, 64 outputs
        # 3x3 square convolutional kernel
        # output size: (W-F)/S + 1 = (110-3)/1 + 1 = 108
        # output Tensor: (64, 108, 108)
        # output Tensor after maxpooling: (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # third conv layer: 64 inputs, 128 outputs
        # 2x2 square convolutional kernel
        # output size: (W-F)/S + 1 = (54-2)/1 + 1 = 53
        # output Tensor: (128, 53, 53)
        # output Tensor after maxpooling: (128, 26, 26), rounded down 
        self.conv3 = nn.Conv2d(64, 128, 2)

        # fourth conv layer: 128 inputs, 256 outputs
        # 1x1 square convolution kernel
        # output size: (W-F)/S + 1 = (26-1)/1 + 1 = 26
        # output tensor: (256, 26, 26)
        # output Tensor after maxpooling: (256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)

        # Fully-connected (dense) linear layers
        
        # 256 outputs * 13x13 filtered/pooled map size
        self.fc1 = nn.Linear(256*13*13, 1000)
        
        self.fc2 = nn.Linear(1000, 1000)
        
        self.fc3 = nn.Linear(1000, 136)

        # Maxpooling layer with shape(2,2)
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layers
        # Note that dropout layers could be nn.Dropout2d (for convolutional 
        # part of network) nn.Dropout (for linear parts of the network)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.dropout3 = nn.Dropout2d(p=0.3)
        self.dropout4 = nn.Dropout2d(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

    def forward(self, x):
        """Defining feedforward behavior inspired by NaimeshNet"""

        # Convolutional/pooling layer 1
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)

        # Convolutional/pooling layer 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        # Convolutional/pooling layer 3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)

        # Convolutional/pooling layer 4
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)

        # Flatten x for linear, dense layers
        x = x.view(x.size(0), -1)
        
        # Fully-connected (dense) layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)

        # Fully-connected (dense) layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)

        # Final fully-connected layer to output
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x