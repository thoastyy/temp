# import torch 
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# import math
# import numpy as np


# class LipNet(torch.nn.Module):
#     def __init__(self, dropout_p=0.5):
#         super(LipNet, self).__init__()
#         self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
#         self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
#         self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
#         self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
#         self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
#         self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
#         self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
#         self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
        
#         self.FC    = nn.Linear(512, 27+1)
#         self.dropout_p  = dropout_p

#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(self.dropout_p)        
#         self.dropout3d = nn.Dropout3d(self.dropout_p)  
#         self._init()
    
#     def _init(self):
        
#         init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
#         init.constant_(self.conv1.bias, 0)
        
#         init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
#         init.constant_(self.conv2.bias, 0)
        
#         init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
#         init.constant_(self.conv3.bias, 0)        
        
#         init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
#         init.constant_(self.FC.bias, 0)
        
#         for m in (self.gru1, self.gru2):
#             stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
#             for i in range(0, 256 * 3, 256):
#                 init.uniform_(m.weight_ih_l0[i: i + 256],
#                             -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0[i: i + 256])
#                 init.constant_(m.bias_ih_l0[i: i + 256], 0)
#                 init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
#                             -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
#                 init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
        
        
#     def forward(self, x):
        
#         print(x.shape)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool1(x)
        
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)        
#         x = self.pool2(x)
        
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)        
#         x = self.pool3(x)
        
#         # (B, C, T, H, W)->(T, B, C, H, W)
#         x = x.permute(2, 0, 1, 3, 4).contiguous()
#         # (B, C, T, H, W)->(T, B, C*H*W)
#         x = x.view(x.size(0), x.size(1), -1)
        
#         self.gru1.flatten_parameters()
#         self.gru2.flatten_parameters()
        
#         x, h = self.gru1(x)        
#         x = self.dropout(x)
#         x, h = self.gru2(x)   
#         x = self.dropout(x)
                
#         x = self.FC(x)
#         x = x.permute(1, 0, 2).contiguous()
#         print(x.shape)
#         return x
        
    
import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import numpy as np
from efficientnet_pytorch import EfficientNet



class LipNet(torch.nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()

        # Load pre-trained EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1000)
        self.efficientnet._fc = nn.Identity()  # Remove the original fully connected layer

        self.gru1 = nn.GRU(1280, 256, 1, bidirectional=True)
        self.gru2 = nn.GRU(512, 256, 1, bidirectional=True)

        self.FC = nn.Linear(512, 28)
        self.dropout_p = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self._init()
    
    def _init(self):
        
        # Initialize the weights of the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
                for i in range(0, 256 * 3, 256):
                    nn.init.uniform_(m.weight_ih_l0[i: i + 256],
                                    -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    nn.init.orthogonal_(m.weight_hh_l0[i: i + 256])
                    nn.init.constant_(m.bias_ih_l0[i: i + 256], 0)
                    nn.init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
                                    -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
                    nn.init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
                    nn.init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)

        
        
    def forward(self, x):
        print(x.shape)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        print(x.shape)
        x=x[:, :3, ...]

        print(x.shape)
        # Pass through EfficientNet
        x = self.efficientnet(x)
        print(x.shape)

        # Reshape back to [batch_size, num_frames, features]
        # x = x.view(x.size(0), -1, x.size(1))
        x = x.view(x.size(0), -1)
        print(x.shape)

        # GRU Layer 1
        x, _ = self.gru1(x.unsqueeze(0))  # Add an extra dimension

        # GRU Layer 2
        x, _ = self.gru2(x)

        # Fully Connected Layer
        x = self.FC(x.squeeze(0))
        print(x.shape)

        # Reshape back to [batch_size, num_frames, 28]
        x = x.view(-1, 75, 28)
        print(x.shape)

        # Apply ReLU activation
        x = self.relu(x)

        # Apply dropout
        x = self.dropout(x)
        print(x.shape)

        
        return x
        
    
