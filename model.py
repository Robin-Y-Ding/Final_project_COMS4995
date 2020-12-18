import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32,64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #self.fc1 = nn.Linear(4608,128)
        self.fc1 = nn.Linear(9216, 128)
        #self.fc1 = nn.Linear(18432 ,128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, get_logits=False, get_feature = False, which_feature = 0):

        
        x = self.conv1(x)
        x = F.relu(x)


        #noise_l1 = torch.Tensor(x.size()).normal_(0,3.1).cuda()
        #x = x + noise_l1
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
