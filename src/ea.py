import torch
import torch.nn as nn

class EmotionAnchorLayer(nn.Module):
    def __init__(
        self, feature_size, num_sentences, anchor_size, kernel_size, stride
    ):
        super(EmotionAnchorLayer, self).__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=num_sentences,
                out_channels=3,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(3),
            nn.ReLU(),
        )
        
        self.filter_feat_size = int((feature_size - kernel_size) / stride + 1)
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=self.filter_feat_size, out_features=1),
            nn.Sigmoid(),
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=self.filter_feat_size, out_features=1),
            nn.Sigmoid(),
        )

        self.linear3 = nn.Sequential(
            nn.Linear(in_features=self.filter_feat_size, out_features=1),
            nn.Sigmoid(),
        )
        
        
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=self.filter_feat_size, out_features=anchor_size),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_pnn = self.conv1d(x)
        
        x_positive = x_pnn[:,0,:].reshape(batch_size, -1)
        x_neutral = x_pnn[:,1,:].reshape(batch_size, -1)
        x_negative = x_pnn[:,2,:].reshape(batch_size, -1)
        w_positive = self.linear1(x_positive)
        w_neutral = self.linear2(x_neutral)
        w_negative = self.linear3(x_negative)
        
        x_positive = x_positive * w_positive
        x_neutral = x_neutral * w_neutral
        x_negative = x_negative * w_negative
        x_pnn = x_positive + x_neutral + x_negative
        
        x_anchor = self.linear_out(x_pnn)
        
        return x_anchor, x_pnn, x_positive, x_neutral, x_negative


