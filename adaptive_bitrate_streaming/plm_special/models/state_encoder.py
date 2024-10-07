"""
Customized state encoder based on Pensieve's encoder.
"""
import torch.nn as nn


class EncoderNetwork(nn.Module):
    """
    The encoder network for encoding each piece of information of the state.
    This design of the network is from Pensieve/Genet.
    """
    def __init__(self, conv_size=4, bitrate_levels=6, embed_dim=128):
        super().__init__()
        self.past_k = conv_size
        self.bitrate_levels = 4
        self.embed_dim = embed_dim
        self.fc1 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # last bitrate
        self.fc2 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # current buffer size
        self.fc3 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks   
        self.fc4 = nn.Sequential(nn.Linear(1, embed_dim), nn.LeakyReLU())  # remain chunks        


    def forward(self, state):
        # state.shape: (batch_size, seq_len, 6, 6) -> (batch_size x seq_len, 6, 6)
        batch_size, seq_len = state.shape[0], state.shape[1]
        state = state.reshape(batch_size * seq_len, -1, 1)
        
        feature_variabl1 = state[..., 0:1, :]
        feature_variabl2 = state[..., 1:2, :]
        feature_variabl3 = state[..., 2:3, :]
        feature_variabl4 = state[..., 3:4, :]
        
        features1 = self.fc1(feature_variabl1).reshape(batch_size, seq_len, -1)
        features2 = self.fc2(feature_variabl2).reshape(batch_size, seq_len, -1)
        features3 = self.fc3(feature_variabl3).reshape(batch_size, seq_len, -1)
        features4 = self.fc4(feature_variabl4).reshape(batch_size, seq_len, -1)
   
        return features1, features2, features3, features4