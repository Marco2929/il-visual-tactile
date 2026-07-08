def foar_encoder(input_dim = 6, obs_feature_dim = 512, num_obs_force = 200):

    force_encoder = FoarForceEncoder(num_obs_force, input_dim, obs_feature_dim)

    batch_size = 24

    force_torque = None
      
    force_torque_feature, force_torque_pos, force_torque_padding_mask = force_encoder(force_torque, batch_size=batch_size)


import torch 
import math

class FoarForceEncoder(nn.Module):
    def __init__(self, 
                num_obs_force = 100,
                input_dim = 6, 
                output_dim = 512):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                    nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
                    nn.Linear(128, output_dim), nn.LayerNorm(output_dim), nn.ReLU())
        self.position_embedding = PositionalEncoding1D(output_dim, num_obs_force)

    def forward(self, force, num_obs_feature=100, batch_size=24):
        num_obs_feature = force.shape[1]
        force_feature = self.mlp(force)
        force_pos = self.position_embedding(force_feature)
        force_padding_mask = torch.zeros([batch_size, num_obs_feature], dtype=torch.bool, device=force.device)
        return force_feature, force_pos, force_padding_mask

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].expand(x.size(0), -1, -1)