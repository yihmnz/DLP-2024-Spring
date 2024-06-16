import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torchinfo import summary

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.d_k = self.d_v = dim // num_heads

        # These linear layers are used for projecting the input tensor to q, k, and v tensors
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_tokens, dim = x.shape

        # Applying the linear layer and splitting query, key, and value for each head
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.d_k)
        # reorder to (3, batch_size, num_heads, num_tokens, d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Compute the scaled dot-product attention (MatMul of queries with k, scaled by sqrt(d_k))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)  # apply softmax to scores
        attn = self.attn_drop(attn)  # apply dropout to the attention weights

        # Multiply the attention weights with the values
        weighted_avg = torch.matmul(attn, value)
        weighted_avg = weighted_avg.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        
        # Project the attention outputs back to the original tensor size
        output = self.out_proj(weighted_avg)
        return output

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    
encoder_model = Encoder(dim=768, hidden_dim=1536)  # Adjust hidden_dim accordingly
summary(encoder_model, input_size=(16, 256, 768))  # Correct input dimensions for testing