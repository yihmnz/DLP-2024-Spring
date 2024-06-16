import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TDEE(nn.Module):
    def __init__(self, Nt, hidden_dim, lstm_layers, output_features):
        super(TDEE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, (32, Nt), padding="valid"),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, (5, Nt), padding="valid"),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, (3, Nt), padding="valid"),
            nn.BatchNorm2d(32)
        )

        self.bi_lstm = nn.Sequential(
            nn.LSTM(input_size=160, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True),
        )
        self.bi_lstm2 = nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(219, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 3, 2, 1).contiguous() # (batch_size, time_steps, seq_length, feature_dim)
        x = x.view(x.size(0), x.size(2), x.size(3) * x.size(1))  #(batch_size, seq_length, feature_dim * time_steps)
        x, _ = self.bi_lstm(x)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        #  (batch, 32, 5)
        x = x.view(-1, 32, 5)
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SDEE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SDEE, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # print(x.shape)
        x = torch.mean(x, dim=2) # merge DE by time (x.shape)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class CombinedModel(nn.Module):
    def __init__(self, Nt, hidden_dim, lstm_layers, gcn_nfeat, gcn_nhid, gcn_nclass, gcn_dropout, output_features, num_heads):
        super(CombinedModel, self).__init__()
        self.tdee = TDEE(Nt, hidden_dim, lstm_layers, output_features).to(device)
        self.gcn = SDEE(gcn_nfeat, gcn_nhid, gcn_nclass, gcn_dropout).to(device)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=output_features, num_heads=num_heads).to(device)
        self.dense_layer = nn.Linear(output_features, 1).to(device)

    def forward(self, x, adj):
        tdee_output = self.tdee(x)  # (batch_size, 32, 5)
        gcn_output = self.gcn(x, adj)  # (batch_size, 32, 5)
        gcn_output = gcn_output.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        tdee_output = tdee_output.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        
        attn_output, _ = self.multihead_attn(gcn_output, tdee_output, tdee_output)
        attn_output = torch.cat([tdee_output, gcn_output, attn_output], dim=0) #two stepp fusion
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        attn_output = attn_output.mean(dim=1)  
        final_output = self.dense_layer(attn_output)
        
        return final_output
