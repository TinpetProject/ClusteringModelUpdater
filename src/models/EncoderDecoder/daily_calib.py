import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.input_size = 6
        self.output_size = 6
        self.hidden_size = 128
        self.output_timestep = 7
        self.num_layers = 1
        self.dropout = 0    

        # self.bn = nn.BatchNorm1d(input_timestep)
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, input, mean, std):
        # print(f'Start: {input.shape}')
        input = (input - mean) / (std + 1e-7)
        # input = self.bn(input)
        context_vec, state = self.encoder(input)
        context_vec = self.relu(context_vec)
        # print(context_vec.shape)
        context_vec = context_vec[:, -1:]
        # print(context_vec.shape)

        outputs = []
        for i in range(self.output_timestep):
            output_i, state = self.decoder(context_vec, state)
            outputs.append(output_i)
            context_vec = output_i
        outputs = torch.cat(outputs, dim=1)
        # print(outputs.shape)
        outputs = self.dense(outputs)
        # print(outputs.shape)
        
        outputs = self.relu(outputs)

        outputs = outputs * (std + 1e-7) + mean
        return outputs

