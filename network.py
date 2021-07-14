import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayCell(nn.Module):
    '''
    Define a highway network cell. 
    The cell contains a normal layer and a gate layer.
    Through argument can set a gate bias, whose default value is 0.    
    '''

    def __init__(self, input_size, output_size, gate_bias=0.0,
                 activation_function=F.relu, gate_activation=F.sigmoid):
        super().__init__()
        self.activation_fuction = activation_fuction
        self.gate_activation = gate_activation
        self.normal_layer = nn.Linear(input_size, output_size)
        self.gate_layer = nn.Linear(input_size, output_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        normal_layer_result = self.activation_fuction(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))
        gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        gate_and_input = torch.mul((1 - gate_layer_result), x)
        return_mat = torch.add(gate_and_normal, gate_and_input)
        return return_mat


class Highway_layer(nn.Module):
    '''
    Generate a multi layer highway network.
    Connect some highway network cells.
    '''

    def __init__(self, depth, input_size, output_size):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(HighwayCell(input_size, output_size))
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net(x)


class Bi_attention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.input_size = input_size
        self.dropout = dropout_rate
        self.att = nn.Linear(hidden_size * 6, 1)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, context, question, mask):
        batch, context_len, question_len = context.size(
            0), context.size(1), question.size(1)
        context = F.dropout(context, self.dropout)
        question = F.dropout(question, self.dropout)
        # attention = torch.bmm(context, question.permute(
        #     0, 2, 1).contiguous()) / (self.input_size ** 0.5)
        reshape_context = context.unsqueeze(1).repeat(1, 1, question_len, 1)
        reshape_question = question.unsqueeze(0).repeat(1, context_len, 1, 1)
        c_q = reshape_context * reshape_question
        cat_mat = torch.cat([context, question, c_q], dim=-1)
        attention = self.att(cat_mat).squeeze(dim=-1)
        # mask, 并趋近于零
        masked_attention = attention - 1e30 * (1 - mask[:, None])
        context = self.input_linear(context)
        question = self.memory_linear(question)

        h_aware_weight = F.softmax(masked_attention, dim=-1)
        U = torch.bmm(h_aware_weight, question)
        u_aware_weight = F.softmax(masked_attention.max(
            dim=-1)[0], dim=-1).view(batch, 1, context_len)
        H = torch.bmm(u_aware_weight, context)
        G = torch.cat([context, U, torch.mul(
            context, U), torch.mul(U, H)], dim=-1)
        return G
