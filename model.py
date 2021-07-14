import torch
import torch.nn as nn
import network


class BiDAF(nn.Module):
    def __init__(self, args, token_vocab, character_vocab):
        super(BiDAF, self).__init__()
        self.args = args
        # 1.character embedding layer
        self.char_emb = nn.Embedding(
            args.char_dim, args.char_vocab_size, padding_idx=1)
        self.char_conv = nn.Conv2d(
            1, args.char_channel_size, (args.char_dim, args.char_channel_width), stride=1)

        # 2.Word Embedding layer
        self.word_emb = token_vocab.load_pretrained_embeddings(
            args.pretrainded_path)

        # highway network
        assert self.args.hidden_size * \
            2 == (self.args.char_channel_size + self.args.word_dim)
        self.highway_network = network.Highway_layer(
            2, args.hidden_size * 2)

        # 3.Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, num_layers=1,
                                    bidirectional=True, batch_first=True, dropout=args.dropout)

        # 4.Attention flow layer
        self.att_weight_c = nn.Linear(args.hidden_size*2, 1)
        self.att_weigth_q = nn.Linear(args.hidden_size*2, 1)
        self.att_weigth_cq = nn.Linear(args.hidden_size * 2, 1)
        self.attention = network.Bi_attention(
            args.hidden_size * 2, args.dropout)

        # 5. Modeling layer
        self.modeling_LSTM = nn.LSTM(input_size=args.hidden_size * 8,
                                     hidden_size=args.hidden_size,
                                     num_layers=2,
                                     bidirectional=True,
                                     batch_first=True,
                                     dropout=args.dropout)
        # 6. Output Layer
        self.output_LSTM = nn.LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):
        def char_emb_layer(batch_data, dropout):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch, seq_len, word_len = [batch_data.size(i) for i in [0, 1, 2]]

            word_data = dropout(word_data)
            word_data = batch_data.view(-1, word_len)
            x = self.char_emb(word_data)
            reshaped_x = x.reshape(-1, seq_len, word_len)
            conved_x = self.char_conv(reshaped_x)
            pooled_x = torch.max(conved_x, -1)[0]
            return_x = conved_x.view(
                batch, seq_len, self.args.char_channel_size)
            return return_x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size)
            """
            combine_mat = torch.cat((x1, x2), 2)
            return_mat = self.highway_network(combine_mat)
            return return_mat

        def contextual_emb_layer(x):
            """
            :param x: (batch, seq_len, hidden_size * 2)
            :return: (batch, seq_len, hidden_size * 2)
            """
            x = x.transpose(0, 1)
            h0 = torch.zeros(2, self.args.batch_size, self.args.hidden_size)
            c0 = torch.zeros(2, self.args.batch_size, self.args.hidden_size)
            x = self.context_LSTM(x, (h0, c0))[0]
            x = x.transpose(0, 1)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, hidden_size * 8)
            """
            mask = None
            G = self.attention(c, q, self.args.hidden_size, mask)
            return G

        def modeling_layer(g):
            """
            :param x: (batch, seq_len, hidden_size * 8)
            :return: (batch, seq_len, hidden_size * 2)
            """
            g = g.transpose(0, 1)
            h0 = torch.zeros(4, self.args.batch_size, self.args.hidden_size)
            c0 = torch.zeros(4, self.args.batch_size, self.args.hidden_size)
            m = self.modeling_LSTM(g, (h0, c0))[0]
            m = m.transpose(0, 1)
            return m

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # p1
            temp_mat1 = torch.cat((g, m), 2)
            wp1 = nn.Parameter(torch.FloatTensor(
                1, self.args.hidden_size * 10))
            p1_distrib = nn.functional.softmax(
                torch.matmul(wp1, temp_mat1)).squeeze()
            m2 = m.transpose(0, 1)
            h0 = torch.zeros(2, self.args.batch_size, self.args.hidden_size)
            # h0 = torch.Tensor(nn.functional.softmax(
            #     torch.matmul(wp1, temp_mat1)).squeeze())
            c0 = torch.zeros(2, self.args.batch_size, self.args.hidden_size)
            m2 = self.output_LSTM(m2, (h0, c0))
            m2 = m2.transpose(0, 1)
            wp2 = nn.Parameter(torch.FloatTensor(
                1, self.args.hidden_size * 10))
            temp_mat2 = torch.cat((g, m2), 2)
            p2_distrib = nn.functional.softmax(
                torch.matmul(wp2, temp_mat2)).squeeze()
            return p1_distrib, p2_distrib

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch['context_char_idxs'], self.dropout)
        q_char = char_emb_layer(batch['ques_char_idxs'], self.dropout)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch['context_idxs'])
        q_word = self.word_emb(batch['ques_idxs'])
        c_lens = batch['context_lens']
        q_lens = batch['ques_lens']

        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        # 3. Contextual Embedding Layer
        h = contextual_emb_layer(c)
        u = contextual_emb_layer(q)

        # 4. Attention Flow Layer
        g = att_flow_layer(h, u)

        # 5. Modeling Layer
        m = modeling_layer(g)

        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)
        # (batch, c_len), (batch, c_len)
        return p1, p2
