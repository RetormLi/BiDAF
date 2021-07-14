import argparse
import copy
import json
import os
import torch
from torch import nn, optim
from model import BiDAF
from dataset import SQuAD
import evaluate
from Vocab import Vocab


def train(args, dataset, token_vocab, char_vocab):

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, token_vocab, char_vocab).to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1
    for epoch in range(args.epoch):
        for i, batch in enumerate(dataset.get_mini_batch(args.batch_size, train=True)):
            print(batch)
            
            p1, p2 = model(batch)
            optimizer.zero_grad()
            batch_loss = criterion(p1, batch.y1) + criterion(p2, batch.y2)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            if (i + 1) % args.print_freq == 0:
                dev_loss, dev_exact, dev_f1 = test(model, args, dataset.args.get_mini_batch(
                    args.dev_batch_size, dev=True), token_vocab)
                print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f}'
                      f' / dev EM: {dev_exact:.3f} / dev F1: {dev_f1:.3f}')
                if dev_f1 > max_dev_f1:
                    max_dev_f1 = dev_f1
                    max_dev_exact = dev_exact
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                loss = 0
        model.train()
        print(
            f'max dev EM: {max_dev_exact:.3f} / max dev F1: {max_dev_f1:.3f}')


def test(model, args, data, token_vocab):
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()
    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            p1, p2 = model(batch)
            batch_loss = criterion(p1, batch.y1s) + criterion(p2, batch.y2s)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = batch.id[i]
                answer = batch.c_word[0][i][s_idx[i]:e_idx[i] + 1]
                answer = ' '.join(
                    [token_vocab.recover_from_ids([idx]) for idx in answer])
                answers[id] = answer
    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    results = evaluate.main(args)
    return loss, results['exact_match'], results['f1']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char_dim', default=8, type=int)
    parser.add_argument('--char_channel-width', default=5, type=int)
    parser.add_argument('--char_channel-size', default=100, type=int)
    parser.add_argument('--dev_batch_size', default=100, type=int)
    parser.add_argument('--train_file', default='train-v1.1.json')
    parser.add_argument('--dev_file', default='dev-v1.1.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.5, type=float)
    parser.add_argument('--print_freq', default=250, type=int)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--para_limit', default=400, type=int)
    parser.add_argument('--char_limit', default=16, type=int)
    parser.add_argument('--question_limit', default=50, type=int)
    parser.add_argument('--pretrained_file', default=None)
    args = parser.parse_args()
    print('create token_vocab')
    vocab_token = Vocab(args.train_file)

    print('create character_vocab')
    char_token = Vocab(args.train_file)

    print('loading SQuAD data...')
    data = SQuAD(args)

    setattr(args, 'char_vocab_size', vocab_token.size())
    setattr(args, 'word_vocab_size', char_token.size())
    setattr(args, 'train_file', f'.data/squad/{args.train_file}')
    setattr(args, 'dev_file', f'.data/squad/{args.dev_file}')
    setattr(args, 'prediction_file', f'prediction.out')
    print('data loading complete!')

    print('training start!')
    train(args, data)
    print('training finished!')


if __name__ == '__main__':
    main()
