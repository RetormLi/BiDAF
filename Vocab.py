import numpy as np
from tqdm import tqdm
import torch
import spacy
import json
from collections import Counter
import numpy as np
import dataset
# from codecs import open

nlp = spacy.blank("en")
import torch
torch

class Vocab(object):
    """
    Implements a vocabulary to store the tokens in the data, with their corresponding embeddings.
    """

    def __init__(self, filename=None, use_token=True, use_character=False, initial_tokens=None, lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}  # token计数
        self.ch2id = {}
        self.id2ch = {}
        self.lower = lower

        self.embed_dim = None
        self.embeddings = None

        self.pad_token = '<blank>'
        self.unk_token = '<unk>'

        self.initial_tokens = initial_tokens if initial_tokens is not None else []
        self.initial_tokens.extend(
            [self.pad_token, self.unk_token])  # initial_tokens是一个列表。加入了unknown和padding.
        for token in self.initial_tokens:
            # pad_token是0，unk_token是1. 0 <-> '<blank>', 1 <-> '<unk>'.
            self.add(token)

        if filename is not None:
            self._create_vocab(filename, use_token=use_token,
                               use_character=use_character)

    def size_of_token(self):
        """
        get the size of vocabulary
        Returns:
            an integer indicating the size
        """
        return len(self.id2token)

    def _create_vocab(self, file_path, use_token=True, use_character=False):
        """
        loads the vocab from file_path
        Args:
            file_path: a file with a word in each line
        """

        with open(file_path, "r") as fh:
            source = json.load(fh)
            for i, article in enumerate(source["data"]):
                for para in article["paragraphs"]:
                    context = para["context"].replace(
                        "''", '" ').replace("``", '" ')
                    context_tokens = dataset.word_tokenize(context)
                    context_chars = [list(token)
                                     for token in context_tokens]

                    if use_token:
                        for token in context_tokens:
                            self.add(token)  # 将token添加到词表
                    else:
                        for token in context_chars:
                            for char in token:
                                self.add(char, is_char=True)  # 将character添加到词表

                    for qa in para["qas"]:
                        ques = qa["question"].replace(
                            "''", '" ').replace("``", '" ')
                        ques_tokens = dataset.word_tokenize(ques)
                        ques_chars = [list(token)
                                      for token in ques_tokens]

                        if use_token:
                            for token in ques_tokens:
                                self.add(token)  # 将token添加到词表
                        else:
                            for token in ques_chars:
                                for char in token:
                                    # 将character添加到词表
                                    self.add(char, is_char=True)

    def get_id(self, token, is_char=False):
        """
        gets the id of a token, returns the id of unk token if token is not in vocab
        Args:
            key: a string indicating the word
        Returns:
            an integer or '<unk>' if no found
        """
        if not is_char:
            token = token.lower() if self.lower else token
            return self.token2id.get(token, default=self.unk_token)
        else:
            char = token.lower() if self.lower else token
            return self.ch2id.get(char, default=self.unk_token)

    def get_token(self, idx):
        """
        gets the token corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string or '<unk>' if not found
        """
        return self.id2token.get(idx, default=self.unk_token)

    def get_char(self, idx):
        """
        gets the character corresponding to idx, returns unk token if idx is not in vocab
        Args:
            idx: an integer
        returns:
            a token string or '<unk>' if not found
        """
        return self.id2ch.get(idx, default=self.unk_token)

    def add(self, token, cnt=True, is_char=False):
        """
        adds the token to vocab
        Args:
            token: a string
            cnt: to decide whether to count
            is_char: change the add mode to character embedding
        Returns:
            idx: the index of the added item
        """
        if not is_char:
            token = token.lower() if self.lower else token
            if token in self.token2id:
                idx = self.token2id[token]
            else:
                idx = self.size_of_token()  # 从0到n的顺序id
                self.id2token[idx] = token
                self.token2id[token] = idx
            if cnt:
                if token in self.token_cnt:
                    self.token_cnt[token] += 1
                else:
                    self.token_cnt[token] = 1
            return idx

        else:
            char = token.lower() if self.lower else token
            if char in self.ch2id:
                idx = self.ch2id[char]
            else:
                idx = len(self.ch2id)
                self.id2ch[idx] = char
                self.ch2id[char] = idx
            return idx

    def filter_tokens_by_cnt(self, min_cnt):
        """
        filter the tokens in vocab by their count
        Args:
            min_cnt: tokens with frequency less than min_cnt is filtered
        """
        filtered_tokens = [
            token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        for token in self.initial_tokens:
            self.add(token, cnt=False)  # 不再计数
        for token in filtered_tokens:
            self.add(token, cnt=False)  # 不再计数

    def randomly_init_embeddings(self, embed_dim):
        """
        randomly initializes the embeddings for each token
        Args:
            embed_dim: the size of the embedding for each token
        """
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros(
                [self.embed_dim])  # 未知token初始化为0向量

    def load_pretrained_embeddings(self, embedding_path, embedding_dim):
        """
        loads the pretrained embeddings from embedding_path,
        tokens not in pretrained embeddings will be filtered.
        Args:
            embedding_path: the path of the pretrained embedding file
        Returns:
            embeddings: a torch tensor 
        """
        trained_embeddings = {}
        with open(embedding_path, 'r') as fin:
            for id_, line in enumerate(fin):
                # pretraind embeddings的内容为：每行以token内容开头，后面是token的embed向量，以空格分割
                contents = line.strip().split()
                token = contents[0]
                token_embed = contents[1:]

                if token not in self.token2id:
                    self.add(token)
                try:
                    if len(np.asarray(token_embed, dtype='float32')) == embedding_dim:
                        trained_embeddings[token] = np.asarray(
                            token_embed, dtype='float32')
                    else:
                        pass
                except ValueError:
                    pass

        # 初始化的embedding与trained_embedding有相同的均值和方差，这里的tokens数目是trained_token和其train集上的token.
        all_trained_embeddings = np.asarray(
            list(trained_embeddings.values()))  # asarray，随values的内容变化
        embeddings_mean = float(np.mean(all_trained_embeddings))
        embedding_std = float(np.std(all_trained_embeddings))
        # 使用与trained embedding一样的均值和方差进行初始化
        self.embeddings = torch.FloatTensor(
            self.size_of_token(), embedding_dim).normal_(embeddings_mean, embedding_std)
        # 用trained embeddings初始化，而后继续使用train set进行添加
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = torch.FloatTensor(
                    trained_embeddings[token])
            else:
                pass  # pretrain中不包含的跳过
        return self.embeddings

    def convert_to_ids(self, tokens):
        """
        Convert a list of tokens to ids, use unk_token if the token is not in vocab.
        Args:
            tokens: a list of token
        Returns:
            a list of ids
        """
        vec = [self.get_id(label) for label in tokens]
        return vec

    def recover_from_ids(self, ids, stop_id=None):
        """
        Convert a list of ids to tokens, stop converting if the stop_id is encountered
        Args:
            ids: a list of ids to convert
            stop_id: the stop id, default is None
        Returns:
            a list of tokens
        """
        tokens = []
        for i in ids:
            tokens += [self.get_token(i)]
            if stop_id is not None and i == stop_id:
                break
        return tokens
