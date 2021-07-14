import json
import spacy
import numpy as np
import torch

nlp = spacy.blank("en")  # 创建空语言模型


def word_tokenize(sent):
    '''Tokenize the document.
    Use spacy tools to tokenize.

    Args:
        sent: the doc that you want to tokenize.
    Returns:
        a list that contains the tokenized item.
    '''
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_span(text, tokens):
    '''Convert to the token's position in text.
    Map the token index to the text index. 
    And return the tuple showing start and end of the token.

    Args:
        text: the text string
        tokens: the token list 

    Returns:
        spans: a list of tuples that contains the start and end index of the tokens.
        'spam and spam are there' returns [(0,4),(5,8),(9,13),(14,17),(18,23)]
    '''
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


class SQuAD:
    '''Process the SQuAD dataset.

    Functions:
        process_file(self, filename)
        _get_mini_batch(self, examples)
        get_batches(self, set_name, batch_size)

    Attributes:
        batch_size, 
        para_limit,
        char_limit,
        question_limit,
        token_vocab,
        char_vocab,
        train_path,
        dev_path.
    '''

    def __init__(self, args, token_vocab, char_vocab):
        self.batch_size = args.batch_size
        self.para_limit = args.para_limit  # 文章的最大长度(以token数计)
        self.char_limit = args.char_limit  # 单词的最大长度限制
        self.question_limit = args.question_limit  # 问题的最大长度
        self.token_vocab = token_vocab  # 对token进行编码
        self.char_vocab = char_vocab
        self.train_path = args.train_file
        self.dev_path = args.dev_file

    def process_file(self, filename):
        '''
        The file is a json file.
        The structure is:
        data
        ----title
        ----paragraphs
            ----context
            ----qas
                ----answers
                    ----answer_start
                    ----text
                ----question
                ----id

        Args:
            filename: the file's path.
        '''
        examples = []
        total = 0
        with open(filename, "r") as fh:
            source = json.load(fh)
            for article in enumerate(source["data"]):
                for para in article["paragraphs"]:
                    context = para["context"].replace(
                        "''", '" ').replace("``", '" ')
                    context_tokens = word_tokenize(context)  # 断开文本
                    if len(context_tokens) < self.para_limit:
                        context_chars = [list(token)
                                         for token in context_tokens]  # 将词断为字符，每个词为一个嵌套列表
                        spans = convert_span(
                            context, context_tokens)  # 获取词的span
                        for qa in para["qas"]:
                            total += 1
                            ques = qa["question"].replace(
                                "''", '" ').replace("``", '" ')
                            ques_tokens = word_tokenize(ques)
                            # 对question进行如上操作
                            ques_chars = [list(token) for token in ques_tokens]
                            y1s = []  # answer的token_idx开头
                            y2s = []  # answer的token_idx结尾
                            answer_texts = []  # 保存全部answer
                            for answer in qa["answers"]:
                                answer_text = answer["text"]
                                answer_start = answer['answer_start']
                                answer_end = answer_start + len(answer_text)
                                answer_texts.append(answer_text)
                                answer_span = []
                                for token_idx, span in enumerate(spans):
                                    if not (answer_end <= span[0] or answer_start >= span[1]):
                                        answer_span.append(token_idx)
                                    # 取answer_span之间的所有token_idx
                                y1, y2 = answer_span[0], answer_span[-1]
                                y1s.append(y1)
                                y2s.append(y2)
                            example = {"context_tokens": context_tokens, "context_chars": context_chars,
                                       "ques_tokens": ques_tokens,
                                       "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                            # 每一个问题为一个example
                            examples.append(example)
        return examples

    def _get_mini_batch(self, examples):
        '''
        Produce the batch that given to yield fuction.

        Args:
            examples: The example list produced by function process_file().
        Returns:
            batch:
            context_idxs, context_char_idxs, ques_idxs, ques_char_idxs, y1s, y2s, context_lens, ques_lens

        '''
        context_idxs = []  # 包含每个example对应的context的token ids
        ques_idxs = []  # 包含每个example对应的question的token ids
        context_char_idxs = []
        ques_char_idxs = []
        context_lens = []
        ques_lens = []
        y1s = []
        y2s = []
        ids = []
        for example in examples:
            # 初始化index矩阵
            context_idx = np.zeros([self.para_limit], dtype=np.int32)
            ques_idx = np.zeros([self.question_limit], dtype=np.int32)
            context_char_idx = np.zeros([self.para_limit, self.char_limit])
            ques_char_idx = np.zeros([self.question_limit, self.char_limit])

            # 生成token ids矩阵
            for i, token in enumerate(example["context_tokens"]):
                if i == self.para_limit:
                    break  # 限制不能超出文章的最大长度
                context_idx[i] = self.token_vocab.get_id(token)
            context_idxs.append(context_idx)
            context_lens.append(len(context_idxs))
            for i, token in enumerate(example["ques_tokens"]):
                if i == self.question_limit:
                    break  # 限制不能超出问题的最大长度
                ques_idx[i] = self.token_vocab.get_id(token)
            ques_idxs.append(ques_idx)
            ques_lens.append(len(ques_idxs))

            # 处理charcter ids
            for i, token in enumerate(example["context_chars"]):
                if i == self.para_limit:
                    break  # 限制不能超出文章的最大长度
                for j, char in enumerate(token):
                    if j == self.char_limit:
                        break  # 限制不能超出单词的最大长度
                    context_char_idx[i, j] = self.char_vocab.get_id(
                        char, is_char=True)
            context_char_idxs.append(context_char_idx)

            for i, token in enumerate(example["ques_chars"]):
                if i == self.para_limit:
                    break  # 限制不能超出文章的最大长度
                for j, char in enumerate(token):
                    if j == self.char_limit:
                        break  # 限制不能超出单词的最大长度
                    ques_char_idx[i, j] = self.char_vocab.get_id(
                        char, is_char=True)
            ques_char_idxs.append(ques_char_idx)

            # train中的example只有一个答案
            start, end = example["y1s"][-1], example["y2s"][-1]
            y1s.append(start)
            y2s.append(end)
            ids.append(example["id"])

        # 创建batch
        batch = {}
        batch['context_idxs'] = torch.tensor(np.array(context_idxs))
        batch['context_char_idxs'] = torch.tensor(np.array(context_char_idxs))
        batch['ques_idxs'] = torch.tensor(np.array(ques_idxs))
        batch['ques_char_idxs'] = torch.tensor(np.array(ques_char_idxs))
        batch['y1s'] = torch.tensor(y1s)
        batch['y2s'] = torch.tensor(y2s)
        batch['context_lens'] = torch.tensor(context_lens)
        batch['ques_lens'] = torch.tensor(ques_lens)
        return batch

    def get_batches(self, set_name, batch_size):
        '''
        Produce an iterator that gives out one mini batch per time.

        Args:
            set_name: to decide it a train batch or a dev batch.
            batch_size
        '''
        if set_name == 'train':
            path = self.train_path
        elif set_name == 'dev':
            path = self.dev_path
        examples = self.process_file(path)
        samples = []
        for example in examples:
            samples.append(example)
            if len(samples) % batch_size == 0:
                yield self._get_mini_batch(samples)
